"""DuckDB-backed cache for the harness.

The harness is the validation and testing arm of the project. It does not
score papers for the end user. Instead it stress-tests the one dependency the filter's
accuracy rests on - can the retrieval process surface the study that disproves a claim the field already
knows was reversed? The harness runs that search over the curated gold slice (claims whose
disproving study is recorded in advance) and reports recall, so we can show the filter works
before trusting it on unseen papers.

This module is the harness's cache. It persists the expensive-to-fetch artifacts so a run can
be repeated offline and deterministically:
  * ``candidates`` holds fetched evidence records.
  * ``embeddings`` holds per-candidate vectors, tagged by model.
  * ``claim_retrieval`` records which candidates each claim's queries surfaced, per channel.
  * ``stance`` holds the per-claim stance verdict for each classified candidate.

Run outputs (metrics, reports) are written to files by ``reporting.report``. Only the
reusable cache lives here. The stance table also feeds the graph visualization. Similarity
is computed by comparing every row directly, not through a separate search index, since
the candidate pools are still small enough for that to be fast.

DuckDB is a declared dependency (see ``pyproject.toml``). If an editor flags the ``duckdb``
import below as unresolved, point its Python interpreter at the project's uv virtualenv.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from .schema import Candidate, Stance, StanceLabel

DEFAULT_DB = Path("data/cache/harness.duckdb")


class Store:
    def __init__(self, path: str | Path = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                ext_id           VARCHAR PRIMARY KEY,
                source           VARCHAR,
                doi              VARCHAR,
                title            VARCHAR,
                abstract         VARCHAR,
                pub_types        VARCHAR,   -- JSON list
                year             INTEGER,
                retracted_by     VARCHAR,   -- JSON list
                is_retraction_of VARCHAR    -- JSON list
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                ext_id VARCHAR,
                model  VARCHAR,
                vector FLOAT[],
                PRIMARY KEY (ext_id, model)
            );
            CREATE TABLE IF NOT EXISTS claim_retrieval (
                claim_id VARCHAR,
                channel  VARCHAR,
                ext_id   VARCHAR,
                rank     INTEGER,
                score    DOUBLE,
                PRIMARY KEY (claim_id, channel, ext_id)
            );
            CREATE TABLE IF NOT EXISTS stance (
                claim_id         VARCHAR,
                candidate_ext_id VARCHAR,
                stance           VARCHAR,
                confidence       DOUBLE,
                rationale        VARCHAR,
                condition_match  BOOLEAN,
                PRIMARY KEY (claim_id, candidate_ext_id)
            );
            """
        )

    # Candidates
    def upsert_candidates(self, candidates: list[Candidate]) -> None:
        for c in candidates:
            self.con.execute(
                """
                INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT (ext_id) DO UPDATE SET
                    source=excluded.source, doi=excluded.doi, title=excluded.title,
                    abstract=excluded.abstract, pub_types=excluded.pub_types,
                    year=excluded.year, retracted_by=excluded.retracted_by,
                    is_retraction_of=excluded.is_retraction_of
                """,
                [
                    c.ext_id, c.source, c.doi, c.title, c.abstract,
                    json.dumps(c.pub_types), c.year,
                    json.dumps(c.retracted_by), json.dumps(c.is_retraction_of),
                ],
            )

    def get_candidate(self, ext_id: str) -> Candidate | None:
        row = self.con.execute(
            "SELECT ext_id, source, doi, title, abstract, pub_types, year, "
            "retracted_by, is_retraction_of FROM candidates WHERE ext_id = ?",
            [ext_id],
        ).fetchone()
        return _row_to_candidate(row) if row else None

    def has_candidate(self, ext_id: str) -> bool:
        return self.con.execute(
            "SELECT 1 FROM candidates WHERE ext_id = ?", [ext_id]
        ).fetchone() is not None

    # Embeddings
    def upsert_embedding(self, ext_id: str, model: str, vector: list[float]) -> None:
        self.con.execute(
            "INSERT INTO embeddings VALUES (?,?,?) "
            "ON CONFLICT (ext_id, model) DO UPDATE SET vector=excluded.vector",
            [ext_id, model, vector],
        )

    def get_embedding(self, ext_id: str, model: str) -> list[float] | None:
        row = self.con.execute(
            "SELECT vector FROM embeddings WHERE ext_id = ? AND model = ?", [ext_id, model]
        ).fetchone()
        return list(row[0]) if row else None

    # Claim to retrieved candidates
    def record_retrieval(
        self, claim_id: str, channel: str, ranked_ext_ids: list[tuple[str, float]]
    ) -> None:
        self.con.execute(
            "DELETE FROM claim_retrieval WHERE claim_id = ? AND channel = ?",
            [claim_id, channel],
        )
        for rank, (ext_id, score) in enumerate(ranked_ext_ids, start=1):
            self.con.execute(
                "INSERT INTO claim_retrieval VALUES (?,?,?,?,?) "
                "ON CONFLICT DO NOTHING",
                [claim_id, channel, ext_id, rank, score],
            )

    def get_retrieval(self, claim_id: str, channel: str) -> list[str]:
        rows = self.con.execute(
            "SELECT ext_id FROM claim_retrieval WHERE claim_id = ? AND channel = ? ORDER BY rank",
            [claim_id, channel],
        ).fetchall()
        return [r[0] for r in rows]

    # Stance
    def upsert_stance(self, labels: list[StanceLabel]) -> None:
        for s in labels:
            self.con.execute(
                "INSERT INTO stance VALUES (?,?,?,?,?,?) "
                "ON CONFLICT (claim_id, candidate_ext_id) DO UPDATE SET "
                "stance=excluded.stance, confidence=excluded.confidence, "
                "rationale=excluded.rationale, condition_match=excluded.condition_match",
                [
                    s.claim_id, s.candidate_ext_id, s.stance.value,
                    s.confidence, s.rationale, s.condition_match,
                ],
            )

    def get_stance(self, claim_id: str) -> list[StanceLabel]:
        rows = self.con.execute(
            "SELECT claim_id, candidate_ext_id, stance, confidence, rationale, condition_match "
            "FROM stance WHERE claim_id = ?",
            [claim_id],
        ).fetchall()
        return [
            StanceLabel(
                claim_id=r[0], candidate_ext_id=r[1], stance=Stance(r[2]),
                confidence=r[3], rationale=r[4] or "", condition_match=r[5],
            )
            for r in rows
        ]

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _row_to_candidate(row: tuple) -> Candidate:
    return Candidate(
        ext_id=row[0], source=row[1], doi=row[2], title=row[3] or "", abstract=row[4] or "",
        pub_types=json.loads(row[5]) if row[5] else [],
        year=row[6],
        retracted_by=json.loads(row[7]) if row[7] else [],
        is_retraction_of=json.loads(row[8]) if row[8] else [],
    )
