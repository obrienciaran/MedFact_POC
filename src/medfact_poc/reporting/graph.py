"""Evidence graph visualization.

Turns stored run data into an interactive network. Claims and evidence papers are
nodes. A claim links to each evidence item the stance step judged, and the edge colour
shows the verdict (refutes, supports, neutral). A study known to disprove a claim is drawn as
a red dot when the search found it, and as a dark-yellow triangle joined by a dashed "not
retrieved" link when the search missed it, so recall gaps are visible at a glance. Retraction
links between papers are drawn as well.

``build_graph_data`` is pure and reads only from the store, so it is unit tested without
a browser. ``render_html`` writes a standalone interactive HTML file built directly on
vis-network (loaded from CDN), so there is no Python rendering dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..schema import Candidate, GoldSet, PaperVerdict
from ..store import Store

NodeGroup = Literal["claim", "evidence"]
EdgeKind = Literal["stance", "retraction", "missing"]


class GraphNode(BaseModel):
    id: str
    label: str
    group: NodeGroup
    title: str  # hover tooltip
    is_answer_key: bool = False
    status: str | None = None  # claim status, for claim nodes
    retrieved: bool = True  # False only for answer keys that were never retrieved at all


class GraphEdge(BaseModel):
    source: str
    target: str
    kind: EdgeKind
    stance: str | None = None
    label: str = ""
    confidence: float | None = None  # stance edges only


class GraphData(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def _truncate(text: str, length: int = 40) -> str:
    text = text.strip()
    return text if len(text) <= length else text[: length - 1] + "…"


def _claim_node_id(claim_id: str) -> str:
    return f"claim:{claim_id}"


def _evidence_node_id(ext_id: str) -> str:
    return f"ev:{ext_id}"


def _evidence_title(c: Candidate) -> str:
    parts = [f"Evidence: {c.title or c.ext_id}"]
    if c.year:
        parts.append(f"Year: {c.year}")
    if c.pub_types:
        parts.append("Study type: " + ", ".join(c.pub_types))
    parts.append(f"Evidence strength: {c.evidence_tier:.2f}")
    parts.append(f"PMID: {c.ext_id}")
    if c.retracted_by:
        parts.append("Retracted by: " + ", ".join(c.retracted_by))
    if c.is_retraction_of:
        parts.append("Retraction notice for: " + ", ".join(c.is_retraction_of))
    return "\n".join(parts)


def build_graph_data(gold: GoldSet, store: Store) -> GraphData:
    """Assemble nodes and edges from the gold set and stored stance results."""
    answer_key_ids = {ak for entry in gold.entries for ak in entry.answer_key}
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()
    evidence_candidates: dict[str, Candidate] = {}

    def add_edge(
        source: str, target: str, kind: EdgeKind, stance: str | None, label: str,
        confidence: float | None = None,
    ) -> None:
        key = (source, target, kind)
        if key in edge_keys:
            return
        edge_keys.add(key)
        edges.append(GraphEdge(
            source=source, target=target, kind=kind, stance=stance, label=label, confidence=confidence,
        ))

    def ensure_evidence_node(c: Candidate) -> str:
        node_id = _evidence_node_id(c.ext_id)
        if node_id not in nodes:
            nodes[node_id] = GraphNode(
                id=node_id,
                label=_truncate(c.title or c.ext_id, 30),
                group="evidence",
                title=_evidence_title(c),
                is_answer_key=c.ext_id in answer_key_ids,
            )
            evidence_candidates[c.ext_id] = c
        return node_id

    for entry in gold.entries:
        claim_node_id = _claim_node_id(entry.id)
        nodes[claim_node_id] = GraphNode(
            id=claim_node_id,
            label=_truncate(entry.claim_text, 40),
            group="claim",
            title=(
                f"Claim: {entry.claim_text}\n"
                f"Status: {_STATUS_LABELS.get(entry.status.value, entry.status.value)}"
            ),
            status=entry.status.value,
        )

        classified: set[str] = set()
        for label in store.get_stance(entry.id):
            cand = store.get_candidate(label.candidate_ext_id)
            if cand is None:
                continue
            node_id = ensure_evidence_node(cand)
            classified.add(cand.ext_id)
            add_edge(
                claim_node_id, node_id, "stance", label.stance.value,
                f"{label.stance.value} ({label.confidence:.2f})", confidence=label.confidence,
            )

        # Answer keys that were never retrieved or classified surface recall gaps.
        for ak in entry.answer_key:
            if ak in classified:
                continue
            node_id = _evidence_node_id(ak)
            if node_id not in nodes:
                nodes[node_id] = GraphNode(
                    id=node_id, label=ak, group="evidence",
                    title=(
                        f"Key evidence (PMID {ak})\n"
                        "This study is known to disprove the claim, but retrieval never found "
                        "it. That is a recall gap: the filter would miss this case."
                    ),
                    is_answer_key=True, retrieved=False,
                )
            add_edge(claim_node_id, node_id, "missing", None, "not retrieved")

    # Retraction links between evidence nodes that are both present. The link is reciprocal
    # (the flawed paper records "retracted by" and the notice records "retracts"), so we
    # draw a single arrow from the retraction notice to the paper it withdrew. Drawing both
    # directions produced two overlapping purple labels on top of each other.
    retraction_pairs: set[frozenset[str]] = set()

    def add_retraction(notice_id: str, paper_id: str) -> None:
        notice_node, paper_node = _evidence_node_id(notice_id), _evidence_node_id(paper_id)
        if notice_node not in nodes or paper_node not in nodes:
            return
        pair = frozenset({notice_node, paper_node})
        if pair in retraction_pairs:
            return
        retraction_pairs.add(pair)
        add_edge(notice_node, paper_node, "retraction", None, "retracts")

    for ext_id, c in evidence_candidates.items():
        for other in c.retracted_by:  # ext_id is the flawed paper, other is its notice
            add_retraction(other, ext_id)
        for other in c.is_retraction_of:  # ext_id is the notice, other is the flawed paper
            add_retraction(ext_id, other)

    return GraphData(nodes=list(nodes.values()), edges=edges)


def build_paper_graph_data(verdicts: list[PaperVerdict]) -> GraphData:
    """Assemble a graph from filter verdicts: each paper is a node coloured by its
    verdict, linked by a refutes edge to each work that refuted one of its claims."""
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    edge_keys: set[tuple[str, str]] = set()

    for v in verdicts:
        paper_id = f"paper:{v.pmid}"
        nodes[paper_id] = GraphNode(
            id=paper_id, label=_truncate(v.title or v.pmid, 40), group="claim",
            status=v.verdict.value,
            title=(
                f"Paper: {v.title or v.pmid}\nPMID: {v.pmid}\nVerdict: {v.verdict.value}\n"
                f"Score: {v.score:.2f}\nAction: {v.action.value}\nClaims: {v.n_claims}"
            ),
        )
        refuting = {p for cv in v.claim_verdicts for p in cv.refuting_pmids}
        supporting = {p for cv in v.claim_verdicts for p in cv.supporting_pmids} - refuting

        def add_evidence(pmid: str, *, refutes: bool) -> None:
            ev_id = _evidence_node_id(pmid)
            if ev_id not in nodes:
                role = "disproves" if refutes else "supports"
                nodes[ev_id] = GraphNode(
                    id=ev_id, label=pmid, group="evidence",
                    title=f"Evidence that {role} this paper\nPMID: {pmid}",
                    is_answer_key=refutes,  # refuting evidence is the red dot, support stays a plain dot
                )
            if (paper_id, ev_id) not in edge_keys:
                edge_keys.add((paper_id, ev_id))
                stance = "refutes" if refutes else "supports"
                edges.append(GraphEdge(source=paper_id, target=ev_id, kind="stance",
                                       stance=stance, label=stance))

        # Refuting evidence is the red dot/edge; supporting evidence is drawn green so a
        # supported paper is no longer a bare node. Neutral evidence is omitted on purpose,
        # because it is non-decisive and would bury the graph in dots.
        for pmid in refuting:
            add_evidence(pmid, refutes=True)
        for pmid in supporting:
            add_evidence(pmid, refutes=False)

    return GraphData(nodes=list(nodes.values()), edges=edges)


# Colour palette shared by the renderer. Stance colours are semantic and match the
# README: red refutes, green supports, grey neutral. Recall gaps (missing answer keys)
# are deliberately loud (amber) because surfacing them is the point of the harness.
_STANCE_COLOURS = {"refutes": "#e03131", "supports": "#2f9e44", "neutral": "#adb5bd"}
_MISSING_COLOUR = "#f08c00"
_RETRACTION_COLOUR = "#9c36b5"
# Evidence node colours. The known disproving study, when the search found it, is drawn RED
# (the same red as the "refutes" lines) so "this is the study that overturns the claim" reads
# instantly. When the search missed it, it becomes a dark-yellow warning triangle, because a
# recall gap is a warning and must stand out. Every other study the search found is a plain
# blue dot, and what that study says about the claim is carried by the colour of the line,
# not the dot.
_EVIDENCE_ANSWER_KEY = {"background": "#e03131", "border": "#a51111"}
_EVIDENCE_MISSING = {"background": "#e8a800", "border": "#9c7400"}
_EVIDENCE_PLAIN = {"background": "#4dabf7", "border": "#1971c2"}

# Stance explanations spelled out in full on hover, since the colour alone (and a bare
# confidence number) isn't enough context to act on.
_STANCE_EXPLANATIONS = {
    "refutes": "Evidence refutes the claim",
    "supports": "Evidence supports the claim",
    "neutral": "Evidence neither clearly supports nor refutes the claim",
}

# Claim/paper node colour by its status string. Covers gold-set statuses (reversed,
# still_true) and filter verdicts (refuted, contested, supported, unverified). Reversed
# claims are the ones under test (navy); still-true controls reuse the "supports" green so
# the colour itself signals "nothing wrong here", matching the stance edge palette.
_CLAIM_COLOURS = {
    "reversed": {"background": "#11335c", "border": "#081f3a"},
    "still_true": {"background": "#2b8a3e", "border": "#13631f"},
    "refuted": {"background": "#c92a2a", "border": "#7d1a1a"},
    "contested": {"background": "#e8590c", "border": "#9c3d05"},
    "supported": {"background": "#2b8a3e", "border": "#1a5828"},
    "unverified": {"background": "#495057", "border": "#2b3035"},
}
_CLAIM_DEFAULT = {"background": "#11335c", "border": "#081f3a"}


def _node_payload(node: GraphNode, control_only: bool) -> dict:
    """One vis-network node object with the extra fields the page's JS reads.

    ``name`` carries the full display name for the detail panel and for the hover reveal.
    Claim labels are drawn on the canvas always because they are the anchors. Evidence labels
    start blank (``label: ""``) and the page reveals them only for the focused
    neighbourhood, so dozens of evidence titles do not pile up into unreadable overlap.
    """
    if node.group == "claim":
        is_control = node.status == "still_true"
        colour = _CLAIM_COLOURS.get(node.status or "", _CLAIM_DEFAULT)
        # box shape sizes itself to the label; widthConstraint wraps long claims so the box
        # stays compact and the lines do not have to cross a very wide box to reach it.
        return {
            "id": node.id, "name": node.label, "label": node.label, "title": node.title,
            "group": "claim", "shape": "box", "margin": 10,
            # Heavier mass than the dots so the rectangular boxes push each other apart
            # harder. avoidOverlap treats a box as a circle and underestimates its width, so
            # extra mass is what actually stops the corners from clipping.
            "mass": 2.5,
            "widthConstraint": {"maximum": 150},
            "color": {**colour, "highlight": colour, "hover": colour},
            "font": {"color": "#ffffff", "size": 15, "face": "Inter, system-ui, sans-serif"},
            "controlFlag": is_control,
            "statusLabel": _STATUS_LABELS.get(node.status or "", node.status or "claim"),
        }
    if node.is_answer_key and not node.retrieved:
        # Missed disproving study: a dark-yellow triangle (a "gap" warning), a distinct shape
        # so the recall gap reads at a glance, with a dashed border to reinforce it.
        colour = _EVIDENCE_MISSING
        shape = "triangle"
        size = 18
        border_width = 2.5
        shape_extra = {"shapeProperties": {"borderDashes": [4, 3]}}
    elif node.is_answer_key:
        # Found disproving study: a red circle, the same red as the "refutes" lines, marking it
        # as the study that overturns the claim.
        colour = _EVIDENCE_ANSWER_KEY
        shape = "dot"
        size = 20
        border_width = 2
        shape_extra = {}
    else:
        colour = _EVIDENCE_PLAIN
        shape = "dot"
        size = 13
        border_width = 1.5
        shape_extra = {}
    # Recall-gap nodes carry their label from the start as the headline finding. All other
    # evidence labels stay blank until the node is focused, so the canvas is not cluttered.
    initial_label = node.label if (node.is_answer_key and not node.retrieved) else ""
    return {
        "id": node.id, "name": node.label, "label": initial_label, "title": node.title,
        "group": "evidence",
        "shape": shape,
        "size": size,
        "color": {**colour, "highlight": colour, "hover": colour},
        "borderWidth": border_width,
        "font": {"color": "#343a40", "size": 12, "face": "Inter, system-ui, sans-serif",
                 "strokeWidth": 3, "strokeColor": "#ffffff"},
        "isAnswerKey": node.is_answer_key,
        "isGap": node.is_answer_key and not node.retrieved,
        "retrieved": node.retrieved,
        "controlOnly": control_only,
        **shape_extra,
    }


def _edge_payload(edge: GraphEdge) -> dict:
    """One vis-network edge object. The hover title always spells out, in full sentences,
    what the edge means rather than leaving the user to decode a colour or a bare number."""
    base = {
        "from": edge.source, "to": edge.target, "arrows": "to",
        "etype": edge.kind, "stance": edge.stance, "smooth": {"type": "continuous", "roundness": 0.25},
    }
    if edge.kind == "retraction":
        # A single directed edge per pair (notice to paper), so its always-visible "retracts"
        # label does not collide with a reciprocal one. The arrow direction carries the rest:
        # the notice retracts the paper, the paper is retracted by the notice.
        title = ("This purple arrow runs from the retraction notice to the paper it formally "
                 "withdrew. The notice retracts the paper, and the paper is retracted by the notice.")
        return {
            **base, "title": title,
            "label": "retracts",
            "smooth": {"type": "curvedCW", "roundness": 0.3},  # bow the line clear of stance edges
            "font": {"color": _RETRACTION_COLOUR, "size": 11, "strokeWidth": 5,
                     "strokeColor": "#ffffff", "align": "top"},
            "color": {"color": _RETRACTION_COLOUR, "highlight": _RETRACTION_COLOUR,
                      "hover": _RETRACTION_COLOUR, "opacity": 0.9},
            "dashes": True, "width": 2,
        }
    if edge.kind == "missing":
        title = "Known refuting evidence for this claim, but retrieval never found it. A recall gap."
        colour = {"color": _MISSING_COLOUR, "highlight": _MISSING_COLOUR, "hover": _MISSING_COLOUR}
        return {**base, "title": title, "color": colour, "dashes": [6, 6], "width": 3}
    stance = edge.stance or "neutral"
    explanation = _STANCE_EXPLANATIONS.get(stance, "Evidence relates to the claim")
    title = f"{explanation} (model confidence {edge.confidence:.2f})" if edge.confidence is not None else explanation
    colour = _STANCE_COLOURS.get(stance, "#adb5bd")
    width = 3 if stance == "refutes" else 2
    return {**base, "title": title,
            "color": {"color": colour, "highlight": colour, "hover": colour}, "width": width}


def _control_only_evidence(data: GraphData) -> set[str]:
    """Evidence nodes reached only from still-true control claims.

    Used by the page's "hide controls" filter so an evidence node shared with a reversed
    claim is never hidden.
    """
    status_by_claim = {n.id: n.status for n in data.nodes if n.group == "claim"}
    claims_by_evidence: dict[str, set[str | None]] = {}
    for edge in data.edges:
        if edge.kind in ("stance", "missing") and edge.source in status_by_claim:
            claims_by_evidence.setdefault(edge.target, set()).add(status_by_claim[edge.source])
    return {
        ev for ev, statuses in claims_by_evidence.items()
        if statuses and statuses <= {"still_true"}
    }


# Display names for claim/paper node statuses, shared by the summary and legend.
_STATUS_LABELS = {
    "reversed": "Claim later overturned", "still_true": "Claim still accepted (control)",
    "refuted": "Refuted paper", "contested": "Contested paper",
    "supported": "Supported paper", "unverified": "Unverified paper",
}

DEFAULT_TITLE = "MedFact Evidence Graph"
DEFAULT_SUBTITLE = (
    "Results of the automated search to find the studies that disprove a wrong medical claim. "
    "That is the basis of this medical quality filter."
)

# Plain-English description of how the picture is built, shown under the subtitle. The two
# graphs answer different questions, so each gets its own accurate text passed to
# render_html. The default here describes the harness.
HARNESS_HOW_TO_READ = (
    "Each dark blue box is a claim the medical field later overturned. Each green box is a "
    "claim that still holds today. For every overturned claim, the system searches the medical "
    "literature and we check what it brings back. A plain blue dot is a study the search "
    "found. A red dot is a disproving study, found by the search. A yellow triangle is that "
    "known disproving study when the search failed to find it. The lines show how each study "
    "relates to the claim: red disproves it, green supports it, grey is neither."
    "<br><br>"
    "A language model is used in only two narrow places, to extract medical claims from a "
    "paper's text, and to judge whether a study disproves or supports it. The LLM does not "
    "decide which papers are kept or dropped. In this POC the claims and their known "
    "disproving studies here were chosen by hand."
)

FILTER_HOW_TO_READ = (
    "What this shows: the filter deciding whether each paper is trustworthy. Papers arrive as "
    "PubMed files. For each paper, the filter first reads the abstract and pulls out the claim "
    "it makes. It then gathers evidence on that claim from two places: the paper's own "
    "correction and retraction links, and a search of trusted medical literature. It reads how "
    "each study it gathered relates to the claim, whether it disproves it, supports it, or "
    "neither. Each box is a paper, coloured by the verdict that results. A red dot joined by a "
    "red line is a study that disproved the paper; a study that supported it is joined by a "
    "green line. Studies judged neutral are left off to keep the picture readable. The verdict "
    "tells downstream training to keep, "
    "down-weight, or drop the paper, and it is computed from the gathered evidence and how "
    "strong that evidence is. A language model is used in only two narrow places, to pull the "
    "claim out of the abstract and to judge each study's stance. It does not make the keep, "
    "down-weight or drop decision and it does not score the paper on its own."
)


def _stats_html(data: GraphData) -> str:
    """Summary stat cards, adapting to whichever node statuses are present."""
    from collections import Counter

    claim_status = Counter(n.status for n in data.nodes if n.group == "claim")
    evidence = [n for n in data.nodes if n.group == "evidence"]
    cards: list[tuple[int, str, str]] = []
    for status, n in claim_status.items():
        label = _STATUS_LABELS.get(status or "", status or "claim")
        cards.append((n, label, "gap" if status == "refuted" else ""))
    cards.append((len(evidence), "evidence", ""))
    answer_keys = sum(1 for n in evidence if n.is_answer_key)
    if answer_keys:
        cards.append((answer_keys, "disproving studies", ""))
    gaps = sum(1 for e in data.edges if e.kind == "missing")
    if gaps:
        cards.append((gaps, "recall gaps", "gap"))
    retractions = sum(1 for e in data.edges if e.kind == "retraction") // 2
    if retractions:
        cards.append((retractions, "retraction links", ""))
    return "".join(
        f'<div class="stat {cls}"><div class="n">{v}</div><div class="k">{label}</div></div>'
        for v, label, cls in cards
    )


def _legend_html(data: GraphData) -> str:
    """Legend items for the node statuses and kinds actually present in the graph."""
    statuses: list[str | None] = []
    for n in data.nodes:
        if n.group == "claim" and n.status not in statuses:
            statuses.append(n.status)
    items = []
    for status in statuses:
        c = _CLAIM_COLOURS.get(status or "", _CLAIM_DEFAULT)
        label = _STATUS_LABELS.get(status or "", status or "claim")
        items.append(
            f'<div class="legend-item"><span class="box" style="background:{c["background"]};'
            f'border:1px solid {c["border"]}"></span>{label}</div>'
        )
    if any(n.group == "evidence" and n.is_answer_key and n.retrieved for n in data.nodes):
        items.append(
            f'<div class="legend-item"><span class="dot" style="background:{_EVIDENCE_ANSWER_KEY["background"]};'
            f'border-color:{_EVIDENCE_ANSWER_KEY["border"]}"></span>Known disproving study, search found it</div>'
        )
    if any(n.group == "evidence" and n.is_answer_key and not n.retrieved for n in data.nodes):
        items.append(
            f'<div class="legend-item"><span class="tri" style="color:{_EVIDENCE_MISSING["background"]}">&#9650;</span>'
            "Known disproving study, search missed it</div>"
        )
    if any(n.group == "evidence" and not n.is_answer_key for n in data.nodes):
        items.append(
            f'<div class="legend-item"><span class="dot" style="background:{_EVIDENCE_PLAIN["background"]};'
            f'border-color:{_EVIDENCE_PLAIN["border"]}"></span>Other study found (line colour shows its stance)</div>'
        )
    return "".join(items)


def render_html(
    data: GraphData,
    out_path: str | Path,
    *,
    physics: bool = True,
    title: str = DEFAULT_TITLE,
    subtitle: str = DEFAULT_SUBTITLE,
    how_to_read: str = HARNESS_HOW_TO_READ,
) -> Path:
    """Render the graph to a standalone interactive HTML file.

    Built directly on vis-network (loaded from CDN) so the page owns its layout: a header,
    a data-driven legend and summary, edge-type filters, a "recall gaps only" view,
    hover-to-focus highlighting, and a click-for-details panel. No pyvis dependency.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    control_only = _control_only_evidence(data)
    nodes = [_node_payload(n, n.id in control_only) for n in data.nodes]
    edges = [_edge_payload(e) for e in data.edges]

    html = (
        _TEMPLATE
        .replace("__TITLE__", title)
        .replace("__SUBTITLE__", subtitle)
        .replace("__HOW_TO_READ__", how_to_read)
        .replace("__STATS_HTML__", _stats_html(data))
        .replace("__LEGEND_HTML__", _legend_html(data))
        .replace("__NODES__", json.dumps(nodes))
        .replace("__EDGES__", json.dumps(edges))
        .replace("__PHYSICS__", "true" if physics else "false")
    )
    out_path.write_text(html)
    return out_path


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MedFact Evidence Graph</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" crossorigin="anonymous" referrerpolicy="no-referrer" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<style>
  :root {
    --bg: #f1f3f5; --panel: #ffffff; --ink: #212529; --muted: #868e96;
    --line: #dee2e6; --accent: #1c3d5a;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
  #app { display: grid; grid-template-rows: auto 1fr; height: 100vh; }
  header { padding: 14px 22px; background: var(--accent); color: #fff; }
  header h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: .2px; }
  header p { margin: 3px 0 0; font-size: 12.5px; color: #c5d3df; }
  header p.how { margin-top: 8px; padding: 8px 11px; font-size: 12px; line-height: 1.55;
                 color: #e7eef4; background: rgba(255,255,255,.10); border-radius: 7px; max-width: 1000px; }
  main { display: grid; grid-template-columns: 250px 1fr; min-height: 0; }
  aside { background: var(--panel); border-right: 1px solid var(--line); overflow-y: auto; padding: 16px; }
  aside h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .8px; color: var(--muted); margin: 18px 0 8px; }
  aside h2:first-child { margin-top: 0; }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .stat { background: var(--bg); border-radius: 8px; padding: 8px 10px; }
  .stat .n { font-size: 19px; font-weight: 700; line-height: 1; }
  .stat .k { font-size: 10.5px; color: var(--muted); margin-top: 3px; }
  .stat.gap .n { color: #e8590c; }
  label.row { display: flex; align-items: center; gap: 8px; font-size: 13px; padding: 4px 0; cursor: pointer; }
  label.row input { accent-color: var(--accent); }
  .swatch { width: 22px; height: 0; border-top-width: 3px; border-top-style: solid; flex: 0 0 22px; }
  .swatch.dash { border-top-style: dashed; }
  .legend-item { display: flex; align-items: center; gap: 9px; font-size: 12.5px; padding: 3px 0; color: #495057; }
  .dot, .box, .star, .tri { flex: 0 0 16px; text-align: center; }
  .dot { flex: 0 0 14px; width: 14px; height: 14px; border-radius: 50%; border: 1px solid; }
  .box { width: 16px; height: 12px; border-radius: 2px; }
  .star { width: 16px; height: 0; font-size: 16px; line-height: 1; }
  .tri { width: 16px; height: 0; font-size: 14px; line-height: 1; }
  button { width: 100%; margin-top: 8px; padding: 8px; border: 1px solid var(--line); background: #fff; border-radius: 8px; font-size: 12.5px; cursor: pointer; }
  button:hover { background: var(--bg); }
  #canvas-wrap { position: relative; min-height: 0; }
  #net { width: 100%; height: 100%; background: radial-gradient(circle at 50% 40%, #ffffff 0%, #eef1f4 100%); }
  #detail { position: absolute; top: 14px; right: 14px; width: 280px; max-height: calc(100% - 28px); overflow-y: auto;
            background: rgba(255,255,255,.97); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
            box-shadow: 0 6px 24px rgba(0,0,0,.10); font-size: 12.5px; display: none; }
  #detail .tag { display: inline-block; font-size: 10px; text-transform: uppercase; letter-spacing: .5px; padding: 2px 7px;
                 border-radius: 99px; color: #fff; margin-bottom: 8px; }
  #detail h3 { margin: 0 0 8px; font-size: 13.5px; line-height: 1.4; }
  #detail pre { white-space: pre-wrap; word-break: break-word; font-family: inherit; color: #495057; margin: 0; line-height: 1.5; }
  #detail .close { float: right; cursor: pointer; color: var(--muted); border: none; background: none; width: auto; margin: 0; font-size: 16px; padding: 0; }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>__TITLE__</h1>
    <p>__SUBTITLE__</p>
    <p class="how">__HOW_TO_READ__</p>
  </header>
  <main>
    <aside>
      <h2>Summary</h2>
      <div class="stat-grid">__STATS_HTML__</div>

      <h2>Edge filters</h2>
      <label class="row"><input type="checkbox" id="f-refutes" checked><span class="swatch" style="border-top-color:#e03131"></span>Refutes</label>
      <label class="row"><input type="checkbox" id="f-supports" checked><span class="swatch" style="border-top-color:#2f9e44"></span>Supports</label>
      <label class="row"><input type="checkbox" id="f-neutral" checked><span class="swatch" style="border-top-color:#adb5bd"></span>Neutral</label>
      <label class="row"><input type="checkbox" id="f-missing" checked><span class="swatch dash" style="border-top-color:#f08c00"></span>Not retrieved</label>
      <label class="row"><input type="checkbox" id="f-retraction" checked><span class="swatch dash" style="border-top-color:#9c36b5"></span>Retraction</label>

      <h2>View</h2>
      <label class="row"><input type="checkbox" id="f-hidecontrols">Hide still-true controls</label>
      <label class="row"><input type="checkbox" id="f-onlygaps">Recall gaps only</label>
      <button id="btn-physics">Freeze layout</button>
      <button id="btn-fit">Reset view</button>

      <h2>Legend</h2>
      __LEGEND_HTML__
    </aside>
    <div id="canvas-wrap">
      <div id="net"></div>
      <div id="detail">
        <button class="close" id="detail-close">×</button>
        <span class="tag" id="detail-tag"></span>
        <h3 id="detail-title"></h3>
        <pre id="detail-body"></pre>
      </div>
    </div>
  </main>
</div>
<script>
  const rawNodes = __NODES__;
  const rawEdges = __EDGES__;
  const physicsOn = __PHYSICS__;

  rawEdges.forEach((e, i) => { e.id = "e" + i; });

  // Nodes touched by a "missing" (recall gap) edge, for the gaps-only view.
  const gapNodes = new Set();
  rawEdges.filter(e => e.etype === "missing").forEach(e => { gapNodes.add(e.from); gapNodes.add(e.to); });

  // Each node's own colour, kept so it can be re-asserted on every update. Without this, a
  // partial nodes.update (used to reveal/hide labels on click and hover) lets vis-network
  // repaint the nodes with its default node colour (a pale blue), which is exactly what made
  // the stance colours "all turn light blue" after a click. Re-pushing the colour on every
  // update keeps green green and navy navy no matter what is selected.
  const baseColor = {};
  rawNodes.forEach(n => { baseColor[n.id] = n.color; });

  const nodes = new vis.DataSet(rawNodes);
  const edges = new vis.DataSet(rawEdges);

  const options = {
    // chosen:false stops vis-network from recolouring a node or edge when it is clicked or
    // hovered. All focus styling is driven by our own code below, so a claim's colour (navy
    // for an overturned claim, green for a still-accepted one) never washes out to the
    // default pale-blue selection colour.
    nodes: { borderWidth: 1.5, chosen: false, shadow: { enabled: true, size: 4, x: 0, y: 1, color: "rgba(0,0,0,0.12)" } },
    edges: { smooth: { type: "continuous", roundness: 0.25 }, chosen: false },
    interaction: { hover: true, tooltipDelay: 120, navigationButtons: false, keyboard: false,
                   selectConnectedEdges: false, hoverConnectedEdges: false },
    physics: {
      // Strong repulsion, long springs and full overlap avoidance spread the nodes out so the
      // connecting lines do not pile on top of the claim boxes and bury their text.
      enabled: physicsOn,
      solver: "barnesHut",
      barnesHut: { gravitationalConstant: -11000, centralGravity: 0.30, springLength: 210, springConstant: 0.045, avoidOverlap: 1, damping: 0.85 },
      stabilization: { iterations: 1800, fit: true },
      minVelocity: 0.5,
    },
  };

  const container = document.getElementById("net");
  const network = new vis.Network(container, { nodes, edges }, options);

  // hover and select neighbourhood focus
  // Claim labels are always drawn. Evidence labels are blank by default and only revealed
  // for the focused node and its neighbours, so the canvas is not buried in overlapping
  // titles. A node's full name lives in n.name for both the reveal and the detail panel.
  // Claim labels and recall-gap (missed disproving study) labels are always drawn. The gap is a
  // single rare node and the headline finding, so it must not be hard to spot. Other
  // evidence labels appear only when their node is focused.
  const labelFor = (n, revealed) => (n.group === "claim" || n.isGap || revealed) ? n.name : "";
  function focusOn(nodeId) {
    const connected = new Set([nodeId]);
    network.getConnectedNodes(nodeId).forEach(id => connected.add(id));
    const connectedEdges = new Set(network.getConnectedEdges(nodeId));
    // Reveal the labels of the focused node and its neighbours. Node colours are left fully
    // solid (no opacity fade) so the verdict colours never wash out to a pale tint. Focus is
    // signalled by dimming the unrelated edges instead.
    nodes.update(rawNodes.map(n => ({
      id: n.id,
      label: labelFor(n, connected.has(n.id)),
      color: baseColor[n.id],  // re-assert so a click/hover never repaints the node pale blue
    })));
    edges.update(rawEdges.map(e => ({
      id: e.id,
      color: { ...e.color, opacity: connectedEdges.has(e.id) ? (e.color.opacity || 1) : 0.12 },
    })));
  }
  function clearFocus() {
    nodes.update(rawNodes.map(n => ({ id: n.id, label: labelFor(n, false), color: baseColor[n.id] })));
    edges.update(rawEdges.map(e => ({ id: e.id, color: e.color })));
  }
  network.on("hoverNode", p => focusOn(p.node));
  network.on("blurNode", clearFocus);

  // detail panel
  const detail = document.getElementById("detail");
  const tagColours = { claim: "#1c3d5a", evidence: "#1971c2" };
  function showDetail(node) {
    document.getElementById("detail-tag").textContent =
      node.group === "claim" ? (node.statusLabel || "claim")
                              : (node.isAnswerKey ? "disproving study" : "evidence");
    document.getElementById("detail-tag").style.background = tagColours[node.group];
    document.getElementById("detail-title").textContent = node.name || node.label;
    document.getElementById("detail-body").textContent = node.title || "";
    detail.style.display = "block";
  }
  network.on("click", p => {
    if (p.nodes.length) { showDetail(nodes.get(p.nodes[0])); focusOn(p.nodes[0]); }
    else { detail.style.display = "none"; clearFocus(); }
  });
  document.getElementById("detail-close").onclick = () => { detail.style.display = "none"; clearFocus(); };

  // filters
  function applyFilters() {
    const on = id => document.getElementById(id).checked;
    const kinds = {
      refutes: on("f-refutes"), supports: on("f-supports"), neutral: on("f-neutral"),
      missing: on("f-missing"), retraction: on("f-retraction"),
    };
    const hideControls = on("f-hidecontrols");
    const onlyGaps = on("f-onlygaps");

    nodes.update(rawNodes.map(n => {
      let hidden = false;
      if (hideControls && (n.controlFlag || n.controlOnly)) hidden = true;
      if (onlyGaps && !gapNodes.has(n.id)) hidden = true;
      return { id: n.id, hidden, color: baseColor[n.id] };  // keep the node colour on filter toggles
    }));
    edges.update(rawEdges.map(e => {
      const key = e.etype === "stance" ? e.stance : e.etype;
      let visible = kinds[key] !== false;
      if (onlyGaps && e.etype !== "missing") visible = false;
      return { id: e.id, hidden: !visible };
    }));
  }
  ["f-refutes", "f-supports", "f-neutral", "f-missing", "f-retraction", "f-hidecontrols", "f-onlygaps"]
    .forEach(id => document.getElementById(id).addEventListener("change", applyFilters));

  // buttons
  let frozen = !physicsOn;
  const physBtn = document.getElementById("btn-physics");
  physBtn.textContent = frozen ? "Resume layout" : "Freeze layout";
  physBtn.onclick = () => {
    frozen = !frozen;
    network.setOptions({ physics: { enabled: !frozen } });
    physBtn.textContent = frozen ? "Resume layout" : "Freeze layout";
  };
  document.getElementById("btn-fit").onclick = () => network.fit({ animation: true });

  // Run the force layout once to position the nodes, then switch physics off so the graph
  // settles and stops drifting. The user can hit "Resume layout" to re-run it.
  network.once("stabilizationIterationsDone", () => {
    frozen = true;
    network.setOptions({ physics: { enabled: false } });
    physBtn.textContent = "Resume layout";
    // Fit to frame the whole graph, then zoom in so the opening view is close, not a tiny
    // cluster lost in whitespace. Capped so a small graph does not over-magnify.
    network.fit({ animation: false });
    network.moveTo({ scale: Math.min(network.getScale() * 1.25, 1.1) });
  });
</script>
</body>
</html>
"""
