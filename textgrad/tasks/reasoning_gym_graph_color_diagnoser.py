"""Graph Color diagnoser (diagnostic mode ``graph_color_top1``), built like the FinQA diagnoser.

Every outcome class comes from the task's own checks, applied in the order the
scorer applies them: JSON parsing, object type, vertex coverage, allowed colors,
and edge constraints. The critic sees the raw parser error with an indexed
excerpt, or the concrete violated constraints. Designer priors (common JSON
and constraint failure patterns, generic advice) are given to the critic;
Reasoning Gym's own solver (greedy first-available coloring in vertex order)
is never described, and the reference coloring is not shown.
"""

import json
from typing import Any, Dict, List

from .finqa_diagnoser import _clamp01
from .reasoning_gym_diagnoser import DiagnoserSpec, outcome_anchor


GRAPH_COLOR_DESIGN_VERSION = "graph_color_top1_verifier_projection_designer_prior_v2"

GRAPH_COLOR_DIRECTIONS = (
    "missing_output",
    "json_format",
    "vertex_coverage",
    "color_domain",
    "edge_constraint",
    "other",
)

GRAPH_COLOR_DESCRIPTIONS = {
    "missing_output": "The response is empty.",
    "json_format": "The response does not parse as a JSON object mapping vertices to colors.",
    "vertex_coverage": "At least one vertex listed in the question has no color.",
    "color_domain": "At least one vertex uses a color outside the question's possible colors.",
    "edge_constraint": "At least one listed edge connects two vertices with the same color.",
    "other": "The checker rejected the answer for a reason the other checks do not capture.",
}

GRAPH_COLOR_FALLBACK_PATCHES = {
    "missing_output": "Keep the reasoning short and always finish with the complete JSON map of vertices to colors.",
    "json_format": (
        "Return only the JSON object, and check that it parses: every key quoted, no stray "
        "quote or character before the closing brace, and no trailing comma."
    ),
    "vertex_coverage": "Include every vertex listed in the question exactly once as a key of the JSON map.",
    "color_domain": "Use only the colors listed in the question.",
    "edge_constraint": (
        "Before answering, check every listed edge; if you change a vertex's color, use "
        "only listed colors and recheck every edge touching that vertex."
    ),
    "other": "Check the coloring against every requirement in the question before answering.",
}

GRAPH_COLOR_GUIDANCE = """- Common failure patterns: JSON that does not parse, for example a stray quote
  or character just before the closing brace, a trailing comma, unquoted keys,
  or text around the object; vertices missing from the map; colors outside the
  listed options; a listed edge whose endpoints share a color.
- Any coloring that satisfies the question is accepted; there is no single
  expected answer.
- Useful interventions make the final check explicit (every listed vertex
  present once, every value from the options, every listed edge checked) and
  keep the answer to the JSON object only."""

GRAPH_COLOR_CONSTRAINTS = {
    "non_empty_response": "The response must contain non-whitespace text.",
    "json_parses": "The whole response must parse as JSON.",
    "json_object": "The parsed value must be an object mapping vertices to colors.",
    "all_vertices_colored": "Every vertex listed in the question must have a color.",
    "colors_in_options": "Every color must be one of the question's possible colors.",
    "edges_properly_colored": "The two endpoints of every listed edge must have different colors.",
    "deterministic_answer_match": "The response must receive full credit from the deterministic checker.",
}


def _excerpt(text: str, position: int, width: int = 40) -> str:
    start, end = max(0, position - width), min(len(text), position + width)
    return f"chars {start}-{end}: {text[start:position]!r} <HERE> {text[position:end]!r}"


def _color_of(coloring: Dict[Any, Any], vertex: Any) -> Any:
    """Same key lookup as the Reasoning Gym graph_color verifier."""
    if vertex in coloring:
        return coloring[vertex]
    if isinstance(vertex, int) and str(vertex) in coloring:
        return coloring[str(vertex)]
    if isinstance(vertex, str):
        try:
            if int(vertex) in coloring:
                return coloring[int(vertex)]
        except ValueError:
            pass
    raise KeyError(vertex)


def _violations(coloring: Dict[Any, Any], puzzle: Dict[str, Any]) -> Dict[str, List[str]]:
    """All coverage, domain and edge violations; total over arbitrary JSON values."""
    vertices, edges = puzzle.get("vertices", []), puzzle.get("edges", [])
    allowed = set(puzzle.get("color_options", []))
    colors, missing, invalid, conflicts = {}, [], [], []
    for vertex in vertices:
        try:
            colors[vertex] = _color_of(coloring, vertex)
        except KeyError:
            missing.append(str(vertex))
    for vertex, color in colors.items():
        try:
            valid = color in allowed  # same membership test as the scorer
        except TypeError:  # list/object colors are unhashable; the scorer fails them too
            valid = False
        if not valid:
            invalid.append(f"{vertex}={color!r}")
    unchecked_edges = 0
    for edge in edges:
        try:
            u, v = edge
        except (TypeError, ValueError):
            unchecked_edges += 1  # malformed edge: report it as unchecked, never as passed
            continue
        if u in colors and v in colors:
            if colors[u] == colors[v]:
                conflicts.append(f"({u}, {v}) both {colors[u]!r}")
        else:
            unchecked_edges += 1
    return {"missing": missing, "invalid": invalid, "conflicts": conflicts, "unchecked_edges": unchecked_edges}


def build_graph_color_projection(
    question: str, prediction: str, entry: Dict[str, Any], score: float
) -> Dict[str, Any]:
    text = "" if prediction is None else str(prediction)
    puzzle = (entry.get("metadata") or {}).get("puzzle", {})
    observations: List[str] = []
    parsed, parse_ok, is_object, checked = None, False, False, False
    violations = {"missing": [], "invalid": [], "conflicts": [], "unchecked_edges": 0}

    if not text.strip():
        direction, interpretation = "missing_output", "No coloring was emitted."
    else:
        try:
            parsed = json.loads(text)
            parse_ok = True
        except json.JSONDecodeError as error:
            observations.append(
                f"json.loads failed: {error.msg} (line {error.lineno}, column {error.colno}, char {error.pos})"
            )
            observations.append("excerpt around the error position: " + _excerpt(text, error.pos))
        except (ValueError, RecursionError) as error:  # e.g. an integer beyond the digit limit
            observations.append(f"json.loads failed: {type(error).__name__}")
        is_object = parse_ok and isinstance(parsed, dict)
        if parse_ok and not is_object:
            observations.append(f"the parsed JSON value is a {type(parsed).__name__}, not an object")
        if not is_object:
            direction, interpretation = "json_format", "The response is not a JSON object mapping vertices to colors."
        else:
            try:
                violations = _violations(parsed, puzzle)
            except Exception as error:  # keep backward total; the scorer also maps errors to failure
                observations.append(f"the coloring could not be checked: {type(error).__name__}")
                direction, interpretation = "other", "The coloring could not be checked."
            else:
                checked = True
                # Same precedence as the scorer: coverage, then colors, then edges.
                if violations["missing"]:
                    direction, interpretation = "vertex_coverage", "Some listed vertices have no color."
                elif violations["invalid"]:
                    direction, interpretation = "color_domain", "Some vertices use colors outside the possible colors."
                elif violations["conflicts"]:
                    direction, interpretation = "edge_constraint", "Some listed edges join two vertices of the same color."
                else:
                    direction, interpretation = "other", "The checker rejected the answer for another reason."
                listed = {str(v) for v in puzzle.get("vertices", [])}
                extra = [str(k) for k in parsed if str(k) not in listed]
                for label, values in (("vertices without a color", violations["missing"]),
                                      ("vertices with a color outside the options", violations["invalid"]),
                                      ("edges whose endpoints share a color", violations["conflicts"]),
                                      ("keys that are not listed vertices", extra)):
                    if values:
                        observations.append(f"{label}: {', '.join(values[:10])}" + (" ..." if len(values) > 10 else ""))
                if violations["unchecked_edges"]:
                    observations.append(f"{violations['unchecked_edges']} edges could not be checked because an endpoint has no color")

    def probe(name, failed, checked, detail):
        return {"name": name, "failed": failed, "status": "checked" if checked else "not_checked",
                "detail": detail if checked else "prerequisite check failed"}

    # A check counts as complete only if it ran and covered every listed item.
    complete = checked and not violations["missing"]
    edges_complete = complete and not violations["unchecked_edges"]
    probes = [
        {"name": "non_empty_response", "failed": not text.strip(), "detail": f"response_length={len(text.strip())}"},
        probe("json_parses", bool(text.strip()) and not parse_ok, bool(text.strip()), "parsed" if parse_ok else "not parsed"),
        probe("json_object", parse_ok and not is_object, parse_ok, type(parsed).__name__),
        probe("all_vertices_colored", bool(violations["missing"]), checked, f"missing={len(violations['missing'])}"),
        probe("colors_in_options", bool(violations["invalid"]), complete or bool(violations["invalid"]),
              f"invalid={len(violations['invalid'])}"),
        probe("edges_properly_colored", bool(violations["conflicts"]), edges_complete or bool(violations["conflicts"]),
              f"conflicts={len(violations['conflicts'])}"),
        {"name": "deterministic_answer_match", "failed": score < 1.0, "detail": f"score={score}"},
    ]
    suppressed = [d for d in GRAPH_COLOR_DIRECTIONS if d != direction]
    return {
        "score": score,
        "loss": 1.0 - _clamp01(score),
        "error_type": direction,
        "score_interpretation": interpretation,
        "probes": probes,
        "failed_constraints": [p["name"] for p in probes if p["failed"]],
        "routing_anchor": outcome_anchor(direction, f"task_check_{direction}", suppressed),
        "observations": observations,
        "reference": "N/A: any coloring that satisfies the question is valid; no reference coloring is shown.",
        "patch_reference": "",
    }


GRAPH_COLOR_SPEC = DiagnoserSpec(
    task="graph_color",
    design_version=GRAPH_COLOR_DESIGN_VERSION,
    directions=GRAPH_COLOR_DIRECTIONS,
    descriptions=GRAPH_COLOR_DESCRIPTIONS,
    fallback_patches=GRAPH_COLOR_FALLBACK_PATCHES,
    constraints=GRAPH_COLOR_CONSTRAINTS,
    output_contract="Return the coloring as the JSON map of vertices to colors that the question requests.",
    target_requires_json=True,
    guidance=GRAPH_COLOR_GUIDANCE,
    project=build_graph_color_projection,
)
