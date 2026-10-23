"""A::B diagnoser (diagnostic mode ``ab_top1``), built like the FinQA diagnoser.

Outcome classes are mechanically verifiable from the response, the question's
own rewrite condition (two neighbor tokens whose '#' face each other) and the
TRAIN reference. Designer priors (common failure patterns, the net-count
invariant of the rules, generic advice) are given to the critic; Reasoning
Gym's own solver (``compute_steps``: rewrite the leftmost facing pair, then
restart the scan) is never described, and no reference trace is shown.
"""

import difflib
import re
from typing import Any, Dict, List

from .finqa_diagnoser import _clamp01
from .reasoning_gym_diagnoser import DiagnoserSpec, outcome_anchor


AB_DESIGN_VERSION = "ab_top1_outcome_projection_designer_prior_v2"
_TOKEN = r"(?:A#|#A|B#|#B)"
_TOKEN_RE = re.compile(_TOKEN)
# One contiguous token sequence: tokens separated only by whitespace.
_RUN_RE = re.compile(rf"{_TOKEN}(?:\s*{_TOKEN})*")
_TOKENS_ONLY_RE = re.compile(rf"\s*(?:{_TOKEN}\s*)*")
_FINAL_MARKER_RE = re.compile(
    rf"(?:final(?:\s+(?:state|answer|program))?|answer|result)\s*(?:is|:|=)?\s*[`*\"']*({_TOKEN}(?:\s*{_TOKEN})*)",
    flags=re.IGNORECASE,
)


def _final_state_candidate(text: str):
    """Return (tokens or None, how it was read, number of token sequences)."""
    runs = [_TOKEN_RE.findall(match.group(0)) for match in _RUN_RE.finditer(text)]
    if _TOKENS_ONLY_RE.fullmatch(text):
        return _TOKEN_RE.findall(text), "tokens_only", len(runs)
    markers = list(_FINAL_MARKER_RE.finditer(text))
    if markers:
        return _TOKEN_RE.findall(markers[-1].group(1)), "explicit_final_marker", len(runs)
    if len(runs) <= 1:
        return (runs[0] if runs else []), "single_sequence", len(runs)
    return None, "ambiguous", len(runs)

AB_DIRECTIONS = (
    "missing_output",
    "serialization",
    "nonterminal_output",
    "wrong_final_state",
)

AB_DESCRIPTIONS = {
    "missing_output": "The response contains no A::B token although the reference final state is not empty.",
    "serialization": (
        "The recovered tokens equal the reference final state (possibly the empty "
        "state), but the returned text is not exactly that space-separated sequence."
    ),
    "nonterminal_output": (
        "The returned sequence still contains two neighbor tokens whose '#' face "
        "each other, so the question's rewrite condition still applies."
    ),
    "wrong_final_state": (
        "The returned sequence has no neighbor pair left to rewrite but differs "
        "from the reference final state; the response does not show where the "
        "computation diverged."
    ),
}

AB_FALLBACK_PATCHES = {
    "missing_output": (
        "Keep the rewriting work compact and always finish with the final state "
        "of the program as space-separated tokens."
    ),
    "serialization": "Return only the final state as tokens separated by single spaces, with no other text.",
    "nonterminal_output": (
        "Before answering, check every neighbor pair of the final sequence and keep "
        "rewriting while any pair still has its '#' facing each other."
    ),
    "wrong_final_state": (
        "Keep the rewriting bookkeeping compact, and before answering check that the "
        "final state's counts of #A minus A# and of #B minus B# equal those of the "
        "original program; output only the final state."
    ),
}

AB_GUIDANCE = """- Common failure patterns: misreading on which side of a token the '#' stands;
  rewriting a pair whose '#' do not face each other; confusing which pairs vanish
  and which swap; dropping, duplicating, or reordering tokens that no rewrite
  touched; answering while a facing pair is still present; running out of
  output budget before stating the final state.
- Every rule either deletes a facing pair with the same letter or swaps a
  facing pair with different letters, so the net counts (#A minus A#) and
  (#B minus B#) never change. A final state whose net counts differ from the
  program's is inconsistent with legal rewrites (matching counts are necessary,
  not sufficient, for a correct answer).
- The output budget is limited and only the final state may be returned.
  Useful interventions keep the bookkeeping compact, verify the final state
  before answering, and keep the answer to the final token sequence only."""

AB_CONSTRAINTS = {
    "tokens_recovered": "The response must contain the A::B tokens of a non-empty final state.",
    "exact_token_sequence": "The response must be exactly the tokens separated by single spaces, with nothing else.",
    "terminal_state": "No two neighbor tokens in the returned sequence may have their '#' facing each other.",
    "matches_reference": "The returned tokens must equal the reference final state.",
    "deterministic_answer_match": "The response must receive full credit from the deterministic checker.",
}


def _rewritable_positions(tokens: List[str]) -> List[int]:
    return [
        index
        for index, (left, right) in enumerate(zip(tokens, tokens[1:]))
        if left.endswith("#") and right.startswith("#")
    ]


def _net_counts(tokens: List[str]):
    return (tokens.count("#A") - tokens.count("A#"), tokens.count("#B") - tokens.count("B#"))


def _program_tokens(question: str):
    match = re.search(r"following program:\s*(.*?)\s*Return the final state", str(question), re.S)
    if not match:
        return None
    tokens = match.group(1).split()
    return tokens if tokens and all(_TOKEN_RE.fullmatch(t) for t in tokens) else None


def _alignment(reference: List[str], tokens: List[str], limit: int = 8) -> str:
    """One minimal token alignment; repeated tokens can make it ambiguous."""
    edits = []
    for op, r1, r2, p1, p2 in difflib.SequenceMatcher(None, reference, tokens, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        ref_part = " ".join(reference[r1:r2]) or "-"
        resp_part = " ".join(tokens[p1:p2]) or "-"
        edits.append(f"{op} reference[{r1}:{r2}] '{ref_part}' -> response[{p1}:{p2}] '{resp_part}'")
    if len(edits) > limit:
        edits = edits[:limit] + [f"... {len(edits) - limit} more edits"]
    return "; ".join(edits) or "no token differences"


def build_ab_projection(question: str, prediction: str, entry: Dict[str, Any], score: float) -> Dict[str, Any]:
    answer = entry.get("answer")
    reference = "" if answer is None else str(answer)
    reference_tokens = reference.split()
    text = "" if prediction is None else str(prediction)
    # A response of tokens and whitespace is one candidate; with surrounding prose
    # an explicit final-answer marker wins; several unmarked sequences are ambiguous.
    candidate, read_as, n_runs = _final_state_candidate(text)
    all_runs = [_TOKEN_RE.findall(m.group(0)) for m in _RUN_RE.finditer(text)]
    tokens = candidate or []
    canonical = text == " ".join(tokens)  # the scorer compares the raw text exactly
    rewritable = _rewritable_positions(tokens) if candidate is not None else []
    observations = []
    authoritative = True

    if candidate is None and reference_tokens in all_runs:
        direction, reason = "serialization", "reference_sequence_present"
        interpretation = "The reference final state appears in the response, but the response is not exactly that sequence."
    elif candidate is None:
        direction, reason, authoritative = None, "ambiguous_final_state", False
        interpretation = (
            "The response contains several token sequences and no explicit final answer, "
            "so which one is the final state is ambiguous."
        )
    elif tokens == reference_tokens:
        # Score < 1 here, so the raw text differs from the reference (this also
        # covers an empty reference state described in words or whitespace).
        direction, reason = "serialization", "tokens_match_reference"
        interpretation = "The tokens are correct, but the returned text is not exactly the requested sequence."
    elif not tokens:
        direction, reason = "missing_output", "no_recoverable_tokens"
        interpretation = "No A::B token was emitted although the final state is not empty."
    elif rewritable:
        direction, reason = "nonterminal_output", "rewrite_condition_still_applies"
        interpretation = "The returned sequence still contains a neighbor pair that the question says must be rewritten."
    else:
        direction, reason = "wrong_final_state", "terminal_but_different"
        interpretation = (
            "The returned sequence has nothing left to rewrite but is not the reference final state; "
            "the step that went wrong is not visible."
        )

    if not reference_tokens:
        observations.append("the reference final state is empty")
    if read_as == "explicit_final_marker":
        observations.append(
            f"the response contains {n_runs} token sequence(s); the one after the explicit final-answer marker is used"
        )
    elif candidate is None:
        observations.append(
            f"the response contains {n_runs} token sequences and no explicit final-answer marker; "
            "no single final state can be read"
        )
    total_tokens = sum(len(run) for run in all_runs)
    if candidate is not None and (tokens or reference_tokens):
        observations.append(
            f"final-state candidate has {len(tokens)} tokens; reference final state has {len(reference_tokens)} tokens"
        )
    if text and not canonical:
        observations.append(
            "the response is not exactly the tokens separated by single spaces "
            "(it has other characters or different spacing)"
        )
    if rewritable:
        pairs = ", ".join(f"positions {i}-{i + 1} '{tokens[i]} {tokens[i + 1]}'" for i in rewritable[:5])
        observations.append(f"neighbor pairs whose '#' still face each other: {pairs}")
    if tokens and candidate is not None:
        program = _program_tokens(question)
        if program is not None:
            observations.append(
                "net counts (#A minus A#, #B minus B#): program "
                f"{_net_counts(program)}, response {_net_counts(tokens)}, reference {_net_counts(reference_tokens)}"
            )
    if tokens and tokens != reference_tokens:
        observations.append(
            "token alignment against the reference (one of possibly several): "
            + _alignment(reference_tokens, tokens)
        )

    identified = candidate is not None
    probes = [
        {"name": "tokens_recovered", "failed": total_tokens == 0 and bool(reference_tokens),
         "detail": f"tokens_in_response={total_tokens}; final_state_candidate="
                   + (f"{len(tokens)} tokens" if identified else "not identified")},
        {"name": "exact_token_sequence", "failed": not canonical,
         "detail": "exactly single-space-separated tokens" if canonical else "other characters or spacing"},
        {"name": "terminal_state", "failed": bool(rewritable),
         "status": "checked" if identified and tokens else "not_checked",
         "detail": f"rewritable_neighbor_pairs={len(rewritable)}" if identified and tokens else "no final state to check"},
        {"name": "matches_reference", "failed": identified and tokens != reference_tokens,
         "status": "checked" if identified else "not_checked",
         "detail": ("token content equals reference" if tokens == reference_tokens else "token content differs")
                   if identified else "no final state identified"},
        {"name": "deterministic_answer_match", "failed": score < 1.0, "detail": f"score={score}"},
    ]
    if authoritative:
        anchor = outcome_anchor(direction, reason, [d for d in AB_DIRECTIONS if d != direction])
    else:
        anchor = {"primary_direction": None, "primary_direction_authoritative": False,
                  "anchor_reason": reason, "suppressed_directions": []}
    return {
        "score": score,
        "loss": 1.0 - _clamp01(score),
        "error_type": direction or "ambiguous_final_state",
        "score_interpretation": interpretation,
        "probes": probes,
        "failed_constraints": [p["name"] for p in probes if p["failed"]],
        "routing_anchor": anchor,
        "observations": observations,
        "reference": reference if reference else "(empty final state)",
        # Copies of the reference in a patch are flagged, not removed. Two-token
        # references collide with restatements of the question's own rules.
        "patch_reference": reference if len(reference_tokens) >= 3 else "",
    }


AB_SPEC = DiagnoserSpec(
    task="ab",
    design_version=AB_DESIGN_VERSION,
    directions=AB_DIRECTIONS,
    descriptions=AB_DESCRIPTIONS,
    fallback_patches=AB_FALLBACK_PATCHES,
    constraints=AB_CONSTRAINTS,
    output_contract="Return only the final state of the program as A::B tokens separated by single spaces.",
    target_requires_json=False,
    guidance=AB_GUIDANCE,
    project=build_ab_projection,
)
