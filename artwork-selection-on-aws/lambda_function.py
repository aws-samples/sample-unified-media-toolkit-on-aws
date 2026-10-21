"""
Workflow Lambda entry point — routes invocations to the correct state
handler via the `operation` field (the family shape; see
shot-segmentation for the canonical copy). An unknown operation
RAISES: a state pointed at a handler nobody decided on must fail
loudly, never pick a default. Handlers import lazily so one state's
cold start never pays another's import cost.
"""
from __future__ import annotations

_OPERATIONS = {
    "ValidateAndCompilePlan": "states.validate_and_compile_plan.lambda_function",
    "TranslateBriefToPlanAmendments": "states.translate_brief_to_plan_amendments.lambda_function",
    "FilterRankCandidateShots": "states.filter_rank_candidate_shots.lambda_function",
    "GrabCandidateFrames": "states.grab_candidate_frames.lambda_function",
    "RerankFramesByVision": "states.rerank_frames_by_vision.lambda_function",
    "AssembleArtworkCandidates": "states.assemble_artwork_candidates.lambda_function",
}


def lambda_handler(event: dict, context) -> dict:
    operation = (event or {}).get("operation")
    module_path = _OPERATIONS.get(operation)
    if module_path is None:
        raise ValueError(
            f"unknown operation {operation!r} — one of {sorted(_OPERATIONS)}")
    import importlib
    return importlib.import_module(module_path).lambda_handler(
        event, context)
