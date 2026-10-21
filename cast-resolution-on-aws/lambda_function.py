"""
Workflow Lambda entry point — routes invocations to the correct state
handler via the `operation` field (the family shape; see
shot-segmentation for the canonical copy). An unknown operation
RAISES: a state pointed at a handler nobody decided on must fail
loudly, never pick a default. Handlers import lazily so one state's
cold start never pays another's import cost.
"""
from __future__ import annotations

import importlib

_OPERATIONS = {
    "ValidateSources": "states.validate_sources.lambda_function",
    "ReadOnscreenText": "states.read_onscreen_text.lambda_function",
    "BindPersons": "states.bind_persons.lambda_function",
    "ResolveAddressees": "states.resolve_addressees.lambda_function",
    "AdjudicateCast": "states.adjudicate_cast.lambda_function",
    "ApplyEnrichment": "states.apply_enrichment.lambda_function",
    # BACK-COMPAT (remove at GA): SFN executions keep the ASL they
    # started with while the Lambda updates in place — a run parked in
    # the old machine across the deploy must still find its operations
    # (review finding, 2026-09-05).
    "ValidateAndPlan": "states.validate_sources.lambda_function",
    "ScheduleReadsAndWait": "states.read_onscreen_text.lambda_function",
    "BindAndFinalize": "states.bind_persons.lambda_function",
}


def lambda_handler(event: dict, context) -> dict:
    operation = (event or {}).get("operation")
    module_path = _OPERATIONS.get(operation)
    if module_path is None:
        raise ValueError(
            f"unknown operation {operation!r} — one of {sorted(_OPERATIONS)}")
    if operation == "BindAndFinalize":
        # old-ASL payload shape -> new handler (key rename only)
        if "wait_result" in event and "reads_result" not in event:
            event = {**event, "reads_result": event["wait_result"]}
    return importlib.import_module(module_path).lambda_handler(
        event, context)
