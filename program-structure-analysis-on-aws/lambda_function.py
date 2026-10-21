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
    "ValidateAndPlan": "states.validate_plan.lambda_function",
    "Assemble": "states.assemble.lambda_function",
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
