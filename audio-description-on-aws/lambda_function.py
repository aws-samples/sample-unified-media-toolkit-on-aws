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
    "Understand": "states.understand.lambda_function",
    "PlanWindows": "states.plan_windows.lambda_function",
    "WriteScript": "states.write_script.lambda_function",
    "SynthFit": "states.synth_fit.lambda_function",
    "SubmitTtsShard": "states.submit_tts_shard.lambda_function",
    "RepairChain": "states.repair_chain.lambda_function",
    "Finalize": "states.finalize.lambda_function",
    "UsageTracking": "states.usage_tracking.lambda_function",
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
