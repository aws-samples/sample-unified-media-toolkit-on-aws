"""
lambda_function.py — root dispatcher.

Routes invocations to the correct state handler based on the "operation"
field. DynamoDB Stream invocations (Records[]) bypass operation routing
and go directly to HandleLockRelease.

Routing table
─────────────
  operation                   → handler module
  ──────────────────────────────────────────────────────────────────────
  StatePrepareDocuments       → states/prepare_documents/lambda_function.py
  StateAcquireLock            → states/acquire_lock/lambda_function.py
  StateIndexDocuments         → states/index_documents/lambda_function.py
  StateGetIndicesToRebuild    → states/get_indices_to_rebuild/lambda_function.py
  StateRebuildIndex           → states/rebuild_index/lambda_function.py
  StateReleaseLock            → states/release_lock/lambda_function.py
  StateDecomposeSearchRequest → states/decompose_search_request/lambda_function.py
  StateSearchIndex            → states/search_index/lambda_function.py
  StateRerankResults          → states/rerank_results/lambda_function.py
  (DDB Stream Records[])      → states/handle_lock_release/lambda_function.py

Design notes
────────────
  • Handlers are imported lazily so a cold start for Search does not pay
    the LanceDB connection cost incurred by IndexDocuments or RebuildIndex.
  • DDB Stream events are detected by the presence of "Records" in the event
    before checking "operation".
"""
from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def lambda_handler(event, context):
    # DynamoDB Stream invocations carry a "Records" list
    if "Records" in event:
        log.info("DDB Stream event — routing to HandleLockRelease")
        from states.handle_lock_release import lambda_function as handler
        return handler.lambda_handler(event, context)

    # EventBridge Step Functions execution status change events
    if event.get("source") == "aws.states" and event.get("detail-type") == "Step Functions Execution Status Change":
        log.info("EventBridge SFN status change event — routing to HandleExecutionFailure")
        from states.handle_execution_failure import lambda_function as handler
        return handler.lambda_handler(event, context)

    operation = event.get("operation")
    if not operation:
        raise ValueError("Missing 'operation' field in event")

    log.info("Routing: '%s'", operation)

    if operation == "StatePrepareDocuments":
        from states.prepare_documents import lambda_function as handler
    elif operation == "StateAcquireLock":
        from states.acquire_lock import lambda_function as handler
    elif operation == "StateIndexDocuments":
        from states.index_documents import lambda_function as handler
    elif operation == "StateGetIndicesToRebuild":
        from states.get_indices_to_rebuild import lambda_function as handler
    elif operation == "StateRebuildIndex":
        from states.rebuild_index import lambda_function as handler
    elif operation == "StateReleaseLock":
        from states.release_lock import lambda_function as handler
    elif operation == "StateDecomposeSearchRequest":
        from states.decompose_search_request import lambda_function as handler
    elif operation == "StateSearchIndex":
        from states.search_index import lambda_function as handler
    elif operation == "StateRerankResults":
        from states.rerank_results import lambda_function as handler
    else:
        raise ValueError(f"Unknown operation: '{operation}'")

    return handler.lambda_handler(event, context)
