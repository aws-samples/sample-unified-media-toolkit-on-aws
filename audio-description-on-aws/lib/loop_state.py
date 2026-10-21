"""
The re-entrant loop-state contract (settled 2026-08-31, replacing the
Next-step/Done?/Step 3-state shape): ONE waitForTaskToken Task + ONE
Choice per loop.

- The Task's Lambda reads its S3 checkpoint (the single source of
  truth), absorbs any completed sibling submission (pointers were saved
  at submit time — the callback payload is a SIGNAL, never a data
  channel), advances until something needs a sibling, submits with
  callback_task_token = this state's token, checkpoints, returns.
- The sibling's completion resolves the token; the Choice sees no
  `ad_loop_done` in the payload and re-enters the SAME state.
- When the loop is finished the Lambda resolves its OWN token with
  {"ad_loop_done": true} — the one payload shape we control — and the
  Choice exits. Sibling payload shapes never matter (absent field =
  continue), which is what makes two states enough.
- FAN-OUT extension (2026-09-02): a loop needing N INDEPENDENT tts
  shards resolves its own token with {"tts_shard_tags": [...]} instead;
  the Choice routes through an inline Map whose per-iteration
  waitForTaskToken Task submits ONE shard, so the shards run in
  parallel, each against its own token (a single token cannot wait for
  N callbacks — the first consumes it and the rest are TOKEN_GONE).
  Both control payloads are OURS; no Choice ever depends on a sibling's
  payload shape.
"""
from __future__ import annotations

import json

import boto3

from lib import ad_common as C

_sfn = boto3.client("stepfunctions")


def self_complete(task_token: str, extra: dict | None = None) -> None:
    """Resolve our own wait state: the loop is done."""
    _sfn.send_task_success(
        taskToken=task_token,
        output=json.dumps({"ad_loop_done": True, **(extra or {})}))


def request_fanout(task_token: str, tags: list) -> dict:
    """Resolve our own wait state into the loop's tts fan-out Map. The
    payload carries only the shard TAGS (control flow); every shard's
    spec lives in the checkpoint the Map's iterations re-read — the
    callback payload stays a signal, never a data channel."""
    _sfn.send_task_success(
        taskToken=task_token,
        output=json.dumps({"tts_shard_tags": sorted(tags)}))
    return {"waiting": f"fanout:{len(tags)}"}


def checkpoint_key(prefix: str, loop: str) -> str:
    return f"{prefix}/_plan/{loop}_state.json"


def reconcile_fanout(got: dict, state: dict, ck: str) -> list | None:
    """Turn collect_fanout's verdict into the loop's next move: None
    when every shard SUCCEEDED (proceed to fit/apply), else the FULL tag
    list to re-fan — landed shards absorb at the gateway and complete
    synchronously, so re-fanning everything is free and keeps this a
    single code path. FAILED shards with attempts left get their spec's
    attempt bumped and CHECKPOINTED FIRST, so the retry's -r{n}
    idempotency token survives a lost send (the fit leg then sees
    row.attempt < spec.attempt and re-fans without a second bump)."""
    if "outs" in got:
        return None
    for tag in got.get("retry", []):
        spec = state["shard_specs"][tag]
        spec["attempt"] = spec.get("attempt", 0) + 1
    if got.get("retry"):
        C.write_json(C.PROXY_BUCKET, ck, state)
    return list(state["shard_specs"])


def assert_fanout_checkpoint(state: dict, loop: str) -> None:
    """Refuse a pre-fan-out checkpoint BY NAME (review 2026-09-02): an
    execution that crossed the 2026-09-02 deploy mid-loop carries the
    serial keys (shards/shard/order/submit) and is also pinned to an ASL
    with no fan-out Map — neither redrive nor this Lambda can finish it.
    A named refusal beats a bare KeyError that looks like a code bug."""
    if "shard_specs" not in state:
        raise RuntimeError(
            f"{loop} checkpoint predates the 2026-09-02 shard fan-out "
            f"(serial keys {sorted(k for k in state if k != 'phase')}) — "
            "this execution started on an older deployment and cannot "
            "be redriven across it; re-run the request under the "
            "current TOOL_VERSION")
