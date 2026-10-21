"""
StateSubmitTtsShard: ONE fan-out Map iteration = ONE tts shard against
the ITERATION'S OWN task token (2026-09-02). The owning loop resolved
its token with {tts_shard_tags} after checkpointing every shard's spec;
this handler re-reads that checkpoint by tag — the payload routed, the
checkpoint carries the data.

Writes NOTHING shared: iterations run concurrently, so the only write
here is tts_submit's per-(kind, tag) register row. The loop's fit leg
reads those rows back when the Map completes. A replayed iteration
(Map redrive) absorbs at the tts gateway on the same idempotency token
and its token is completed synchronously — redrive IS the reconcile.
"""
from __future__ import annotations

import logging

from lib import ad_common as C
from lib import loop_state
from lib import submit

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = C.read_json(*C.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    loop = event["loop"]                      # "synth" | "repair"
    tag = event["tag"]
    state = C.read_json(C.PROXY_BUCKET,
                        loop_state.checkpoint_key(prefix, loop))
    spec = state["shard_specs"][tag]          # a stale tag must RAISE
    if loop == "synth":
        cues = {int(k): v for k, v in C.read_json(
            C.PROXY_BUCKET, f"{prefix}/cues.json").items()}
        texts = [cues[k].strip() for k in spec["order"]]
    else:
        texts = [state["texts"][str(k)] for k in spec["order"]]
    submit.tts_submit(texts, event["task_token"], plan, tag,
                      seed=spec.get("seed", 1234),
                      attempt=spec.get("attempt", 0))
    logger.info("%s shard %s: %d transcript(s) submitted (attempt %d)",
                loop, tag, len(texts), spec.get("attempt", 0))
    return {"waiting": tag}
