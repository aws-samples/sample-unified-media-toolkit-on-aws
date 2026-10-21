"""
StateRenderAndStage (README.md §2, §8): per chunk, build the self-describing
WebVTT envelope and render it through the COPIED `prompt_render` module —
the identity-bearing step — then stage the whole record set as
`records.jsonl` under this run's own prefix. Always staged,
never inline (§8): one 15 k-token chunk is already ~60 KB against the
256 KB state-payload limit, and staging is uniform with the image-bearing
sibling workflows this design templates.

Rendering is deterministic (same plan → same bytes) — which is what makes
a redriven submit absorb into its own bedrock run, and a fresh
caller-token rerun re-render identical records under a NEW token.

record_id convention: `chunk-{index:03d}` — the join key from
response.jsonl lines back to plan chunks.
"""
from __future__ import annotations

import json
import logging

from lib import ca_common
from lib import envelope
from lib import prompt_render
from lib import transcript as transcript_mod

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = ca_common.read_json(
        *ca_common.parse_s3_uri(event["plan_s3uri"]))

    t_bucket, t_key = ca_common.parse_s3_uri(plan["transcript"]["s3uri"])
    normalized = transcript_mod.load_transcript(
        ca_common.read_text(t_bucket, t_key),
        transcript_format=plan["transcript"]["source_format"])
    sentences = normalized["sentences"]

    artifact = plan["prompt"]
    total = len(plan["chunks"])
    lines = []
    for chunk in plan["chunks"]:
        vtt = envelope.build_envelope(
            sentences, chunk,
            duration_sec=plan["transcript"]["duration_sec"],
            total_chunks=total)
        rendered = prompt_render.render(artifact, {"{{transcript}}": vtt})
        record = {"record_id": f"chunk-{chunk['index']:03d}",
                  "system": rendered["system"],
                  "messages": rendered["messages"]}
        lines.append(json.dumps(record, ensure_ascii=False,
                                separators=(",", ":")))

    records_key = f"{plan['output_prefix']}/records.jsonl"
    ca_common.write_text(ca_common.PROXY_BUCKET, records_key,
                         "\n".join(lines) + "\n")
    logger.info("staged %d record(s) at %s", len(lines), records_key)
    return {
        "uuid": event["uuid"],
        "plan_s3uri": event["plan_s3uri"],
        "records_s3uri": f"s3://{ca_common.PROXY_BUCKET}/{records_key}",
        "record_count": len(lines),
    }
