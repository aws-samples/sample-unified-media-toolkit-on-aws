"""
StatePlanWindows: pure code — snap on-screen-text beats to the shots
that carry them (the Meridian title-card root cause, fixed UPSTREAM of
everything that consults the beats), then build windows + orientation.
"""
from __future__ import annotations

import logging

from lib import ad_common as C
from lib import evidence
from lib import oracles
from lib import windows as windows_mod

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    plan = C.read_json(*C.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    ev = evidence.fetch(plan["inputs"])
    notes = C.read_json(C.PROXY_BUCKET, f"{prefix}/watch_notes.json")
    docs = evidence.docs_by_id(ev["shot_docs"])
    tm = evidence.shot_times(ev["shots"])

    fixed = oracles.snap_text_beats(notes, docs, tm)
    if fixed:
        logger.info("on-screen-text beats: %d re-timed", len(fixed))
        C.write_json(C.PROXY_BUCKET, f"{prefix}/watch_notes.json", notes)

    out = windows_mod.build_windows(
        {"transcript": ev["transcript"], "scenes": ev["scenes"],
         "shots": ev["shots"], "people": ev["people"],
         "shot_docs": docs, "stem_activity": ev["stem_activity"]},
        notes, sung_vocal_policy=plan["sung_vocal_policy"])

    C.write_json(C.PROXY_BUCKET, f"{prefix}/windows.json", out["windows"])
    C.write_json(C.PROXY_BUCKET, f"{prefix}/orientation.json",
                 out["orientation"])
    logger.info("windows: %s", out["summary"])
    return {"uuid": plan["workflow_id"], "summary": out["summary"]}
