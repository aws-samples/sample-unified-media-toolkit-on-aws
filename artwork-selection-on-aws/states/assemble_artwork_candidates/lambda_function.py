"""
AssembleArtworkCandidates — the product writer, AMENDS ONLY: applies
the vision verdict onto the slates FilterRank wrote (reorder within an
archetype by the ranking, attach unusable flags and the top reason —
never add, never re-qualify), and persists artwork_candidates.json. A
skipped/failed rerank ships the metadata order with the reason on
record: the slate is always delivered, the judgment layer is the
enhancement.
"""
from __future__ import annotations

import json
import logging
import re

from lib import s3io

logger = logging.getLogger(__name__)


def _verdict(rerank_result) -> tuple[dict | None, dict]:
    """(parsed verdict | None, rerank status block for the product)."""
    if not isinstance(rerank_result, dict):
        return None, {"status": "skipped", "reason": "no_result"}
    if rerank_result.get("skipped"):
        return None, {"status": "skipped",
                      "reason": rerank_result["skipped"]}
    status = rerank_result.get("status")
    out_prefix = rerank_result.get("output_s3_prefix")
    if status != "SUCCEEDED" or not out_prefix:
        return None, {"status": "failed", "reason": str(status),
                      "output_s3_prefix": out_prefix}
    try:
        texts = s3io.get_response_texts(out_prefix)
        m = re.search(r"\{.*\}", texts[0], re.DOTALL)
        return json.loads(m.group(0)), \
            {"status": "applied", "output_s3_prefix": out_prefix}
    except Exception as exc:                # noqa: BLE001 — degrade
        logger.warning("vision verdict unreadable: %s", exc)
        return None, {"status": "failed",
                      "reason": f"verdict_unreadable: {str(exc)[:120]}",
                      "output_s3_prefix": out_prefix}


def _apply(cands: list, group: dict) -> dict:
    """Reorder one archetype's candidates by the vision ranking (file
    basenames); unranked candidates keep metadata order at the tail.
    A candidate may carry several frames of its one shot (the
    FR-informed grab): its rank is its BEST frame's, that frame
    becomes THE frame, and it is unusable only when every frame is.
    Returns the vision block for the product."""
    ranking = group.get("ranking") or []
    unusable = {u.get("file"): u.get("why")
                for u in group.get("unusable") or [] if u.get("file")}
    order = {f: i for i, f in enumerate(ranking)}

    def files(c):
        return [u.rsplit("/", 1)[-1]
                for u in (c.get("frames") or
                          ([c["frame"]] if c.get("frame") else []))]

    def best(c):
        fs = files(c)
        if not fs:
            return None, len(order)
        f = min(fs, key=lambda f: order.get(f, len(order)))
        return f, order.get(f, len(order))

    cands.sort(key=lambda c: best(c)[1])
    for c in cands:
        f, _ = best(c)
        if f is None:
            continue
        if c.get("frames"):
            by_name = {u.rsplit("/", 1)[-1]: u for u in c["frames"]}
            c["frame"] = by_name[f]
        whys = [unusable.get(x) for x in files(c)]
        if whys and all(whys):
            c["unusable"] = unusable.get(f) or whys[0]
    return {"ranking": ranking,
            "top_reason": group.get("top_reason"),
            "n_unusable": len(unusable)}


def lambda_handler(event: dict, context) -> dict:
    plan_s3uri = event["plan_s3uri"]
    doc = s3io.get_json(event["candidates_s3uri"])
    prefix = event["candidates_s3uri"].rsplit("/", 1)[0]

    verdict, rerank = _verdict(event.get("rerank_result"))
    vision_blocks = {}
    if verdict:
        for archetype, cands in doc["slates"].items():
            group = verdict.get(archetype)
            if isinstance(group, dict):
                vision_blocks[archetype] = _apply(cands, group)

    product = {
        "uuid": doc["uuid"],
        "plan_s3uri": plan_s3uri,
        "archetype_counts": doc["archetype_counts"],
        "cast_names": doc["cast_names"],
        "identity_source": doc.get("identity_source"),
        "identity_fallback": doc.get("identity_fallback") or {},
        "brief_amendments": doc["brief_amendments"],
        "spoiler_cut_sec": doc["spoiler_cut_sec"],
        # the rejection accounting rides to the caller: an empty or
        # thin slate always says why (eligibility clause counts +
        # eligible shots per tier), never a silent zero
        "eligibility": doc.get("eligibility"),
        "rerank": rerank,
        "slates": {a: {"candidates": cands,
                       "vision": vision_blocks.get(a)}
                   for a, cands in doc["slates"].items()},
    }
    product_s3uri = f"{prefix}/artwork_candidates.json"
    s3io.put_json(product_s3uri, product)
    return {"uuid": doc["uuid"],
            "artwork_candidates_s3uri": product_s3uri,
            "rerank": rerank["status"],
            "n_candidates": {a: len(c) for a, c in
                             doc["slates"].items()}}
