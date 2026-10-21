"""
StateValidateAndPlan: everything that can refuse, refuses HERE — before
any inference dollar (the family rule). Resolves every asset's spans
(auto-trigger via lib/spans, or the caller's explicit spans), binds every
category's registry prompt, validates the model chain, indexes each
shot-segmentation run's window manifests by time, and persists ONE plan
document; states pass URIs, never bodies.

The Map unit is a SPAN (one span = one iteration item, grouped
SPANS_PER_BATCH per iteration): a span's work is bounded by the windows
overlapping it, and the <=IMAGES_PER_RECORD chunking happens at selection
time inside the iteration — planning does not need to know frame counts
(README.md §10).
"""
from __future__ import annotations

import json
import logging

from lib import fi_common
from lib import spans as spans_mod

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class FrameInspectionValidationError(Exception):
    """Refusal before any spend; the message names what to fix."""


def lambda_handler(event: dict, context) -> dict:
    workflow_id = event["uuid"]
    request = event.get("input") or {}
    assets = _normalize_assets(request)

    prompts = {}
    planned_assets = []
    for asset_idx, asset in enumerate(assets):
        windows = _index_windows(asset["shot_segmentation_s3uri"])
        planned_requests = []
        for req in asset["requests"]:
            category = req.get("category")
            conf = fi_common.CATEGORIES.get(category)
            if conf is None:
                raise FrameInspectionValidationError(
                    f"unknown category {category!r} — one of "
                    f"{sorted(fi_common.CATEGORIES)}")
            if category not in prompts:
                # plan["prompts"] stays keyed by the REQUEST category
                # (the tool's API vocabulary); the registry category is
                # the owner-prefixed taxonomy name (fi-{task}, renamed
                # 2026-08-31 to match the ad-* convention)
                prompts[category] = _fetch_prompt(conf["prompt_category"])
            resolved = _resolve_spans(asset, req, conf)
            planned_requests.append({"category": category,
                                     "spans": resolved})
        planned_assets.append({
            "asset_idx": asset_idx,
            "shot_segmentation_s3uri": asset["shot_segmentation_s3uri"],
            "windows": windows,
            "requests": planned_requests,
        })

    use_batch = bool(request.get("use_batch", True))
    model_chain = list(request.get("model_chain")
                       or fi_common.default_model_chain(use_batch))

    units = _units(planned_assets)
    if not units:
        raise FrameInspectionValidationError(
            "no spans to inspect — every auto-trigger came up empty and no "
            "explicit spans were given; nothing here would spend a dollar "
            "usefully")

    prefix = fi_common.run_prefix(workflow_id)
    plan = {
        "workflow_id": workflow_id,
        "tool_version": fi_common.TOOL_VERSION,
        "output_prefix": prefix,
        "assets": planned_assets,
        "prompts": prompts,          # full artifact snapshots, per category
        "model_chain": model_chain,
        "use_batch": use_batch,
        "scope": (request.get("history_scope") or "").strip()
                 or f"frame-inspection.{workflow_id}",
        "units": units,
        "images_per_record": fi_common.IMAGES_PER_RECORD,
    }
    plan_key = f"{prefix}/_plan/plan.json"
    fi_common.write_json(fi_common.PROXY_BUCKET, plan_key, plan)

    batches = [{"batch_index": i // fi_common.SPANS_PER_BATCH,
                "unit_indices": list(range(
                    i, min(i + fi_common.SPANS_PER_BATCH, len(units))))}
               for i in range(0, len(units), fi_common.SPANS_PER_BATCH)]
    logger.info("planned %d asset(s), %d unit(s), %d batch(es)",
                len(planned_assets), len(units), len(batches))
    return {"uuid": workflow_id,
            "plan_s3uri": f"s3://{fi_common.PROXY_BUCKET}/{plan_key}",
            "batches": batches}


def _normalize_assets(request: dict) -> list:
    """Accept the pooled form (assets: [...]) and the single-asset sugar
    (shot_segmentation_s3uri + category at the top level) — one shape out."""
    if request.get("assets"):
        assets = request["assets"]
    elif request.get("shot_segmentation_s3uri"):
        assets = [{
            "shot_segmentation_s3uri": request["shot_segmentation_s3uri"],
            "shot_metadata_s3uri": request.get("shot_metadata_s3uri"),
            "requests": [{"category": request.get("category"),
                          "spans": request.get("spans")}],
        }]
    else:
        raise FrameInspectionValidationError(
            "either assets[] or a top-level shot_segmentation_s3uri is "
            "required")
    for asset in assets:
        if not asset.get("shot_segmentation_s3uri"):
            raise FrameInspectionValidationError(
                "every asset needs shot_segmentation_s3uri")
        if not asset.get("requests"):
            raise FrameInspectionValidationError(
                "every asset needs at least one request {category, ...}")
    return assets


def _resolve_spans(asset: dict, req: dict, conf: dict) -> list:
    explicit = req.get("spans")
    if explicit:
        for s in explicit:
            if not isinstance(s.get("start"), (int, float)) \
                    or not isinstance(s.get("end"), (int, float)) \
                    or s["end"] <= s["start"]:
                raise FrameInspectionValidationError(
                    f"bad span {s!r} — needs numeric start < end")
            if conf["needs_question"] and not (s.get("question") or "").strip():
                raise FrameInspectionValidationError(
                    f"{req['category']} spans each need a question — "
                    f"span {s.get('id') or s['start']} has none")
            if s.get("crop") is not None \
                    and s["crop"] not in fi_common.CROP_MODES:
                raise FrameInspectionValidationError(
                    f"span {s.get('id') or s['start']}: crop must be one "
                    f"of {fi_common.CROP_MODES} — got {s['crop']!r}")
        return explicit
    if conf["needs_question"]:
        raise FrameInspectionValidationError(
            f"{req['category']} takes explicit spans only (each with its "
            "question); there is no auto-trigger for it")
    if conf["trigger"] is None:
        raise FrameInspectionValidationError(
            f"{req['category']} has no auto-trigger; pass spans")
    docs, times = _load_shot_metadata(asset)
    trigger = getattr(spans_mod, conf["trigger"])
    return trigger(docs, times)


def _load_shot_metadata(asset: dict):
    uri = asset.get("shot_metadata_s3uri")
    if not uri:
        raise FrameInspectionValidationError(
            "auto-trigger needs shot_metadata_s3uri (a shot-metadata-"
            "enrichment run prefix); pass explicit spans to skip it")
    bucket, key = fi_common.parse_s3_uri(uri.rstrip("/"))
    if not key.endswith("shot_metadata.jsonl"):
        key = f"{key}/shot_metadata.jsonl"
    docs = [json.loads(line) for line in
            fi_common.read_bytes(bucket, key).decode("utf-8").splitlines()
            if line.strip()]
    sbucket, skey = fi_common.parse_s3_uri(
        asset["shot_segmentation_s3uri"].rstrip("/"))
    shots = fi_common.read_json(sbucket, f"{skey}/embeddings/shots.json")
    times = {s["shot_id"]: (s["start_time"], s["end_time"])
             for s in shots["shots"]}
    return docs, times


def _index_windows(shotseg_s3uri: str) -> list:
    """Every window manifest's time range + artifact keys, so selection
    knows which windows a span touches without listing S3 again. Manifests
    are small; the frame_index arrays stay behind their own keys."""
    bucket, prefix = fi_common.parse_s3_uri(shotseg_s3uri.rstrip("/"))
    out = []
    for key in fi_common.list_keys(bucket, f"{prefix}/embeddings/"):
        if not key.endswith(".json") or "_win" not in key:
            continue
        if key.endswith(("shots.json", "recurrence.json")):
            continue
        m = fi_common.read_json(bucket, key)
        if not m.get("frame_index"):
            continue
        from fractions import Fraction
        rate = float(Fraction(m["frame_rate"]))
        out.append({
            "manifest_key": key,
            "npy_key": key[:-5] + ".npy",
            "tar_key": key[:-5] + "_frames.tar.gz",
            "bucket": bucket,
            "start": m["frame_index"][0] / rate,
            "end": m["frame_index"][-1] / rate,
        })
    if not out:
        raise FrameInspectionValidationError(
            f"no window manifests under {shotseg_s3uri!r}/embeddings/ — is "
            "this a shot-segmentation run prefix?")
    return sorted(out, key=lambda w: w["start"])


def _units(planned_assets: list) -> list:
    units = []
    for asset in planned_assets:
        for req in asset["requests"]:
            for span_idx, span in enumerate(req["spans"]):
                units.append({"asset_idx": asset["asset_idx"],
                              "category": req["category"],
                              "span_idx": span_idx})
    return units


def _fetch_prompt(category: str) -> dict:
    response = fi_common.invoke_gateway_tool(
        fi_common.PROMPT_REGISTRY_ARN, "prompt-registry___get_prompt",
        {"category": category})
    if response.get("error"):
        raise FrameInspectionValidationError(
            f"prompt-registry refused get_prompt({category!r}): "
            f"{response.get('message')}")
    return response
