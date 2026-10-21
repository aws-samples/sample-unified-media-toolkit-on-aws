"""
StateCollectAndFinalize: response.jsonl -> per-asset per-category contract
products. Parsing uses the copied salvage ladder per output line;
schema checks are per category (claim-verify's verdict is validated
against its enum — an out-of-enum answer is a parse failure, never a
passed-through surprise). Every span that produced no usable output lands
in its product's gaps[] with a reason — honest gap, never fabricated;
partial success IS success. Raw model output is NEVER copied: the
manifest's inference_runs[] points at the bedrock run's own prefix.
"""
from __future__ import annotations

import json
import logging

from lib import fi_common
from lib import parse_partial_output

logger = logging.getLogger()
logger.setLevel(logging.INFO)

VERDICTS = ("supported", "contradicted", "indeterminate")


def lambda_handler(event: dict, context) -> dict:
    plan = fi_common.read_json(*fi_common.parse_s3_uri(event["plan_s3uri"]))
    prefix = plan["output_prefix"]
    runs = fi_common.read_json(fi_common.PROXY_BUCKET,
                               f"{prefix}/_plan/runs.json")
    outputs = _read_outputs(runs[0]["output_s3_prefix"])

    products = []
    for asset in plan["assets"]:
        for req in asset["requests"]:
            category = req["category"]
            spans_out, gaps = [], []
            for span_idx, span in enumerate(req["spans"]):
                parsed, why = _parse_span(
                    outputs, asset["asset_idx"], category, span_idx)
                if parsed is None:
                    gaps.append({"span": _span_ref(span, span_idx),
                                 "reason": why})
                else:
                    spans_out.append((span, span_idx, parsed))
            product = _assemble(category, spans_out, gaps)
            key = (f"{prefix}/{asset['asset_idx']}/"
                   f"{fi_common.CATEGORIES[category]['product']}")
            fi_common.write_json(fi_common.PROXY_BUCKET, key, product)
            products.append({"asset_idx": asset["asset_idx"],
                             "category": category,
                             "s3uri": f"s3://{fi_common.PROXY_BUCKET}/{key}",
                             "spans": len(req["spans"]),
                             "gaps": len(gaps)})

    manifest_key = f"{prefix}/inspection_manifest.json"
    fi_common.write_json(fi_common.PROXY_BUCKET, manifest_key, {
        "workflow_id": plan["workflow_id"],
        "tool_version": plan["tool_version"],
        "products": products,
        "units": plan["units"],
        "inference_runs": runs,
    })
    logger.info("finalized %d product(s)", len(products))
    return {"uuid": event["uuid"], "plan_s3uri": event["plan_s3uri"],
            "products": products}


def _read_outputs(bedrock_prefix: str) -> dict:
    """record_id -> output_text for every SUCCESS record."""
    bucket, key = fi_common.parse_s3_uri(
        bedrock_prefix.rstrip("/") + "/response.jsonl")
    out = {}
    for line in fi_common.read_bytes(bucket, key).decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "SUCCESS":
            out[row["record_id"]] = row.get("output_text") or ""
    return out


def _parse_span(outputs: dict, asset_idx: int, category: str,
                span_idx: int):
    """Every chunk record of one span, salvage-parsed and pooled.
    (parsed_rows, None) or (None, reason)."""
    rows, chunks = [], 0
    while True:
        record_id = f"{asset_idx}:{category}:{span_idx}:{chunks}"
        if record_id not in outputs:
            break
        for line in outputs[record_id].splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            obj, tier = parse_partial_output.parse_jsonl_line(line, ())
            if obj is not None:
                rows.append(obj)
        chunks += 1
    if chunks == 0:
        return None, "no record reached the model (span selected no frames, or the read failed)"
    if not rows:
        return None, f"{chunks} record(s) answered but no line parsed"
    return rows, None


def _assemble(category: str, spans_out: list, gaps: list) -> dict:
    if category == "claim-verify":
        verdicts, bad = [], []
        for span, span_idx, rows in spans_out:
            row = rows[0]
            verdict = (row.get("verdict") or "").strip().lower()
            if verdict not in VERDICTS:
                bad.append({"span": _span_ref(span, span_idx),
                            "reason": f"verdict {verdict!r} outside "
                                      f"{VERDICTS} — treated as unparsed"})
                continue
            verdicts.append({"id": span.get("id") or f"span_{span_idx}",
                             "verdict": verdict,
                             "evidence": row.get("evidence") or ""})
        return {"verdicts": verdicts, "gaps": gaps + bad}

    # credits-read and title-read share the card shape {t, lines[]};
    # credits cards additionally carry the per-name pairings ({name,
    # paired_with, header, kind} — text verbatim, kind the card's own
    # typed statement, 2026-09-09) and title cards the typed `names`
    # ({name, type, descriptor}). Malformed pairings/names degrade to
    # an ABSENT key on that card alone, never a dropped card; an
    # attested [] is kept (see below).
    blocks = []
    for span, span_idx, rows in spans_out:
        cards = []
        for r in rows:
            if not (isinstance(r.get("t"), (int, float))
                    and isinstance(r.get("lines"), list)):
                continue
            card = {"t": r["t"], "lines": [str(l) for l in r["lines"]]}
            raw_pairings = r.get("pairings")
            pairings = [
                {"name": str(p["name"]),
                 "paired_with": (str(p["paired_with"])
                                 if p.get("paired_with") is not None
                                 else None),
                 "header": (str(p["header"])
                            if p.get("header") is not None else None),
                 **({"kind": str(p["kind"])} if p.get("kind") else {})}
                # the CONTAINER guard matters as much as the element
                # guard: a model answering `"pairings": 3` must degrade
                # this card, never TypeError the whole state after the
                # inference spend (review, 2026-09-05)
                for p in (raw_pairings
                          if isinstance(raw_pairings, list) else [])
                if isinstance(p, dict) and p.get("name")]
            raw_names = r.get("names")
            names = [
                {"name": str(n["name"]),
                 "type": (str(n["type"]) if n.get("type") else None),
                 "descriptor": (str(n["descriptor"])
                                if n.get("descriptor") is not None
                                else None)}
                for n in (raw_names
                          if isinstance(raw_names, list) else [])
                if isinstance(n, dict) and n.get("name")]
            if names or raw_names == []:
                card["names"] = names
            # []-vs-absent is the family attestation convention and a
            # consumer branches on it (review, 2026-09-06): an ATTESTED
            # empty list ("pairings": []) is kept — the model looked and
            # paired nothing, so the consumer must NOT fall back to text
            # heuristics. Only a card whose pairings never arrived or
            # arrived unusable (missing key, non-list, or entries that
            # ALL failed validation) omits the key — degraded, and the
            # consumer's per-card fallback applies there.
            if pairings or raw_pairings == []:
                card["pairings"] = pairings
            cards.append(card)
        cards.sort(key=lambda c: c["t"])
        blocks.append({"start": span["start"], "end": span["end"],
                       "kinds": span.get("kinds") or [],
                       "shot_ids": span.get("shot_ids") or [],
                       "cards": cards})
    return {"blocks": blocks, "gaps": gaps}


def _span_ref(span: dict, span_idx: int):
    return {"id": span.get("id") or f"span_{span_idx}",
            "start": span["start"], "end": span["end"]}
