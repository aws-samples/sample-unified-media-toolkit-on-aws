"""
ledger.py — the cross-run read ledger (README.md §9: "no run ever re-pays
for evidence it already has").

One JSON file per UNDERLYING VIDEO, keyed by the shot-segmentation
workflow_id (the stable per-video handle every read is scheduled
against), shared by all binding runs of that video:

    micro-workflow/cast-resolution/_ledger/{shotseg_wid}.json
    {"entries": [{"t0", "t1", "category", "crop"?, "output_s3_prefix"}]}

Entries are appended by BindPersons only AFTER the witness products
were fetched — the ledger records EVIDENCE OBTAINED, never frames
merely submitted (a failed read must not suppress the retry). The
scheduler passes entries to plan_reads as the never-re-pay filter and,
when everything is already ledgered, hands the recorded product
prefixes straight to binding.

Last-writer-wins on concurrent binds of the same video is accepted: a
lost append costs one duplicate read (money), never correctness.
"""
from __future__ import annotations

from lib import s3io


def uri_for(plan_s3uri: str, shotseg_wid: str) -> str:
    """The ledger uri, derived from the run's own plan uri — same
    bucket, same target prefix, no extra configuration.
    .../cast-resolution/{uuid}/_plan/plan.json -> .../_ledger/{wid}.json"""
    target_prefix = plan_s3uri.rsplit("/_plan/", 1)[0].rsplit("/", 1)[0]
    return f"{target_prefix}/_ledger/{shotseg_wid}.json"


def load(ledger_uri: str) -> list[dict]:
    try:
        return (s3io.get_json(ledger_uri) or {}).get("entries") or []
    except Exception:                        # noqa: BLE001 — absent or bad
        return []


def as_spans(entries: list[dict]) -> list[tuple]:
    """plan_reads' ledger shape: [(t0, t1, category, crop, product_v)] —
    crop is None for full-frame entries (and for pre-crop entries, which
    were all full-frame); product_v is None for entries recorded before
    the typed-extraction products (reads.PRODUCT_V), which therefore
    never satisfy a current need. A cropped entry must never satisfy a
    full-frame need (review, 2026-09-05)."""
    return [(float(e["t0"]), float(e["t1"]), e["category"],
             e.get("crop"), e.get("product_v"))
            for e in entries]


def prefixes(entries: list[dict]) -> list[str]:
    return sorted({e["output_s3_prefix"] for e in entries
                   if e.get("output_s3_prefix")})


def append(ledger_uri: str, new_entries: list[dict]) -> None:
    """Merge-append; exact duplicates collapse so a redrive is a no-op."""
    if not new_entries:
        return
    entries = load(ledger_uri)
    seen = {(e["t0"], e["t1"], e["category"], e.get("output_s3_prefix"))
            for e in entries}
    for e in new_entries:
        key = (e["t0"], e["t1"], e["category"], e.get("output_s3_prefix"))
        if key not in seen:
            entries.append(e)
            seen.add(key)
    s3io.put_json(ledger_uri, {"entries": entries})
