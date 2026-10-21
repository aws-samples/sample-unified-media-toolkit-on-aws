"""
Per-shot ROLES for news content (README.md §3): studio
vs field vs graphics is a property of a SHOT, not a grouping. The primary
evidence already ships (enrichment default_news `news_setting`); this
module normalizes it and adds the anchor-set corroboration from
shot-segmentation recurrence (a locked-off anchor camera is a huge
recurring group — 69 members measured on real news).
"""
from __future__ import annotations

STORY_ROLES = {"studio_anchor_desk", "studio_interview", "field",
               "remote_box", "studio_video_wall"}
INTERSTITIAL_ROLES = {"graphic_full_screen"}
MIN_ANCHOR_GROUP = 12       # a recurring set this big on news is a studio
                            # camera, not furniture. Measured at n=1 per
                            # genre: news anchor cam 69, furniture < ~8 —
                            # but NARRATIVE interiors (confessionals) sit
                            # at 11-25, so 12 is only safe because this
                            # runs on news mode (the narrative band never
                            # reaches production here); re-measure before
                            # any reuse on narrative content, and when a
                            # second news asset lands.


def shot_roles(shot_docs: list, recurrence: dict | None) -> dict:
    """{shot_id: {"role": ..., "anchor_set": bool}} — role straight from
    the shipped news_setting; anchor_set marks membership in a large
    interior recurring group (the locked-off camera corroboration)."""
    big_groups = set()
    for g in (recurrence or {}).get("recurrence_groups", []):
        members = g.get("shots", [])
        if len(members) >= MIN_ANCHOR_GROUP:
            big_groups.update(m["shot_id"] for m in members
                              if m.get("position") == "interior")
    out = {}
    for d in shot_docs:
        sid = d["shot_id"]
        role = d.get("news_setting") or _fallback_role(d)
        out[sid] = {"role": role, "anchor_set": sid in big_groups}
    return out


def is_story_role(entry: dict) -> bool:
    return entry["role"] in STORY_ROLES or entry["anchor_set"]


def _fallback_role(d: dict) -> str:
    """Non-news enrichment (no news_setting): map the general fields so
    the module degrades instead of refusing."""
    kind = d.get("shot_kind") or "normal"
    if kind in ("title_card", "logo_bumper", "credit_roll", "black"):
        return "graphic_full_screen"
    return "field"
