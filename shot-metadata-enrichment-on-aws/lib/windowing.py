"""
The image-bounded window planner (README.md §3, settled 2026-08-19).

One record = a window of consecutive shots. **Pack until the shot target
OR the image budget is reached, whichever binds first** (WINDOW_SHOT_TARGET
/ WINDOW_IMAGE_BUDGET — see sme_common for the current numbers and their
provenance) — the budget counts EVERY Converse image block in the record:
the context shot's frames, every required shot's frames, and the face
gallery's single block when the window carries named faces. The budget must
never exceed the primary model's registry max_images: bedrock-inference's
validate_plan hard-refuses records over it before a token is spent, so a
window that overflows here would be a whole-job refusal there.

The shot count therefore floats DOWNWARD from the target when the 3-frame
tier stacks up, never above it. Frame counts are planned from shots.json's
own `selected_frame_count` (the tarball's contents), so the plan can never
promise a frame the archive does not hold.
"""
from __future__ import annotations


def frames_for_shot(shot: dict, tier) -> int:
    """The README §5 tier — 1 under 4 s, 2 under 12 s, 3 at >= 12 s — capped
    by what the shot's archive actually holds."""
    duration = float(shot["end_time"]) - float(shot["start_time"])
    want = next(n for bound, n in tier if duration < bound)
    available = int(shot.get("selected_frame_count")
                    or len(shot.get("selected_frames") or []) or 1)
    return max(1, min(want, available))


def burst_fold(burst_runs: list, representatives: int) -> tuple[dict, set]:
    """Fold plan for shot-segmentation's `burst_runs` envelope evidence.

    A burst run is a contiguous interval of >= 6 consecutive sub-second
    shots (a jingle, a montage, a promo sting). Every member keeps its own
    REQUIRED entry and its own output document; what folds is the IMAGES:
    only `representatives` members per run — first, middle, last — carry
    frames, because the members of a run are visually near-interchangeable
    for cataloguing purposes (a jingle's cards measure internally frozen at
    frame_coherence ~0.99) and each frame is the dominant token cost.

    Returns (run_of, frameless): run_of maps member shot_id -> run index,
    frameless is the set of member shot_ids that get NO frame images.
    representatives <= 0 disables folding (run_of still returned, so the
    prompt annotation survives the knob).
    """
    run_of: dict = {}
    frameless: set = set()
    for k, run in enumerate(burst_runs or []):
        ids = list(range(int(run["start_shot_id"]),
                         int(run["end_shot_id"]) + 1))
        for sid in ids:
            run_of[sid] = k
        if representatives > 0 and len(ids) > representatives:
            keep = {ids[0], ids[len(ids) // 2], ids[-1]}
            frameless.update(sid for sid in ids if sid not in keep)
    return run_of, frameless


def plan_windows(shots: list, *, shot_target: int, image_budget: int,
                 tier, gallery_shots=None, frameless=None) -> list:
    """Pack `shots` (already sorted by shot_id, consecutive) into windows.

    gallery_shots: the set of shot_ids that carry at least one named face —
    a window touching any of them spends 1 image block on its gallery, and
    the budget must reserve it BEFORE the shots are packed (a gallery that
    appears because shot k joined must not invalidate shots 1..k-1).

    frameless: shot_ids planned at ZERO frames — non-representative members
    of burst runs (see burst_fold). They still occupy a REQUIRED slot and
    count against shot_target (each still costs the model an output
    document); they just carry no image blocks.

    Returns [{index, shot_ids, context_shot_id, frames, image_blocks}]:
    `frames` maps shot_id -> planned frame count (context shot included).
    """
    gallery_shots = gallery_shots or set()
    frameless = frameless or set()

    def _plan_frames(shot):
        if shot["shot_id"] in frameless:
            return 0
        return frames_for_shot(shot, tier)

    windows = []
    i = 0
    while i < len(shots):
        context = shots[i - 1] if i > 0 else None
        # The context shot is EXEMPT from burst folding: it exists solely to
        # give the first required shot's boundary judgment visual evidence,
        # a frameless context gives none, and the exemption costs at most
        # one image per window.
        ctx_frames = frames_for_shot(context, tier) if context else 0
        # Reserve the gallery block up front whenever faces exist AT ALL in
        # this run: cheaper than re-packing, and a reserved-but-unused
        # block only means one window packs a frame fewer — never illegal.
        reserve = 1 if gallery_shots else 0
        used = ctx_frames + reserve
        members, frames = [], {}
        if context:
            frames[context["shot_id"]] = ctx_frames
        while i < len(shots) and len(members) < shot_target:
            n = _plan_frames(shots[i])
            if members and used + n > image_budget:
                break            # the budget binds first
            if not members and used + n > image_budget:
                # A single shot + context must always fit: shed context
                # frames before refusing (a 3-frame shot after a 3-frame
                # context inside a 4-block budget is a config error).
                raise ValueError(
                    f"shot {shots[i]['shot_id']} needs {n} frames but the "
                    f"budget {image_budget} cannot fit them beside "
                    f"{ctx_frames} context frames + {reserve} gallery "
                    "block — raise WINDOW_IMAGE_BUDGET or fix the tier")
            frames[shots[i]["shot_id"]] = n
            members.append(shots[i]["shot_id"])
            used += n
            i += 1
        windows.append({
            "index": len(windows),
            "shot_ids": members,
            "context_shot_id": context["shot_id"] if context else None,
            "frames": frames,
            "image_blocks": used if gallery_shots else used,
        })
    return windows
