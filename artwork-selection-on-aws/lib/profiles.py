"""Layer B: WHICH candidates to hunt, as declarative data.

A profile row is a structural claim about composition, never a scalar
fitted to a title. ONE neutral profile ships until genre evidence
exists; adding a genre is a row here, not a code path anywhere.
"""

# Closed camera vocabulary (shot-metadata-enrichment), grouped by the
# composition class each archetype wants.
_CLOSE = ("close_up", "medium_close_up")
_MID = ("medium", "medium_wide", "over_the_shoulder")
_WIDE = ("wide", "extreme_wide", "aerial")

# Archetype -> constraints over one shot_metadata row.
#   camera        allowed camera_shot_type values
#   people_count  allowed values (None = field absent is acceptable)
#   people        "lead" = the top-screen-time person must appear,
#                 "top_cast" = any of the top-N, None = no requirement
ARCHETYPES = {
    "hero_closeup": {
        "camera": _CLOSE, "people_count": ("1",), "people": "lead"},
    # "lead" hunts the TOP HERO_CAST_N cast members round-robin (user
    # direction 2026-10-06): a two-anchor desk gets BOTH anchors in the
    # hero slate — four frames of one person violated the diversity
    # rule ("distinct compositions and people") and starved the
    # co-lead. On a true star vehicle the supporting cast rarely has
    # qualifying solo close-ups, so the lead fills the slate anyway —
    # self-correcting, no genre branch.
    "two_shot": {
        "camera": _MID, "people_count": ("2",), "people": "top_cast"},
    "establishing_vista": {
        "camera": _WIDE, "people_count": ("0", "several", None),
        "people": None},
}

# genre -> {archetype: slate size}. Callers pass genre; it is never
# inferred. Unknown or absent genre gets the neutral row.
PROFILES = {
    "default": {"hero_closeup": 4, "two_shot": 4,
                "establishing_vista": 4},
}

# The shot kinds that are FOOTAGE (news support, 2026-10-05):
# graphics_overlay is real footage wearing channel furniture — on
# broadcast news it is 80%+ of all shots, so excluding the kind
# excluded the genre. Everything else (title_card, logo_bumper,
# black, slate, credit_roll) is not footage and never artwork.
ARTWORK_SHOT_KINDS = ("normal", "graphics_overlay")

HERO_CAST_N = 3

# Frames grabbed per PERSON-archetype candidate (2026-10-06): the shot
# archives sample ~1 fps and a talking face's single middle frame is
# frequently mid-blink/mid-word (measured: a news hero slate where the
# judge's own reason was "the most open eyes AMONG ALL FOUR" and half
# were flagged unusable). FR's per-detection attributes (facegroup
# faces: eyes_open/sunglasses/face_occluded, timestamped) pick the
# members to grab; the vision judge arbitrates the pixels.
FRAMES_PER_PERSON_CANDIDATE = 3

DEFAULT_SPOILER_CUTOFF = 0.75
MIN_SHOT_SEC = 1.2            # shorter shots are transitions
RANK_DURATION_CAP_SEC = 8.0   # stability preference saturates here
TOP_CAST_N = 5


def profile_for(genre):
    return PROFILES.get((genre or "").strip().lower(),
                        PROFILES["default"])
