"""
Canonical event shape and single source of truth for all user-configurable
defaults. Imports nothing from this project.

Event shape (Step Functions input):

    {
      "uuid": "<workflow_id>",
      "input": {
        "s3uri": "s3://bucket/path/to/source.mp4",
        "input_config": { ...all fields optional, defaults below... }
      },
      "output": {
        "s3uri": "s3://proxy-bucket/micro-workflow/face-recognition/<workflow_id>"
      },
      "data": {}
    }

Recognition policy (GDPR use cases — README.md §2b)
─────────────────────────────────────────────────
Three orthogonal toggles compose every use case. The sandbox collection
(transient, deleted at terminal state) is ALWAYS available — GDPR storage
limitation governs persistent biometric templates, not transient processing.

    celebrity_recognition   false → no RecognizeCelebrities calls
    collection_id           ""    → no persistent-collection ops (search AND index)
    max_faces_to_index      0     → auto-index off (persistent search still on)
    index_named_faces_only  true  → auto-index only high-confidence-named faces
"""

from __future__ import annotations

import os

# ── Frame extraction (media-processing child) ─────────────────────────────────
DEFAULT_FRAME_INTERVAL_SEC = 1.0     # one frame per second
MIN_FRAME_INTERVAL_SEC     = 0.2
MAX_FRAME_INTERVAL_SEC     = 10.0
# Fit-inside target; MP scale_dimension never upscales and handles portrait.
FRAME_TARGET_WIDTH  = 1920
FRAME_TARGET_HEIGHT = 1080

# ── Probe-then-fill detection ──────────────────────────────────────────────────
DEFAULT_PROBE_STRIDE = 4             # detect every Nth frame in pass 1; 0 = dense
MAX_PROBE_STRIDE     = 30
DEFAULT_DEDUP_HAMMING_THRESHOLD = 4  # near-dup skip inside dense regions; 0 = off
MAX_DEDUP_HAMMING_THRESHOLD     = 16
# Patch dHash noise classification (ticker discount)
PATCH_GRID = 3                       # 3×3 patches + 1 global hash
PATCH_NOISE_CHANGE_FREQ = 0.70       # patch changing in >70% of pairs = chronic noise
PATCH_NOISE_MAX_DISCOUNT = 3         # never discount more than 3 of 9 patches
GLOBAL_HASH_SANITY_FACTOR = 2.0      # endpoint global distance < 2× threshold

# ── Frame triage ───────────────────────────────────────────────────────────────
# A byte-size pre-filter (drop below a threshold scaled by pixel count) was
# tried and removed — JPEG size tracks scene complexity, not face presence;
# see lib/frame_triage.py docstring. Every frame is decoded and classified
# from actual pixel content.
TRIAGE_BLACK_LUMA_MEAN       = 8.0    # 0-255
TRIAGE_BLACK_LUMA_P99        = 24.0
TRIAGE_MONO_STDDEV           = 4.0
TRIAGE_BLUR_LAPLACIAN_VAR    = 3.0

# ── Grid packing ───────────────────────────────────────────────────────────────
FRAMES_PER_GRID       = 20           # cap; layout solver may pack fewer
GRID_MAX_EDGE_PX      = 9_800        # margin under Rekognition's 10000 px bound
GRID_MAX_JPEG_BYTES   = 14_000_000   # margin under the 15 MB S3Object bound
GRID_JPEG_QUALITY     = 85           # step down 75, 65 if size exceeded
GRID_CELL_SEPARATOR_PX = 2           # black lines between cells (cross-cell bleed guard)
GRID_SPLIT_FACE_COUNT = 95           # ≥95 faces returned → split grid, re-detect
GRID_SPLIT_MIN_FRAMES = 4
CELL_BLEED_MAX_OVERHANG = 0.15       # box crossing cell boundary >15% → bleed artifact

# ── Face detection / quality ───────────────────────────────────────────────────
DEFAULT_MIN_CONFIDENCE = 80.0
# Absolute brightness/sharpness windows (CENTROID_QUALITY_TIERS,
# SHARPNESS_RANGE 30-100, BRIGHTNESS_RANGE 4-70) were REMOVED after measuring
# them against a real 2-hour broadcast (7,327 detected faces): the sharpness
# floor rejected 86.8% of all faces (population median 9.9), the brightness
# ceiling rejected 47% — including 98-100% of the LIGHTEST-lit people's faces
# while passing 89% of a dark-skinned anchor's — so when the window did bind,
# it selected light-skinned people's darkest 2% of frames. Brightness is a
# lighting measure that correlates with skin tone (same set/lighting: light
# skin mean 81, dark skin mean 65); gating on it is a fairness liability with
# no upside since IndexFaces applies Rekognition's own quality model
# (QualityFilter) as the real gate. Selection now ranks RELATIVELY within
# the candidate pool (see lib/face_quality.py); only extreme floors remain.
CENTROID_MIN_SHARPNESS = 2.40
# Noise filter for single-appearance faces, at 960×540 reference (scaled by dim).
NOISE_MIN_FACE_PX_AT_REF = 42
NOISE_LOW_BRIGHTNESS     = 4.0
NOISE_LOW_SHARPNESS      = 30.0
# Extreme floor only — skip near-black crops; NOT a quality window.
QUALITY_MIN_BRIGHTNESS = 10.0
# Pose gate for best-face selection (search tiles + persistent indexing):
# fusion_rank's (1 - pose_dist) is only a soft multiplicative penalty that a
# large-enough face overrides — on real content it picked a yaw=-25 3/4
# profile for indexing because its area dominated. Hard-cap the pose instead,
# with fallback to the ungated pool when a cluster has no frontal face
# (measured pass rate 64% population-wide; 53-96% per cluster — not a trap).
POSE_MAX_ABS_YAW   = 20.0
POSE_MAX_ABS_PITCH = 20.0

# ── Shot-segmentation supplement (README.md §21) ────────────────────────────────
# Optional shots.json (shot-segmentation-on-aws schema) supplants the
# dHash cut detector for track containment + the dense-skip predicate.
# Same-asset cross-checks — fail-hard: a wrong-asset shots file silently
# fragmenting or gluing tracks is worse than a crash.
SHOT_FPS_TOLERANCE          = 0.01    # per-shot implied fps vs probe fps (1%)
SHOT_DURATION_TOLERANCE_SEC = 2.0     # last end_time vs source duration:
SHOT_DURATION_TOLERANCE_PCT = 0.02    #   within max(2 s, 2%)
SHOT_FPS_MIN_START_TIME_SEC = 5.0     # fps check needs a late-enough shot

# ── Temporal tracking (no shots) ──────────────────────────────────────────────
DEFAULT_MAX_TRACK_GAP_SEC = 3.0
# IoU dilation grows with the frame gap: dilate = 1 + IOU_GAP_DILATION * gap_sec
IOU_GAP_DILATION   = 0.5
MIN_TRACK_IOU      = 0.10            # against the (dilated) previous box

# ── Celebrity recognition ──────────────────────────────────────────────────────
DEFAULT_CELEBRITY_RECOGNITION = True
HEADSHOT_DIMENSION      = (120, 160)  # w, h per headshot cell
HEADSHOTS_PER_GRID      = 64          # RecognizeCelebrities API cap
HEADSHOT_CROP_MARGIN    = 0.30
# Structural floor for "this face IS the tile's crop subject": the crop
# guarantees the subject >= 1/(1+2*margin) of the tile linearly (=62.5%
# at margin 0.30); allow up to 2x dilution from the aspect-ratio
# expansion and up to 2x for a tighter re-detection box -> the subject's
# bbox can never fall below ~a tenth of the tile's area. A "dominant"
# face smaller than that is crop BACKGROUND whose real subject was
# dropped (quality filter / detection miss) — mapping it would put a
# stranger's face behind the tile's identity.
MIN_SUBJECT_TILE_SHARE  = (1.0 / (1.0 + 2 * HEADSHOT_CROP_MARGIN)) ** 2 / 4
MAX_TRACKS_PER_ITERATOR = 30
# Per-person representative face crop (debug/people/person_NNNN_face.jpg):
# 3:4 aspect at native resolution, downscale-only cap — a consumer-facing
# reference (shot-metadata-enrichment's face gallery renders 192×256 cells
# from it), so it must not be the 96×128 thumbnail (measured too small to
# re-identify against) nor an upscaled blur of a small face.
REP_FACE_RATIO       = (3, 4)          # w, h
REP_FACE_MAX_EDGE_PX = 512
# Majority-vote naming
NAME_VOTE_MIN_DATAPOINTS = 10
NAME_VOTE_MIN_SCORE      = 90.0
NAME_VOTE_INSTANT_SCORE  = 95.0

# ── Identity clustering (sandbox collection) ───────────────────────────────────
SANDBOX_INDEX_QUALITY_FILTER = "MEDIUM"
SANDBOX_MAX_FACES_PER_INDEX  = 100    # IndexFaces API cap
SANDBOX_SEARCH_THRESHOLD     = 75.0
SANDBOX_SEARCH_MAX_FACES     = 4096
# Two-tier merge (real contamination found on real content — see
# cluster_identities docstring): a single edge merges two identity groups
# only at >= SANDBOX_MERGE_SIMILARITY; edges in
# [SANDBOX_SEARCH_THRESHOLD, SANDBOX_MERGE_SIMILARITY) merge only when
# >= SANDBOX_WEAK_CORROBORATION independent edges connect the same two
# groups. Measured on 9 real runs: 89% of genuine same-person edges score
# >= 95 (sandbox faces come from the same footage), while every observed
# cross-person bridge lived in the 75-85 band.
SANDBOX_MERGE_SIMILARITY     = 90.0
SANDBOX_WEAK_CORROBORATION   = 2
CENTROIDS_PER_PERMUTATION_BATCH = 100

# ── Persistent collection search/index ─────────────────────────────────────────
DEFAULT_COLLECTION_ID = os.environ.get("DEFAULT_COLLECTION_ID", "")
# 0 = auto-index OFF by default (changed 2026-10-08, user decision for
# the public sample): writing strangers' faces into a persistent
# collection is opt-in, never a side effect of a default run.
# Persistent-collection SEARCH is unaffected.
DEFAULT_MAX_FACES_TO_INDEX = 0
DEFAULT_INDEX_NAMED_FACES_ONLY = False
DEFAULT_MIN_APPEARANCES = 3           # distinct tracks to validate an identity
SEARCH_BY_IMAGE_THRESHOLD  = 80.0
SEARCH_ACCEPT_SIMILARITY   = 85.0
PERSISTENT_INDEX_QUALITY_FILTER = "HIGH"
# Stored face derivatives (README.md §19): thumbnail = 3:4 face crop; the
# full_image derivative is a fit-inside downscale of the WHOLE source
# (FULLIMAGE_MAX_EDGE_PX below) with `coord` locating the face — the old
# FACE_FULLIMAGE_DIM face-region crop was removed with that change.
FACE_THUMBNAIL_DIM = (96, 128)      # w, h
# Persistent IndexFaces crop: preserve native resolution. Downscale only
# when the crop's long edge exceeds this cap; NEVER upscale. The old fixed
# 180x240 resize threw away ~4x linear resolution on close-up faces (measured:
# largest crops ~929x858 source px) before Rekognition ever saw them — the
# API takes 15 MB via S3Object, so shrinking bought nothing.
INDEX_CROP_MAX_EDGE_PX = 1024

# ── Phase 2: face index management (README.md §19) ──────────────────────────────
# collection-index GSI: pk=collection_id, sk=celeb_created ("{celeb}#{epoch_ms}").
# Unnamed faces sort last via the tilde sentinel (0x7E > any letter/digit).
COLLECTION_INDEX_GSI = "collection-index"
CELEB_INDEX_GSI = "celeb-index"
UNNAMED_CELEB_SENTINEL = "~unnamed"
TOMBSTONE_PK_PREFIX = "deleted#"
TOMBSTONE_TTL_SEC = 30 * 86400        # audit trail for hard-deleted faces
# Batch indexing (state machine): fit-inside tiles into a SQUARE-ISH grid.
# TILE/GRID SIZE — measured, not derived (README.md §20 addendum): a
# 1080x1080/9-col variant was deployed and MEASURED WORSE (7 IndexFaces
# calls vs 1 on the same 25 real images). Rekognition's ~40px min face is
# relative to 1080p and SCALES UP with the image's LONG edge
# (floor ~= 40 * long_edge / 1920):
#   - downscaled sources: tile size cancels out of face/floor entirely —
#     bigger tiles buy nothing (group shots are the solo-retry fallback's
#     job at any tile size);
#   - native-size small sources (avatars, 720px web stills): the face
#     stays at native pixels while a bigger grid raises the floor — they
#     MISS in large grids and pass in small ones.
# So minimize the grid LONG edge: 8x6 = 3854x3850 px, floor ~80px — the
# exact geometry that passed 25/25 real images in one call. Rows are capped
# (not 10) because the floor tracks the long edge; a tall 8x10 grid
# (6418px) raises the floor to ~134px and re-breaks small sources.
# 48/grid => 1000 images ~= 21 calls ($0.021) vs 13 ($0.013) — trivial.
BATCH_TILE_DIM = (480, 640)           # w, h — fit-inside, never upscale
BATCH_TILES_PER_GRID = 48
BATCH_TILE_COLS = 8                   # 8x6 grid = 3854x3850 px (square-ish)
BATCH_SOLO_MAX_EDGE_PX = 4096         # tiny-face fallback: gated full-res solo call
BATCH_INDEX_QUALITY_FILTER = "AUTO"   # curated user images (seed-script precedent)
MAX_BATCH_UPDATE_FACES = 500          # gateway-side rename cap per call
DDB_BATCH_WRITE_SIZE = 25             # DynamoDB BatchWriteItem hard limit
REKOGNITION_DELETE_FACES_MAX = 4096   # DeleteFaces hard limit per call
LIST_FACES_MAX_PAGE_SIZE = 100
LIST_FACES_DEFAULT_PAGE_SIZE = 25
# Accepted image formats, gated by MAGIC BYTES (never extension). BMP/JPEG2000
# rejected (no magickload/jp2kload in the shipped libvips wheel); SVG/PDF by
# policy. HEIC decode can still fail at runtime on HEVC-less wheel builds —
# that surfaces per-line as failed/decode_error.
BATCH_ACCEPTED_FORMATS = ("jpeg", "png", "webp", "tiff", "heif", "gif")
# full_image derivative = fit-inside downscale of the WHOLE source image
# (video frame or user photo) — NOT a face-region crop. coord locates the
# face within it (normalized [x, y, w, h], resolution-agnostic).
FULLIMAGE_MAX_EDGE_PX = 1280

# ── Output rollup ──────────────────────────────────────────────────────────────
MIN_APPEARANCE_MS = 800

# ── Concurrency / API pool bounds (README.md §7 TPS budget) ──────────────────────
TRIAGE_POOL_THREADS  = 24             # S3 GET + pyvips decode overlap

# ── Checkpoint / re-enter ──────────────────────────────────────────────────────
REMAINING_TIME_BUFFER_MS = 60_000     # raise LambdaResourceTimeoutException below this
CHECKPOINT_EVERY_N_GRIDS = 5
# SearchFaces calls are cheaper to lose than grids (1 API call vs 1 call + S3
# churn) but far more numerous — checkpoint the clustering search loop less
# often per unit of work, same order of protection.
CHECKPOINT_EVERY_N_SEARCHES = 25

# ── Semaphore ──────────────────────────────────────────────────────────────────
SEMAPHORE_HOLDER_TTL_SEC  = 3600      # refreshed on every Rekognition-state entry
# Counter TTL: same window as a holder's, refreshed alongside it (try_acquire,
# decrement_once, refresh_holder) so the counter self-heals from a
# permanently-lost stream decrement without falsely expiring under a
# saturated, long-lived queue — see lib/semaphore.py invariant 5.
SEMAPHORE_COUNTER_TTL_SEC = 3600
SEMAPHORE_PENDING_TTL_SEC = 86400
SEMAPHORE_PK   = "semaphore"
SEMAPHORE_COUNTER_SK = "counter"
SEMAPHORE_HOLDER_SK_PREFIX = "holder#"
PENDING_PK = "pending"
STREAM_DEDUP_PK = "stream-dedup"
STREAM_DEDUP_TTL_SEC = 3600
# max_holders only ever ratchets UP across requests (never lowered by a later,
# smaller request) — see lib/semaphore.py try_acquire.
DEFAULT_MAX_CONCURRENT_EXECUTIONS = 5
MIN_MAX_CONCURRENT_EXECUTIONS     = 1
MAX_MAX_CONCURRENT_EXECUTIONS     = 50
DEFAULT_DETECT_MAP_CONCURRENCY = 4
MIN_DETECT_MAP_CONCURRENCY     = 1
MAX_DETECT_MAP_CONCURRENCY     = 10


# ── Defaults tables ────────────────────────────────────────────────────────────

_INPUT_CONFIG_DEFAULTS: dict = {
    # frame extraction
    "frame_interval_sec": DEFAULT_FRAME_INTERVAL_SEC,
    "use_partial": False,
    "mark_in_out": [0, 0],
    # detection strategy
    "probe_stride": DEFAULT_PROBE_STRIDE,
    "dedup_hamming_threshold": DEFAULT_DEDUP_HAMMING_THRESHOLD,
    "shot_segmentation_s3uri": "",
    # recognition policy (GDPR §2b)
    "celebrity_recognition": DEFAULT_CELEBRITY_RECOGNITION,
    "collection_id": DEFAULT_COLLECTION_ID,
    "max_faces_to_index": DEFAULT_MAX_FACES_TO_INDEX,
    "index_named_faces_only": DEFAULT_INDEX_NAMED_FACES_ONLY,
    # quality / validation
    "min_confidence": DEFAULT_MIN_CONFIDENCE,
    "min_appearances": DEFAULT_MIN_APPEARANCES,
    "max_track_gap_sec": DEFAULT_MAX_TRACK_GAP_SEC,
    # concurrency (request-level; no longer stack Parameters)
    "max_concurrent_executions": DEFAULT_MAX_CONCURRENT_EXECUTIONS,
    "detect_map_concurrency": DEFAULT_DETECT_MAP_CONCURRENCY,
}


def validate_and_normalize(event: dict) -> dict:
    """
    Validate required fields, merge input_config over defaults, range-check.
    Returns a NEW event dict (non-mutating). Raises ValueError with the full
    dotted path on any invalid field.
    """
    _validate_required(event)

    user_cfg = (event.get("input") or {}).get("input_config") or {}
    cfg = {**_INPUT_CONFIG_DEFAULTS, **user_cfg}
    _validate_input_config(cfg)

    return {
        "uuid": event["uuid"],
        "input": {**event["input"], "input_config": cfg},
        "output": event["output"],
        "data": event.get("data") or {},
    }


def get_input_config(event: dict) -> dict:
    return (event.get("input") or {}).get("input_config") or {}


def _validate_required(event: dict) -> None:
    if not event.get("uuid"):
        raise ValueError("uuid is required")
    s3uri = (event.get("input") or {}).get("s3uri") or ""
    if not s3uri.startswith("s3://"):
        raise ValueError(f"input.s3uri must be an s3:// uri (got {s3uri!r})")
    out = (event.get("output") or {}).get("s3uri") or ""
    if not out.startswith("s3://"):
        raise ValueError(f"output.s3uri must be an s3:// uri (got {out!r})")


def _validate_input_config(cfg: dict) -> None:
    v = float(cfg["frame_interval_sec"])
    if not (MIN_FRAME_INTERVAL_SEC <= v <= MAX_FRAME_INTERVAL_SEC):
        raise ValueError(
            f"input_config.frame_interval_sec must be in "
            f"[{MIN_FRAME_INTERVAL_SEC}, {MAX_FRAME_INTERVAL_SEC}] (got {v})")
    cfg["frame_interval_sec"] = v

    ps = int(cfg["probe_stride"])
    if not (0 <= ps <= MAX_PROBE_STRIDE):
        raise ValueError(
            f"input_config.probe_stride must be in [0, {MAX_PROBE_STRIDE}] (got {ps})")
    cfg["probe_stride"] = ps

    dh = int(cfg["dedup_hamming_threshold"])
    if not (0 <= dh <= MAX_DEDUP_HAMMING_THRESHOLD):
        raise ValueError(
            f"input_config.dedup_hamming_threshold must be in "
            f"[0, {MAX_DEDUP_HAMMING_THRESHOLD}] (got {dh})")
    cfg["dedup_hamming_threshold"] = dh

    shots_uri = cfg["shot_segmentation_s3uri"]
    if shots_uri is None:
        shots_uri = ""
    if not isinstance(shots_uri, str):
        raise ValueError(
            "input_config.shot_segmentation_s3uri must be a string")
    shots_uri = shots_uri.strip()
    if shots_uri and not shots_uri.startswith("s3://"):
        raise ValueError(
            f"input_config.shot_segmentation_s3uri must be an s3:// uri "
            f"(got {shots_uri!r})")
    cfg["shot_segmentation_s3uri"] = shots_uri

    cfg["celebrity_recognition"] = bool(cfg["celebrity_recognition"])
    cfg["index_named_faces_only"] = bool(cfg["index_named_faces_only"])

    cid = cfg["collection_id"]
    if cid is None:
        cid = ""
    if not isinstance(cid, str):
        raise ValueError("input_config.collection_id must be a string")
    cfg["collection_id"] = cid

    mfi = int(cfg["max_faces_to_index"])
    if mfi < 0:
        raise ValueError(
            f"input_config.max_faces_to_index must be >= 0 (got {mfi})")
    cfg["max_faces_to_index"] = mfi

    mc = float(cfg["min_confidence"])
    if not (0.0 <= mc <= 100.0):
        raise ValueError(
            f"input_config.min_confidence must be in [0, 100] (got {mc})")
    cfg["min_confidence"] = mc

    ma = int(cfg["min_appearances"])
    if ma < 1:
        raise ValueError(
            f"input_config.min_appearances must be >= 1 (got {ma})")
    cfg["min_appearances"] = ma

    gap = float(cfg["max_track_gap_sec"])
    if gap <= 0:
        raise ValueError(
            f"input_config.max_track_gap_sec must be > 0 (got {gap})")
    cfg["max_track_gap_sec"] = gap

    if cfg["use_partial"]:
        mio = cfg.get("mark_in_out") or [0, 0]
        if (not isinstance(mio, (list, tuple)) or len(mio) != 2
                or float(mio[1]) <= float(mio[0]) or float(mio[0]) < 0):
            raise ValueError(
                f"input_config.mark_in_out must be [start_sec, end_sec] with "
                f"0 <= start < end (got {mio!r})")
        cfg["mark_in_out"] = [float(mio[0]), float(mio[1])]

    mce = int(cfg["max_concurrent_executions"])
    if not (MIN_MAX_CONCURRENT_EXECUTIONS <= mce <= MAX_MAX_CONCURRENT_EXECUTIONS):
        raise ValueError(
            f"input_config.max_concurrent_executions must be in "
            f"[{MIN_MAX_CONCURRENT_EXECUTIONS}, {MAX_MAX_CONCURRENT_EXECUTIONS}] "
            f"(got {mce})")
    cfg["max_concurrent_executions"] = mce

    dmc = int(cfg["detect_map_concurrency"])
    if not (MIN_DETECT_MAP_CONCURRENCY <= dmc <= MAX_DETECT_MAP_CONCURRENCY):
        raise ValueError(
            f"input_config.detect_map_concurrency must be in "
            f"[{MIN_DETECT_MAP_CONCURRENCY}, {MAX_DETECT_MAP_CONCURRENCY}] "
            f"(got {dmc})")
    cfg["detect_map_concurrency"] = dmc


# ── shared identity/name rules ────────────────────────────────────────────────

def names_agree(a, b) -> bool:
    """ONE definition of "these two name strings claim the same person":
    casefolded equality. Shared by the §23 conflict trigger and
    finalize's celebrity_id gate — two comparators for one question is
    how a case-variant collection record fired a "conflict" that
    finalize then treated as agreement (review, 2026-09-07)."""
    return bool(a) and bool(b) and str(a).casefold() == str(b).casefold()


# The Rekognition API counters, in ONE place: CallCounter derives its
# fields from these, and usage_tracking prices exactly the charged
# tuple — a counter added there but not here vanished silently once
# (recognize_celebrities_api=0, the shipped under-billing bug).
CHARGED_API_COUNTER_KEYS = (
    "detect_faces_api", "recognize_celebrities_api",
    "index_faces_api", "search_faces_api",
)
# counted for observability (facemetadata usage), never priced and never
# reported in usage_tracking.json (user, 2026-09-07: no per-call charge)
UNCHARGED_API_COUNTER_KEYS = ("get_celebrity_info_api",)
API_COUNTER_KEYS = CHARGED_API_COUNTER_KEYS + UNCHARGED_API_COUNTER_KEYS
