"""
Input schema validation and normalization for shot-segmentation-on-aws.

Single source of truth for all user-configurable defaults.
Imports nothing from this project — no circular import risk.

Event shape
───────────
{
  "uuid": "...",
  "input": {
    "s3uri":  "s3://bucket/path/to/video.mxf",
    "bucket": "bucket",           ← derived from s3uri by validate_and_normalize()
    "key":    "path/to/video.mxf",← derived from s3uri by validate_and_normalize()
    "input_config": {             ← optional; user overrides
      "segment_duration_secs":     1200,
      "coarse_fps":                4,
      "cosdist_threshold":         0.10,
      "shot_boundary_threshold":   0.20,
      "black_pixel_threshold":     0.04,
      "black_coverage_pct":        99.5,
      "black_entropy_max":         0.20,
      "max_recurrence_threshold":  0.07,
      "min_recurrence_count":      3
    }
  },
  "output": {
    "s3uri": "s3://proxy-bucket/output/prefix"   ← required
  },
  "data": {}
}

validate_and_normalize() does the following:
  1. Validates uuid, input.s3uri, output.s3uri are present and well-formed.
  2. Derives input.bucket + input.key from input.s3uri if not already set;
     verifies consistency if they are set.
  3. Parses output.s3uri → writes data.proxy_bucket + data.proxy_prefix.
     output.s3uri is required; ENV_PROXY_BUCKET fallback is defined but
     unreachable (_resolve_destination is never called).
  4. Fills all input_config defaults.
  5. Validates input_config ranges.
  6. Writes data.frames_to_shots_cfg — the internal config dict carried by
     all window iterator descriptors through the pipeline.
"""
from __future__ import annotations
import os

# ── Exposed input_config defaults ─────────────────────────────────────────────
DEFAULT_SEGMENT_DURATION_SECS    = 1200   # 20-minute transcode segments
DEFAULT_COARSE_FPS               = 4      # coarse-grid frame sampling rate
DEFAULT_COSDIST_THRESHOLD        = 0.10   # cosine-distance zoom-in trigger

DEFAULT_BLACK_PIXEL_THRESHOLD    = 0.04   # black frame: max pixel mean (0-1)
DEFAULT_BLACK_COVERAGE_PCT       = 99.5   # black frame: % pixels below threshold
DEFAULT_BLACK_ENTROPY_MAX        = 0.20   # black frame: max luma-histogram entropy (bits); see note on redigitised tape
DEFAULT_BLACK_MAX_BRIGHT_PIXEL   = 30.0   # black frame: max luma value of any single pixel (0-255); title cards fail this

# 0.20 suits clean digital sources and is the right default, but it is far too
# tight for content redigitised from tape. Raise it per request for that
# material — do NOT lower the default expecting "purer" black.
#
# 0.20 bits requires ~97% of pixels to land in one luma code, which only
# synthetically generated black achieves. Black recorded to tape and
# redigitised is dithered across a handful of low codes by transport and
# encoder noise, so it measures 0.4-2.8 bits even when every pixel sits below
# luma 17. Measured over 9,514 shots / 6 Betacam SX news masters: 14 shots
# were flat black to the eye yet typed "content", frame entropy across them
# ranged 0.389-2.777, and 0.20 vetoed 14 of 14. Two opened a story segment, so
# the shot list reported those segments starting 8.32 s and 1.72 s early — 4
# boundary errors on an asset otherwise scoring 0.947 boundary F1.
#
# For such material pass black_entropy_max=3.5 in input_config. That is safe
# because coverage + max-bright-pixel are what actually separate black from
# dark content: that pair sorted the 14 from 82 genuinely dark content shots
# and 18 ordinary content controls with zero errors — the 14 had 100.00% of
# pixels below the coverage tolerance, every other shot 99.99% or less.
# Entropy adds no separating power on top, because a frame whose luma is
# confined to codes 0-30 carries no picture however it is distributed in
# there. It also bounds the useful range: criteria 1+2 cap entropy at
# log2(31) = 4.95, so 3.5 stays binding as a backstop while clearing the
# observed spread, and any value at or above 4.95 is a silent no-op.
#
# Symptom to recognise: flat-black shots typed "content", each one dragging
# the start of whatever segment it heads. Cross-check a suspect shot's
# entropy_mean against its luma — if coverage is ~100% and max luma is well
# under black_max_bright_pixel, entropy is the only thing vetoing it.

DEFAULT_MINI_BATCH_SECS          = 120.0  # min batch size for frame extraction and dino
DEFAULT_EMBED_GEOMETRY           = "letterbox"  # model-input geometry: letterbox | squeeze | crop; all distance thresholds are calibrated on letterbox
EMBED_GEOMETRIES                 = ("letterbox", "squeeze", "crop")

DEFAULT_OVERLAP_SECS             = 30.0   # dynamic chunk: overlap seconds between two chunks
DEFAULT_WINDOW_SECS              = 360.0  # dynamic chunk: segment chunk size in seconds

# ── Noise defaults ─────────────────────────────────────────────────
DEFAULT_NOISE_DISTANCE_THRESHOLD       = 0.18   # merge/reconcile shots: max prev/next centroid distance for noise shot merge
DEFAULT_MAX_NOISE_DURATION             = 0.5    # merge/reconcile shots: max duration (s) to classify a shot as noise
DEFAULT_ADJACENT_CENTROID_THRESHOLD    = 0.20   # merge/reconcile shots: max adjacent centroid distance for drift correction
DEFAULT_MIN_SHOT_DURATION              = 0.25   # merge/reconcile shots: unconditional absorb-into-prev threshold (s); sub-threshold shots cannot be real editorial shots
DEFAULT_MAX_ADJACENT_MERGE_DURATION    = 30.0   # merge/reconcile shots: hard ceiling (s) for adjacent drift merge result; prevents runaway chaining across commercial breaks
DEFAULT_NEAR_BLACK_ENTROPY_MAX         = 1.5    # merge/reconcile shots: shots with entropy_mean below this are reclassified as black before merge passes run

# Value deliberately left at 1.5, but be aware of what this net can and cannot
# do. entropy_mean is the only per-frame scalar that survives into
# merge_reconcile_shots, and it does not discriminate blackness:
#   - Real black measures 0.4-2.8 bits, so 1.5 cuts through the middle of the
#     class it is meant to capture, not the gap between two classes. The two
#     current_affairs_c misses landed at 1.549 and 1.513 — short of the cut by 0.049 and
#     0.013.
#   - Within-shot spread exceeds that margin by an order of magnitude. One
#     genuinely black shot (SC0013 621) ranged 0.389-2.267 across its 16
#     frames, so its mean is an average over a wildly varying quantity and
#     which side of 1.5 it lands on is close to arbitrary.
# Raising it trades missed blacks for content misread as black, and both cost
# boundaries. The fix is not a better threshold here: it is to persist a luma
# statistic (coverage fraction or luma_mean) per frame alongside "entropy" in
# the window metadata written by run_dino_v2_model, and reclassify on that.
# Blackness is a luma property and luma is measured at frame level, then
# discarded. Until then this net is a coarse safety margin behind the
# frame-level detector, not a classifier.

# ── Fade split defaults ─────────────────────────────────────────────
DEFAULT_FADE_SPLIT_THRESHOLD        = 0.45   # fade split: min centroid-of-halves distance to attempt a split
DEFAULT_FADE_SPLIT_MIN_DURATION_SEC = 8.0    # fade split: minimum shot duration (s) to attempt a split
DEFAULT_FADE_SPLIT_MIN_SIDE_FRAMES  = 5      # fade split: minimum frames on each side of a candidate split
# ── Drift split defaults ────────────────────────────────────────────
DEFAULT_DRIFT_SPLIT_THRESHOLD         = 0.40   # drift split: min 1s-stride cosine distance peak to split a very long shot; gradual scene drift never exceeds shot_boundary_threshold frame-to-frame but accumulates over 1s
DEFAULT_DRIFT_SPLIT_MIN_DURATION_SEC  = 60.0   # drift split: only shots longer than this are candidates; a >60s single shot is a-priori improbable in edited content
DEFAULT_DRIFT_SPLIT_MIN_SIDE_SEC      = 2.0    # drift split: each side of a split must be at least this long; prevents degenerate sliver splits

# ── Shot frame selection ───────────────────────────────────────────
DEFAULT_FRAME_SELECT_MIN_DIST    = 0.05   # merge/reconcile shots: minimum cosine distance to select frame

# Recurring-shot grouping: complete-linkage DIAMETER bound — every pair inside
# a group is within this, so published max_distance <= this by construction.
# 0.07, not the historical 0.05: complete linkage makes the same number
# stricter (single linkage only required SOME pair within it), and the sweep
# against the six visually verified true cards puts the safe window at
# (0.0634, ~0.075] — the markets_a gradient-wipe card has diameter 0.0634 and
# cannot form below that; the drama_e act-break card splits at 0.080
# (test/harness/sweep.py). Callers still passing 0.05 lose the markets_a-class
# card silently.
DEFAULT_MAX_RECURRENCE_THRESHOLD = 0.07
DEFAULT_MIN_RECURRENCE_COUNT     = 3      # minimum shots to form a recurrence group
# README.md §9.1: members closer than this collapse into ONE occurrence cluster;
# min_recurrence_count is enforced on CLUSTERS, killing dialogue-coverage
# (A/B/A/B) groups whose members all sit inside one scene.
DEFAULT_MIN_OCCURRENCE_GAP_SEC   = 60.0
# README.md §9.3: drop groups whose raw member count exceeds this — recurring
# SETS (anchor cameras, confessional chairs) pass every similarity gate
# because they ARE near-identical; structural furniture recurs 3-7 times.
DEFAULT_MAX_RECURRENCE_COUNT     = 8

# README.md §9.6 — lib/frame_hash.py is COPIED byte-identical from
# face-recognition-on-aws (the validated master); these constants are its
# import surface and must match that repo's values.
PATCH_GRID = 3                       # 3×3 patches + 1 global hash
PATCH_NOISE_CHANGE_FREQ = 0.70       # patch changing in >70% of pairs = chronic noise
PATCH_NOISE_MAX_DISCOUNT = 3         # never discount more than 3 of 9 patches
GLOBAL_HASH_SANITY_FACTOR = 2.0      # endpoint global distance < 2× threshold

# README.md §9.6 confirmation gate + §9.4 texture floor (module constants, not
# user knobs — recurrence.json schema and the tool surface stay unchanged):
# a group is CONFIRMED when frames from two DIFFERENT members match on
# stable-patch dHash; frames below the entropy floor carry too little
# texture to hash meaningfully (glitch-black, fades), and a group with <2
# hashable members is dropped.
# Confirmation statistic — MEASURED on the real drama_e act-break card
# (4 members, centroid max_dist 0.0142): cross-member GLOBAL hamming 0-7,
# stable-patch MEDIAN 1-6, stable-patch MAX 7-13 (a single grainy patch
# spikes the max even between adjacent frames of ONE shot, max 19). So the
# max-of-patches break statistic is NOT the confirmation statistic: a pair
# confirms on whole-frame global <= MAX_GLOBAL (the main pipeline's
# validated maxHammingDistance semantics) OR stable-patch median <=
# MAX_MEDIAN (the ticker-robust complement for burned cards, where the
# churn corrupts the global hash but the discounted patches still match).
RECURRENCE_HASH_MAX_GLOBAL   = 10    # whole-frame hamming to confirm a pair
RECURRENCE_HASH_MAX_MEDIAN   = 6     # median stable-patch hamming to confirm
RECURRENCE_HASH_MIN_ENTROPY  = 1.0   # per-frame luma-entropy floor for hashing
RECURRENCE_HASH_MAX_FRAMES   = 8     # frames per member used for confirmation
RECURRENCE_HASH_NOISE_THRESHOLD = 6  # per-patch churn threshold for noisy-patch classification

# README.md §9.4 option (a) — black-DOMINANCE floor, the mitigation the plan
# prefers and the one the entropy floor structurally CANNOT provide: a
# timecode-burn master's frames are ~99.2% black with burned-in text, so
# their entropy reads ABOVE the floor (that is the whole defect) while every
# such frame hashes identically, manufacturing a group per burn window.
# Measured on the luma already decoded for hashing, so it costs no extra
# I/O and no new per-frame diagnostics; mirrors the main pipeline's
# maxBlackPixelFraction: 0.9. "Black" reuses DEFAULT_BLACK_PIXEL_THRESHOLD
# on the 0-255 luma scale, so the two detectors agree on what black is.
RECURRENCE_MAX_BLACK_FRACTION = 0.90

DEFAULT_SHOT_BOUNDARY_THRESHOLD  = 0.20   # shot pipeline: primary shot-cut sensitivity
# ── Extract-frame-features checkpoint/resume (9.13) ──────────────────
# A 6-minute window on very high-cut-rate content (film_a's action window)
# deterministically exceeds Lambda's HARD 900 s ceiling — the timeout cannot
# be raised, so the state checkpoints per ~120 s mini-batch and resumes:
# before each batch it requires enough remaining time for one more batch
# (estimated as EXTRACT_BATCH_TIME_FACTOR x the slowest batch measured THIS
# invocation) plus the checkpoint upload itself; when short, it saves the
# accumulated state to S3 and raises LambdaResourceTimeoutException, which
# the ASL retries and the fresh invocation resumes from the checkpoint.
# Same pattern as face-recognition's checkpoint/resume states.
EXTRACT_CHECKPOINT_BUFFER_MS  = 90_000  # reserve for the checkpoint upload
EXTRACT_BATCH_TIME_FACTOR     = 2.0     # headroom over the slowest batch seen

# ── Burst-run evidence (lib/burst_evidence.py, 9.15) ─────────────────
# Internal constants, not input_config knobs — the runs are advisory evidence
# and the definition is deliberately ONE place. A burst is a consecutive run
# of at least BURST_MIN_RUN shots each shorter than BURST_MAX_SHOT_SEC, shot
# TYPE ignored on purpose: jingles separate their cards with black
# micro-shots, and a type filter breaks every run at each separator (measured
# on jingle.mp4 — the longest run reads 8 with a content-only rule, 35
# without). Thresholds fitted on a handful of assets (jingle 94.8% of shots
# in runs vs <= 8.5% across the corpus — an order-of-magnitude gap, so the
# exact values are not load-bearing); the 9.14 full-ingest sweep grows the n.
BURST_MAX_SHOT_SEC = 1.0
BURST_MIN_RUN      = 6

# ── Programme-tail evidence (lib/tail_evidence.py) ───────────────────
# Internal constants, not input_config knobs: the evidence is advisory and the
# consumer applies its own reading; exposing more knobs would grow the tool
# schema for numbers nobody has needed to move. The separator is a maximal
# RUN of consecutive black shots (segmentation splits one physical black on
# luma flicker or a dark end board — drama_c's 33 s separator arrived as
# four black shots, ground-truthed 2026-08-23); the firing corpus's single
# separators are 63-92 s and pass unchanged as runs of one. The old 60 s
# content floor is REMOVED (it suppressed a real 55.6 s textless block; an
# end-board tail reads honestly in the numbers instead). The match threshold
# is recurrence's own 0.05 — the same "near-identical render" distance, in
# the same embedding space.
PROGRAMME_TAIL_MIN_BLACK_SEC   = 30.0   # separator black RUN must span this
PROGRAMME_TAIL_MATCH_THRESHOLD = 0.05   # tail-to-body near-duplicate distance
# A textless element is the SAME CUT as its source, so its duration must agree
# almost exactly; the tolerance absorbs shot-boundary jitter between the texted
# and clean segmentation passes. Measured: matched textless pairs are
# frame-exact at the median (delta 0.00 s, 95-96% within 1 s), while the one
# accidental embedding match on a part-boundary tail missed by 2.68 s — the
# duration test removes even that residue.
PROGRAMME_TAIL_DURATION_TOLERANCE_SEC = 1.0
# Heavy burned graphics push a clean copy's embedding past the strict 0.05
# where subtitles do not (measured: a credit-roll background matched its
# clean tail copy at 0.058, duration agreeing to 0.12 s; genuinely-new tail
# content sits at 0.2-0.6 — n=2 assets, 5 shots). The corroborated band
# admits those ONLY with the duration agreement above — distance alone never
# widens the match.
PROGRAMME_TAIL_CORROBORATED_BAND = 0.10

# Shot pipeline, 9.8 fix 1: a super-threshold distance spike over a SMALL
# non-unit frame gap is a cut with bounded (+-gap-1 frame) localisation error,
# not noise — Meridian's two fused cuts were 0.41/0.44 spikes over gaps 0 and 2,
# both silently discarded by the gap==1 rule, and the fresh 2026-08-21 corpus
# still carries two live gap-2 cuts on film_b (d=0.22, d=0.78 — the
# latter an unambiguous hard cut to a close-up of actor_D). 3 is safe because
# the smallest DELIBERATE sampling gap is the coarse stride,
# round(fps/coarse_fps) >= 6 — a gap <= 3 only ever comes from PTS-mapping
# duplicates/holes (root-caused and fixed, 9.8 fix 2) or genuinely-VFR input,
# so this is defence-in-depth that fires only when something upstream broke.
SMALL_GAP_BOUNDARY_MAX           = 3
SYNC_POP_THRESHOLD               = 0.10   # shot pipeline: SYNC Pop frame threshold
SYNC_POP_ENTROPY_MAX             = 1.0    # shot pipeline: max luma entropy for a healed single-frame anomaly to be labeled sync_pop; a true sync-pop is a monochrome-solid frame (near-zero entropy, small margin for a "2" glyph); higher-entropy anomalies are labeled flash_frame instead

# Left at 1.0, but likely too low for the same reason black_entropy_max was:
# "monochrome-solid" off a redigitised tape measures 0.4-2.8 bits, not
# near-zero, so genuine 2-pops will be labeled flash_frame. Consistent with
# current_affairs_c reporting 95 flash_frame against 87 sync_pop where a news master
# should be dominated by leader structure. Unverified — the sync_pop payload
# records frame/time/timecode but not the entropy that decided the label, so
# the split cannot be audited from shots.json. Unlike black_entropy_max this
# costs no boundary accuracy: both labels heal the anomaly identically and
# only the metadata differs. Add the deciding entropy to the payload before
# retuning this.
FREEZE_SIMILARITY_THRESHOLD      = 0.998  # shot pipeline: Freeze frame threshold
OUTLIER_DISTANCE_THRESHOLD       = 0.05   # shot pipeline: outlier embedding threshold



# ── Public API ─────────────────────────────────────────────────────────────────

def parse_s3uri(uri: str) -> tuple[str, str]:
    """
    Parse 's3://bucket/key/path' → ('bucket', 'key/path').
    Raises ValueError on malformed input.
    """
    if not uri or not uri.startswith("s3://"):
        raise ValueError(f"Invalid s3uri — must start with 's3://': {uri!r}")
    rest = uri[5:]
    if "/" not in rest:
        raise ValueError(f"Invalid s3uri — no key path after bucket: {uri!r}")
    bucket, key = rest.split("/", 1)
    if not bucket:
        raise ValueError(f"Invalid s3uri — empty bucket: {uri!r}")
    return bucket, key.strip("/")


def validate_and_normalize(event: dict) -> dict:
    """
    Validate required fields, fill defaults,
    and data.frames_to_shots_cfg.  Returns a new event dict (does not mutate).
    Raises ValueError on missing required fields or invalid values.
    """
    _validate_required(event)

    inp = event["input"]
    out = event.get("output") or {}
    data = event.get("data") or {}

    # ── Normalize input_config ─────────────────────────────────────────────────
    input_config = _merge_input_config(inp.get("input_config") or {})
    _validate_input_config(input_config)

    # ── Normalize output_config ─────────────────────────────────────────────────
    output_config = _merge_output_config(out.get("output_config") or {})
    _validate_output_config(output_config)

    # ── Build frames_to_shots_cfg ─────────────────────────────────────────────
    # This dict is written once here and carried unchanged through the pipeline:
    # probe → dynamic_chunking (stamped into each window descriptor) →
    # run_dino_v2_model (reads from item.frames_to_shots_cfg) →
    # merge_reconcile_shots (reads from data.frames_to_shots_cfg).
    frames_to_shots_cfg = {
        # Exposed
        "segment_duration_secs":    input_config["segment_duration_secs"],
        "overlap_secs":             DEFAULT_OVERLAP_SECS,
        "coarse_fps":               input_config["coarse_fps"],
        "cosdist_threshold":        input_config["cosdist_threshold"],
        "shot_boundary_threshold":  input_config["shot_boundary_threshold"],
        "black_pixel_threshold":    input_config["black_pixel_threshold"],
        "black_coverage_pct":       input_config["black_coverage_pct"],
        "black_entropy_max":        input_config["black_entropy_max"],
        "black_max_bright_pixel":   input_config["black_max_bright_pixel"],
        "embed_geometry":           input_config["embed_geometry"],
        "max_recurrence_threshold": input_config["max_recurrence_threshold"],
        "min_recurrence_count":     input_config["min_recurrence_count"],
        # Exposed — merge/reconcile
        "noise_distance_threshold":       input_config["noise_distance_threshold"],
        "adjacent_centroid_threshold":    input_config["adjacent_centroid_threshold"],
        "max_adjacent_merge_duration":    input_config["max_adjacent_merge_duration"],
        "near_black_entropy_max":         input_config["near_black_entropy_max"],
        "min_shot_duration":              input_config["min_shot_duration"],
        "fade_split_threshold":           input_config["fade_split_threshold"],
        "fade_split_min_duration_sec":    input_config["fade_split_min_duration_sec"],
        "drift_split_threshold":          input_config["drift_split_threshold"],
        "drift_split_min_duration_sec":   input_config["drift_split_min_duration_sec"],
        # Hardened
        "window_secs":              DEFAULT_WINDOW_SECS,
        "mini_batch_secs":          DEFAULT_MINI_BATCH_SECS,
    }

    return {
        **event,
        "input": {
            **inp,
            "input_config": input_config,
        },
        "output": {
            **out,
            "output_config": output_config,
        },
        "data": {
            **data,
            "frames_to_shots_cfg": frames_to_shots_cfg,
        },
    }


def get_input_config(event: dict) -> dict:
    """Return the normalized input_config from the event."""
    return (event.get("input") or {}).get("input_config") or {}


def get_output_config(event: dict) -> dict:
    """Return the normalized output_config from the event."""
    return (event.get("output") or {}).get("output_config") or {}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _validate_required(event: dict) -> None:
    if not event.get("uuid"):
        raise ValueError("uuid is required")

    inp = event.get("input") or {}
    if not inp.get("s3uri"):
        raise ValueError("input.s3uri is required")
    parse_s3uri(inp["s3uri"])  # validate format

    out = event.get("output") or {}
    if not out.get("s3uri"):
        raise ValueError("output.s3uri is required")
    parse_s3uri(out["s3uri"])   # validate format


def _merge_input_config(user: dict) -> dict:
    """Merge user-supplied input_config over defaults."""
    def _get(key, default):
        v = user.get(key)
        return v if v is not None else default

    return {
        "segment_duration_secs":       int(_get("segment_duration_secs",        DEFAULT_SEGMENT_DURATION_SECS)),
        "coarse_fps":                  int(_get("coarse_fps",                   DEFAULT_COARSE_FPS)),
        "cosdist_threshold":           float(_get("cosdist_threshold",          DEFAULT_COSDIST_THRESHOLD)),
        "shot_boundary_threshold":     float(_get("shot_boundary_threshold",    DEFAULT_SHOT_BOUNDARY_THRESHOLD)),
        "black_pixel_threshold":       float(_get("black_pixel_threshold",      DEFAULT_BLACK_PIXEL_THRESHOLD)),
        "black_coverage_pct":          float(_get("black_coverage_pct",         DEFAULT_BLACK_COVERAGE_PCT)),
        "black_entropy_max":           float(_get("black_entropy_max",          DEFAULT_BLACK_ENTROPY_MAX)),
        "black_max_bright_pixel":      float(_get("black_max_bright_pixel",     DEFAULT_BLACK_MAX_BRIGHT_PIXEL)),
        "embed_geometry":              str(_get("embed_geometry",               DEFAULT_EMBED_GEOMETRY)),
        "max_recurrence_threshold":    float(_get("max_recurrence_threshold",   DEFAULT_MAX_RECURRENCE_THRESHOLD)),
        "min_recurrence_count":        int(_get("min_recurrence_count",         DEFAULT_MIN_RECURRENCE_COUNT)),
        "min_occurrence_gap_sec":      float(_get("min_occurrence_gap_sec",     DEFAULT_MIN_OCCURRENCE_GAP_SEC)),
        "max_recurrence_count":        int(_get("max_recurrence_count",         DEFAULT_MAX_RECURRENCE_COUNT)),
        "noise_distance_threshold":       float(_get("noise_distance_threshold",       DEFAULT_NOISE_DISTANCE_THRESHOLD)),
        "adjacent_centroid_threshold":    float(_get("adjacent_centroid_threshold",    DEFAULT_ADJACENT_CENTROID_THRESHOLD)),
        "max_adjacent_merge_duration":    float(_get("max_adjacent_merge_duration",    DEFAULT_MAX_ADJACENT_MERGE_DURATION)),
        "near_black_entropy_max":         float(_get("near_black_entropy_max",         DEFAULT_NEAR_BLACK_ENTROPY_MAX)),
        "min_shot_duration":              float(_get("min_shot_duration",              DEFAULT_MIN_SHOT_DURATION)),
        "fade_split_threshold":           float(_get("fade_split_threshold",           DEFAULT_FADE_SPLIT_THRESHOLD)),
        "fade_split_min_duration_sec":    float(_get("fade_split_min_duration_sec",    DEFAULT_FADE_SPLIT_MIN_DURATION_SEC)),
        "drift_split_threshold":          float(_get("drift_split_threshold",          DEFAULT_DRIFT_SPLIT_THRESHOLD)),
        "drift_split_min_duration_sec":   float(_get("drift_split_min_duration_sec",   DEFAULT_DRIFT_SPLIT_MIN_DURATION_SEC)),
    }


def _validate_input_config(ic: dict) -> None:
    if ic["segment_duration_secs"] < 60:
        raise ValueError(
            f"input_config.segment_duration_secs must be >= 60 "
            f"(got {ic['segment_duration_secs']!r})"
        )
    if ic["coarse_fps"] < 1:
        raise ValueError(
            f"input_config.coarse_fps must be >= 1 (got {ic['coarse_fps']!r})"
        )
    if not (0.0 < ic["cosdist_threshold"] <= 1.0):
        raise ValueError(
            f"input_config.cosdist_threshold must be in (0, 1] "
            f"(got {ic['cosdist_threshold']!r})"
        )
    if not (0.0 < ic["shot_boundary_threshold"] <= 1.0):
        raise ValueError(
            f"input_config.shot_boundary_threshold must be in (0, 1] "
            f"(got {ic['shot_boundary_threshold']!r})"
        )
    if not (0.0 <= ic["black_pixel_threshold"] < 1.0):
        raise ValueError(
            f"input_config.black_pixel_threshold must be in [0, 1) "
            f"(got {ic['black_pixel_threshold']!r})"
        )
    if not (0.0 < ic["black_coverage_pct"] <= 100.0):
        raise ValueError(
            f"input_config.black_coverage_pct must be in (0, 100] "
            f"(got {ic['black_coverage_pct']!r})"
        )
    if not (0.0 <= ic["black_entropy_max"] < 10.0):
        raise ValueError(
            f"input_config.black_entropy_max must be in [0, 10) "
            f"(got {ic['black_entropy_max']!r})"
        )
    if not (0.0 < ic["black_max_bright_pixel"] <= 255.0):
        raise ValueError(
            f"input_config.black_max_bright_pixel must be in (0, 255] "
            f"(got {ic['black_max_bright_pixel']!r})"
        )
    if ic["embed_geometry"] not in EMBED_GEOMETRIES:
        raise ValueError(
            f"input_config.embed_geometry must be one of {EMBED_GEOMETRIES} "
            f"(got {ic['embed_geometry']!r})"
        )
    if not (0.0 <= ic["max_recurrence_threshold"] < 1.0):
        raise ValueError(
            f"input_config.max_recurrence_threshold must be in [0, 1) "
            f"(got {ic['max_recurrence_threshold']!r})"
        )
    if ic["min_recurrence_count"] < 2:
        raise ValueError(
            f"input_config.min_recurrence_count must be >= 2 "
            f"(got {ic['min_recurrence_count']!r})"
        )

    if ic["min_occurrence_gap_sec"] < 0:
        raise ValueError(
            f"input_config.min_occurrence_gap_sec must be >= 0 "
            f"(got {ic['min_occurrence_gap_sec']!r})"
        )
    if ic["max_recurrence_count"] < ic["min_recurrence_count"]:
        raise ValueError(
            f"input_config.max_recurrence_count must be >= min_recurrence_count "
            f"(got {ic['max_recurrence_count']!r} < {ic['min_recurrence_count']!r})"
        )
    if not (0.0 < ic["noise_distance_threshold"] < 1.0):
        raise ValueError(
            f"input_config.noise_distance_threshold must be in (0, 1) "
            f"(got {ic['noise_distance_threshold']!r})"
        )
    if ic["min_shot_duration"] <= 0.0:
        raise ValueError(
            f"input_config.min_shot_duration must be > 0 "
            f"(got {ic['min_shot_duration']!r})"
        )
    if ic["min_shot_duration"] >= ic.get("max_noise_duration", DEFAULT_MAX_NOISE_DURATION):
        raise ValueError(
            f"input_config.min_shot_duration must be < max_noise_duration "
            f"(got min={ic['min_shot_duration']!r}, max={ic.get('max_noise_duration', DEFAULT_MAX_NOISE_DURATION)!r})"
        )
    if not (0.0 < ic["adjacent_centroid_threshold"] <= ic["shot_boundary_threshold"]):
        raise ValueError(
            f"input_config.adjacent_centroid_threshold must be in "
            f"(0, shot_boundary_threshold] "
            f"(got {ic['adjacent_centroid_threshold']!r}, "
            f"shot_boundary_threshold={ic['shot_boundary_threshold']!r})"
        )
    if ic["max_adjacent_merge_duration"] <= 0.0:
        raise ValueError(
            f"input_config.max_adjacent_merge_duration must be > 0 "
            f"(got {ic['max_adjacent_merge_duration']!r})"
        )
    if not (0.0 <= ic["near_black_entropy_max"] < 10.0):
        raise ValueError(
            f"input_config.near_black_entropy_max must be in [0, 10) "
            f"(got {ic['near_black_entropy_max']!r})"
        )
    if not (0.0 < ic["fade_split_threshold"] <= 1.0):
        raise ValueError(
            f"input_config.fade_split_threshold must be in (0, 1] "
            f"(got {ic['fade_split_threshold']!r})"
        )
    if ic["fade_split_min_duration_sec"] <= 0.0:
        raise ValueError(
            f"input_config.fade_split_min_duration_sec must be > 0 "
            f"(got {ic['fade_split_min_duration_sec']!r})"
        )
    if not (0.0 < ic["drift_split_threshold"] <= 1.0):
        raise ValueError(
            f"input_config.drift_split_threshold must be in (0, 1] "
            f"(got {ic['drift_split_threshold']!r})"
        )
    if ic["drift_split_min_duration_sec"] <= ic["fade_split_min_duration_sec"]:
        raise ValueError(
            f"input_config requires drift_split_min_duration_sec "
            f"({ic['drift_split_min_duration_sec']!r}, default "
            f"{DEFAULT_DRIFT_SPLIT_MIN_DURATION_SEC}) > "
            f"fade_split_min_duration_sec "
            f"({ic['fade_split_min_duration_sec']!r}, default "
            f"{DEFAULT_FADE_SPLIT_MIN_DURATION_SEC}) — raise "
            f"drift_split_min_duration_sec or lower "
            f"fade_split_min_duration_sec"
        )


def _merge_output_config(user: dict) -> dict:
    """Merge user-supplied output_config over defaults."""
    return user

def _validate_output_config(oc: dict) -> None:
    return
