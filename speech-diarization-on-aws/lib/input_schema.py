"""
Input schema validation and normalization for speech-diarization-on-aws.

Single source of truth for all user-configurable defaults.
Imports nothing from this project — no circular import risk.

Event shape
───────────
{
  "uuid": "...",
  "input": {
    "s3uri":        "s3://bucket/key",
    "input_config": {           ← optional; user overrides for language/vad/asr/clustering
      "language":     null,
      "use_partial":  False,
      "mark_in_out":  [t0, t1],
      "vad":          { ... },
      "asr":          { ... },
      "clustering":   { ... }
    }
  },
  "output": {
    "s3uri":         "s3://bucket/prefix",
    "output_config": {          ← optional; user overrides for output format/speaker channels
      "formats":          [...],
      "speaker_channels": true,
      "channel_mode":     "aligned"
    }
  },
  "data":     {}   ← transient/intermediate state written by each state handler
}

validate_and_normalize() fills defaults into input_config and output_config in-place
so all downstream state handlers can read from those locations without re-computing defaults.
"""
from __future__ import annotations
import os

# ── Fixed chunking  ──────────────────────────────────────────────────────────
DEFAULT_FIXED_CHUNK_SIZE_SECS = 720   # 12 minutes
DEFAULT_DYNAMIC_SPEECH_CHUNK_SECS = 360   # 6 minutes per iterator
# ── Dynamic chunking  ────────────────────────────────────────────────────────
# Adjacent VAD turns closer than this are merged into one speech group.
# 1.0s captures mid-sentence pauses and rapid exchanges while preserving
# natural sentence/speaker-change boundaries.  Inter-group gaps are always
# ≥ this value — well above Whisper's reliable segment-split threshold.
DEFAULT_SPEECH_GROUP_MAX_GAP_SEC = 1.0

# ── VAD  ────────────────────────────────────────────────────────────────────
DEFAULT_VAD_MODEL_ID = "pyannote.community1"
# 0.5 -> 0.0 (2026-10-19): the 0.5 s discard deleted real short utterances
# ("No.", names, interjections) — recovered words flow into speech groups;
# <0.75 s sentences never enter clustering (HIGH/LOW/SKIPPED tiers), so
# admission cannot pollute AHC. Gate of record: pooled 19-ep caption-GT
# battery — deletions down, insertions FLAT at threshold 0.30.
DEFAULT_MIN_SPEECH_DURATION_SEC = 0.0
DEFAULT_MIN_SILENCE_DURATION_SEC = 0.3

# ── Language Detect Probing ─────────────────────────────────────────────────
DEFAULT_NUM_PROBES = 5
DEFAULT_MIN_PROBE_SEC = 10.0

# ── ASR  ────────────────────────────────────────────────────────────────────
DEFAULT_ASR_MODEL_ID = "whisper"
DEFAULT_ASR_MODEL_VARIANT = "turbo"
# Merge gap: maximum inter-sentence gap (seconds) for merge_short to merge
# across.  Gaps larger than this are likely speaker changes.
DEFAULT_MERGE_GAP_THRESHOLD_SEC = 0.20
# Hidden (ASR decode hygiene) - exposed at the config level (same pattern
# as vad.seg_threshold: settable via input_config.asr for harness sweeps,
# never a gateway knob). Defaults pin faster-whisper's own effective
# values (verified against the faster-whisper==1.2.1 transcribe()
# signature at the tag), so exposing them changed NO behavior; note the
# temperature FALLBACK LADDER [0.0..1.0] is also active at beam>1 (our
# code pins 0.0 only at beam 1) - a sweep candidate. Retune only against the
# pooled caption-GT battery. condition_on_previous_text: rolling-context
# conditioning (initial_prompt is independent and applies either way).
# no_speech_threshold / log_prob_threshold: the segment-level
# self-distrust gates (no_speech_prob / avg_logprob fields).
DEFAULT_ASR_CONDITION_ON_PREVIOUS_TEXT = True
DEFAULT_ASR_NO_SPEECH_THRESHOLD = 0.6
DEFAULT_ASR_LOG_PROB_THRESHOLD = -1.0

# ── Embed  ──────────────────────────────────────────────────────────────────
# minimum turn duration for embedding
# 0.75s is sufficient for WeSpeaker ResNet34;
# 1.5s was too conservative for film (short turns common)
DEFAULT_MIN_EMBEDDING_DURATION_SEC = 0.75
# Soft embedding: minimum sentence duration for a low-confidence embedding
# used only for centroid-based speaker assignment (not clustering).
DEFAULT_SOFT_EMBEDDING_MIN_SEC = 0.30


# Elbow plateau: silhouette within this fraction of sil_max is "plateau"
DEFAULT_ELBOW_PLATEAU_PCT = 0.03


# ── Speaker channel separation ───────────────────────────────────────────────
DEFAULT_ENABLE_SPEAKER_CHANNELS = True
DEFAULT_SPEAKER_CHANNEL_MODE = "aligned"   # "aligned" | "compact"
# merge adjacent same-speaker regions
DEFAULT_SPEAKER_MERGE_GAP_SEC = 0.5
DEFAULT_SPEAKER_FADE_MS = 10          # fade in/out at segment boundaries
DEFAULT_COMPACT_GAP_SEC = 0.1         # silence between segments in compact mode

# ── Clustering ───────────────────────────────────────────────────────────────
DEFAULT_ALGORITHM = "ahc"
DEFAULT_MAX_K = 50
DEFAULT_MIN_TURNS_PER_SPEAKER = 3
DEFAULT_MIN_CLUSTER_SIZE = 2
DEFAULT_SIL_WEIGHT = 0.6
DEFAULT_CH_WEIGHT = 0.4
DEFAULT_K_SELECTION_METHOD = "elbow"  # "elbow" or "combined"
# Z[-1,2] below this → single speaker
DEFAULT_SINGLE_SPEAKER_MERGE_THRESHOLD = 0.20
DEFAULT_SINGLE_SPEAKER_MEAN_DIST = 0.12    # mean pairwise dist confirms single
DEFAULT_OUTLIER_GLOBAL_PERCENTILE = 97.0    # remove top (100-x)% by mean dist
DEFAULT_MAX_OUTLIER_FRACTION = 0.03    # never remove more than 3% of turns
# Hidden (AHC)
DEFAULT_CENTROID_TRIM_PERCENTILE = 90.0
DEFAULT_MIN_TURNS_FOR_TRIM = 5
DEFAULT_K_PENALTY = 0.02
DEFAULT_UNRELIABLE_SIL = 0.20
DEFAULT_ELBOW_PLATEAU_PCT = 0.03
# Hidden (VAD) — seg_threshold was hardcoded 0.35 in lib/segmentation.py
# (lowered from 0.5 with no recorded rationale); exposed 2026-09-16 so the
# DER harness can sweep it — the measured 3-10% SOLO miss was its suspect
# (README.md Design record P2, miss decomposition). 0.35 -> 0.30
# (2026-10-19): the knee of the measured WER curve, bracketed from both
# sides on the pooled 19-ep caption-GT battery (0.35/0 and 0.25/0 are
# both worse; insertions FELL at 0.30 — fuller speech groups reduce
# boundary hallucinations). Retune only against that battery, pooled.
DEFAULT_SEG_THRESHOLD = 0.30
# Hidden (P3): cross-talk constraints DEFAULT OFF — measured 2026-09-16
# (DER harness, build 20260916b): span-level evidence is 100% precise,
# but the span->SENTENCE mapping is only 77-93% under every rule swept
# (zh 56-79%: sentences still coarse), and a hard cannot-link built on
# a wrong mapping shreds a true cluster — net DER regression on every
# corpus (ami 31.8->40.1). Re-enable per-run for experiments; v2 =
# constrain span-level embeddings, not sentences.
DEFAULT_USE_CROSSTALK_CONSTRAINTS = False
# Hidden (P3 overlap evidence; lib/segmentation.py extract_overlap_evidence)
# — minimum solo/overlap durations for a within-window cannot-link record;
# structural floors (a constraint from <0.3 s of evidence is noise), n=0
# tuning so far — the sim overlap knob + AMI are the calibration bench.
OVERLAP_EVIDENCE_MIN_SOLO_SEC = 0.40
OVERLAP_EVIDENCE_MIN_OVERLAP_SEC = 0.30
# Powerset guides (2026-09-25 design; lib/powerset_guides.py): keep what
# the segmentation model knows — fences, overlap spans, stitched local
# speaker tracks, co-activity — instead of collapsing to plain VAD.
# ALL powerset consumption rides ONE flag (default off until the
# multilingual DER harness + KPI replay gates pass). Constants are the
# offline-rehearsal-validated shapes (10-asset corpus, 2026-09-25):
DEFAULT_USE_POWERSET_GUIDES = False
POWERSET_FENCE_SUPPORT = 0.5      # fraction of covering windows voting
POWERSET_BIND_MARGIN = 0.2        # junction best-vs-second permutation
POWERSET_MIN_EVENT_SEC = 0.07     # sustained frames each side of a flip
POWERSET_MIN_SEG_SEC = 0.6        # a track segment worth keeping
POWERSET_MIN_TRACK_SEC = 1.5      # a chain worth calling a track
POWERSET_TRACK_GAP_SEC = 0.3      # unvoiced gap a track may bridge
POWERSET_OVERLAP_THRESHOLD = 0.5  # mean P(pair) above this = cross-talk
# Track F0 hygiene (2026-09-25 round-2 forensics: flag-on impurity ROSE
# on mitj/nnn — minority modes sat in TRACK CORES: some chains weld two
# registers at a junction, and some clean cross-register tracks merge
# in AHC anyway). Two deterministic counters, both from per-sentence F0
# already computed at the embedding state:
#   QUARANTINE — a track whose own F0 is bimodal is a weld: excluded
#   from track units, its sentences fall back to margin assignment.
#   CANNOT-LINK — two confident tracks with cross-register medians may
#   never share a cluster (the cast-side F0 veto, applied where it can
#   PREVENT the merge). Ratio matches the corpus-calibrated bands
#   (male p95 178 / female p05 143 → 1.45 spans the registers).
POWERSET_F0_SPLIT_RATIO = 1.45
POWERSET_F0_MIN_SENTENCES = 5
POWERSET_F0_MINORITY = 0.25
# ...and the modes must STRADDLE the male/female register divide
# (round-3 live failure, 2026-09-25: excited same-register speech
# legitimately spans 160-360 Hz — a ratio test alone read healthy
# reality-TV tracks as welds, quarantined most of them, and reality_a
# collapsed back to K=3. Prosody is not register.) Bands anchor on the
# n=66 corpus calibration (male p95=178, female p05=143): a genuine
# cross-register mix has one mode clearly below the divide and one
# clearly above.
POWERSET_F0_STRADDLE_LOW = 170.0
POWERSET_F0_STRADDLE_HIGH = 195.0
# Post-clustering split (2026-09-28, the tdy lesson): the K-sweep can
# choose a K below the separation height of two SAME-REGISTER voices
# who never share a scene (two anchors of different show blocks: no
# co-activity, no alternation, no F0 evidence — but their pooled chain
# embeddings separate at AUC 0.94). After labels are assigned, any
# cluster whose member chains 2-split with silhouette >= this bound is
# divided. Runs AFTER the sweep — touches neither units nor the
# dendrogram (knife-edge-safe), and errs only toward over-split, the
# downstream-recoverable direction. Threshold swept offline on the 3
# acceptance exemplars (n=3, gate on the corpus round): the anchor
# pair separates at 0.45 and not at 0.55; reality_a retention 0.971 with
# a single collateral split at 0.45.
POWERSET_SPLIT_SIL = 0.45
POWERSET_SPLIT_MIN_CHAINS = 4
# Small-N guard (2026-09-29): tracks are the clustering unit, so the
# K sweep's candidate range is 2..n_tracks // min_turns_per_speaker.
# The failure mode is that range becoming TRIVIAL — one 12-min short
# yielded 8 chains on mix / 6 on stem, a single-candidate "sweep" that
# collapsed to one forced speaker on both, while sentence units give
# K=2 on the same audio. The floor is therefore structural, not an
# absolute count: the sweep must have at least this many candidate K
# values, i.e. n_chains >= (candidates + 1) * min_turns_per_speaker
# (15 at the default min_turns=3; every healthy corpus asset, n=11,
# carries 82-350 chains). A second, constant-free net catches
# anything the floor misses: a tracks sweep that self-reports
# fallback_no_valid_candidates is rerun on sentence units.
POWERSET_MIN_SWEEP_CANDIDATES = 4
# Cross-chunk continuity binding (2026-09-29): chunk ownership splits
# hard at fixed marks and tracks clip there, guaranteeing a split for
# any voice speaking across a mark. Two clipped chain ends within this
# tolerance of the same mark, exactly one per side, bind. Measured on
# the 13-asset corpus: 27/27 observed straddles were uncontested.
POWERSET_XCHUNK_BIND_TOL = 0.6
# Voice gate for cross-chunk binds: continuity at the mark plus chain
# embedding agreement. 0.70 sits far above different-speaker pooled
# sims measured at handovers and below same-voice chain sims.
POWERSET_XCHUNK_BIND_SIM = 0.70
# Reconcile evidence gate 3 — conversation veto: fragments of one
# person interleave at scene granularity, different people alternate
# with quick handovers (you cannot hand over to yourself in under a
# second). This many sub-QUICK_SEC A<->B alternations block a merge —
# but only when CORROBORATED by >= CORROB_SEC of measured co-presence
# between the pair, because assignment flap inside a monologue also
# "hands over" sub-second. The corroboration bar sits above one
# emission-floor blip and below a real exchange (measured on the two
# decisive corpus cases, n=2: a lone narrator's fragment pair carried
# exactly ONE link — a music/applause transient, the merge is correct
# — while a same-register two-person weld carried THREE links across
# two chunks and merging cost retention ARI 0.961 -> 0.767 against
# the human-validated labels).
POWERSET_RECONCILE_HANDOVER_MAX = 3
POWERSET_RECONCILE_QUICK_SEC = 1.0
POWERSET_RECONCILE_CORROB_SEC = 0.5
# Durable local-speaker floor (2026-09-29): a window's 0/1/2 labels
# are distinct people by model semantics, so the max durable
# per-window count is a K floor that needs no overlap and no
# surviving tracks — proof of existence doesn't need a good
# embedding. Durability: >= MIN_SEC active within the window AND the
# count corroborated by >= MIN_WINDOWS windows (one flickery window
# must never set the floor: a phantom third label on a music
# transient would force over-split, and a phantom second would
# overrule the single-speaker fast-path on narrator content).
POWERSET_LOCAL_FLOOR_MIN_SEC = 0.6
POWERSET_LOCAL_FLOOR_MIN_WINDOWS = 2
# Reconcile pass (2026-09-29): two clusters whose duration-weighted
# centroids exceed this cosine similarity MUST merge unless evidence
# separates them (durable co-presence, corroborated conversation, or
# a cross-register F0 gap). Measured corridor on the 13-asset corpus:
# confirmed same-person fragments 0.886-0.975; genuinely-different
# people ceiling ~0.68 pooled (7 assets' named pairs) — EXCEPT
# same-register family voices, which pool at 0.909 on the reality
# corpus and are exactly what the evidence gates exist to catch.
# Threshold sweep on the ear-labeled sets + retention (2026-09-29,
# n=13 assets): 0.80 and 0.85 both pass everything; 0.90 loses the
# lone-narrator reunification (its fragment pair sits at 0.892); 0.95
# additionally loses the quiz-host merge. Retention is flat at 0.961
# across the sweep because wrong merges are stopped by evidence, not
# by this number. 0.85 keeps headroom on both sides.
POWERSET_RECONCILE_SIM = 0.85
# A cannot-link blocks a reconcile merge only when DURABLE: the two
# clusters' chains must genuinely co-occur for >= this many seconds of
# pair-class frames, deduped on the frame grid and shipped in the
# artifact as co_active_secs (2026-09-29: track-segment intersection
# CANNOT measure this — the frame grid votes one chain per frame, so
# chain segments are disjoint by construction; 434/434 corpus links
# measured 0.00 s, silently disabling the gate). False co-activity
# from applause/laughter flicker runs sub-second and must not veto a
# high-similarity merge of one person's fragments (measured: two
# ear-confirmed fragment pairs at sim 0.95/0.97 were blocked by
# flicker links). Artifacts predating co_active_secs read each link
# at 0.25 s — the emission rule's guaranteed lower bound (a link only
# exists when one window carried >= 0.25 s of pair frames), never an
# estimate above it.
POWERSET_RECONCILE_LINK_MIN_SEC = 2.0
# Hidden (post-clustering assignment; lib/assign.py — measured 2026-09-15,
# see README.md Design record P5: global assignment scores at/below
# chance on short embeddings, locality scores 7x chance)
DEFAULT_ASSIGN_LOCAL_WINDOW_SEC = 30.0  # candidate speakers need an anchor here
DEFAULT_ASSIGN_MARGIN = 0.05            # best must lead runner-up by this
# Hidden (VBx)
VBX_DEFAULT_FA = 0.3
VBX_DEFAULT_FB = 17.0
VBX_DEFAULT_LDA_DIM = 0
VBX_DEFAULT_MAX_ITER = 40
VBX_DEFAULT_INIT_THRESHOLD = 0.350


# ── WAV INFO ─────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_NUM_CHANNELS = 1
DEFAULT_BITS_PER_SAMPLE = 16
WAV_HEADER_BYTES = 512    # fetch enough for non-standard headers

_VALID_ALGORITHMS = frozenset(("ahc",))  # "vbx" is currently disabled
_VALID_METHODS = frozenset(("elbow", "combined"))
_VALID_CHANNEL_MODES = frozenset(("aligned", "compact"))


# ── Defaults ──────────────────────────────────────────────────────────────────

_INPUT_CONFIG_DEFAULTS: dict = {
    "language": None,

    "use_partial": False,

    "mark_in_out": [0, 0],

    "model": {
        "model_bucket": os.environ.get("MODEL_BUCKET",          ""),
        "model_prefix": os.environ.get("MODEL_ARTEFACTS_PREFIX", "_model_artefacts").strip("/"),
        "cf_host":      os.environ.get("CLOUDFRONT_URL",         ""),
    },

    "vad": {
        "model_id":                  DEFAULT_VAD_MODEL_ID,
        "fixed_chunk_size_secs":     DEFAULT_FIXED_CHUNK_SIZE_SECS,
        "dynamic_speech_chunk_secs": DEFAULT_DYNAMIC_SPEECH_CHUNK_SECS,
        "min_speech_duration_sec":   DEFAULT_MIN_SPEECH_DURATION_SEC,
        "min_silence_duration_sec":  DEFAULT_MIN_SILENCE_DURATION_SEC,
        "speech_group_max_gap_sec":  DEFAULT_SPEECH_GROUP_MAX_GAP_SEC,
        # ONE flag, read by the VAD state only: when on, powerset_{n}
        # guides are written under {prefix}/powerset/ and every
        # downstream consumer keys off ARTIFACT PRESENCE (absent = not
        # assessed — the three-state rule; no flag plumbing).
        "use_powerset_guides":       DEFAULT_USE_POWERSET_GUIDES
    },

    "asr": {
        "model_id":                      DEFAULT_ASR_MODEL_ID,
        "model_variant":                 DEFAULT_ASR_MODEL_VARIANT,
        "language_detection_num_probes": DEFAULT_NUM_PROBES,
        "language_detection_min_sec":    DEFAULT_MIN_PROBE_SEC,
        "num_workers":                   1,
        "beam_size":                     2,
        "initial_prompt":                "",
        "merge_gap_threshold_sec":       DEFAULT_MERGE_GAP_THRESHOLD_SEC,
        "condition_on_previous_text":    DEFAULT_ASR_CONDITION_ON_PREVIOUS_TEXT,
        "no_speech_threshold":           DEFAULT_ASR_NO_SPEECH_THRESHOLD,
        "log_prob_threshold":            DEFAULT_ASR_LOG_PROB_THRESHOLD
    },

    "embed": {
        "min_embedding_duration_sec":  DEFAULT_MIN_EMBEDDING_DURATION_SEC,
        "soft_embedding_min_sec":      DEFAULT_SOFT_EMBEDDING_MIN_SEC
    },

    "clustering": {
        "algorithm":                 DEFAULT_ALGORITHM,
        "num_speakers":              None,
        "max_k":                     DEFAULT_MAX_K,
        "min_cluster_size":          DEFAULT_MIN_CLUSTER_SIZE,
        "min_turns_per_speaker":     DEFAULT_MIN_TURNS_PER_SPEAKER,
        "k_selection_method":        DEFAULT_K_SELECTION_METHOD,
        "sil_weight":                DEFAULT_SIL_WEIGHT,
        "ch_weight":                 DEFAULT_CH_WEIGHT,
        "outlier_global_percentile": DEFAULT_OUTLIER_GLOBAL_PERCENTILE,
        "single_speaker_threshold":  DEFAULT_SINGLE_SPEAKER_MERGE_THRESHOLD,
        "apply_snorm":               False
    }
}

_OUTPUT_CONFIG_DEFAULTS: dict = {
    "speaker_channels": True,
    "channel_mode":     DEFAULT_SPEAKER_CHANNEL_MODE,
    # Hidden
    # "merge_gap_sec":    DEFAULT_SPEAKER_MERGE_GAP_SEC,
    # "fade_ms":          DEFAULT_SPEAKER_FADE_MS,
    # "compact_gap_sec":  DEFAULT_COMPACT_GAP_SEC
}


# ── Public API ────────────────────────────────────────────────────────────────

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
    Validate required fields and fill all defaults into input_config and output_config.
    Raises ValueError on missing required fields or invalid values.
    Returns a new event dict (does not mutate the input).
    """
    _validate_required(event)

    input_config = _merge_input_config(
        event["input"].get("input_config") or {})
    output_config = _merge_output_config(
        event["output"].get("output_config") or {})

    _validate_input_config(input_config)
    _validate_output_config(output_config)

    return {
        **event,
        "input": {
            **event["input"],
            "input_config": input_config,
        },
        "output": {
            **event["output"],
            "output_config": output_config,
        },
        "data": event.get("data") or {},
    }


def get_input_config(event: dict) -> dict:
    """Return the normalized input_config from the event."""
    return (event.get("input") or {}).get("input_config") or {}


def get_output_config(event: dict) -> dict:
    """Return the normalized output_config from the event."""
    return (event.get("output") or {}).get("output_config") or {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate_required(event: dict) -> None:
    if not event.get("uuid"):
        raise ValueError("uuid is required")

    inp = event.get("input") or {}
    if not inp.get("s3uri"):
        raise ValueError("input.s3uri is required")
    parse_s3uri(inp["s3uri"])   # validate format

    out = event.get("output") or {}
    if not out.get("s3uri"):
        raise ValueError("output.s3uri is required")
    parse_s3uri(out["s3uri"])   # validate format


def _merge_input_config(user: dict) -> dict:
    """Deep-merge user input_config over defaults. One level deep per section."""
    d = _INPUT_CONFIG_DEFAULTS

    def _section(name: str) -> dict:
        return {**d[name], **(user.get(name) or {})}

    model = _section("model")
    vad = _section("vad")
    vad["model_id"] = DEFAULT_VAD_MODEL_ID
    asr = _section("asr")
    asr["model_id"] = DEFAULT_ASR_MODEL_ID
    embed = _section("embed")
    clustering = _section("clustering")

    language = user.get("language", d["language"])
    use_partial = user.get("use_partial", d["use_partial"])
    mark_in_out = user.get("mark_in_out", d["mark_in_out"])

    return {
        "language":     language,
        "use_partial":  use_partial,
        "mark_in_out":  mark_in_out,
        "model":        model,
        "vad":          vad,
        "asr":          asr,
        "embed":        embed,
        "clustering":   clustering,
    }


def _merge_output_config(user: dict) -> dict:
    """Merge user output_config over defaults."""
    d = _OUTPUT_CONFIG_DEFAULTS
    return {
        **user,
        "speaker_channels": bool(user.get("speaker_channels", d["speaker_channels"])),
        "channel_mode":     str(user.get("channel_mode",      d["channel_mode"])),
        # Hidden settings
        # "merge_gap_sec":    d["merge_gap_sec"],
        # "fade_ms":          d["fade_ms"],
        # "compact_gap_sec":  d["compact_gap_sec"],
    }


def _validate_input_config(ic: dict) -> None:
    # -- vad ----------------------------------
    vad = ic["vad"]

    fixed_chunk_size_secs = vad.get("fixed_chunk_size_secs")
    if fixed_chunk_size_secs is not None and int(fixed_chunk_size_secs) < 60:
        raise ValueError(
            f"input_config.vad.fixed_chunk_size_secs must be >= 60 (got {fixed_chunk_size_secs!r})"
        )

    dynamic_speech_chunk_secs = vad.get("dynamic_speech_chunk_secs")
    if dynamic_speech_chunk_secs is not None and int(dynamic_speech_chunk_secs) < 60:
        raise ValueError(
            f"input_config.vad.dynamic_speech_chunk_secs must be >= 60 (got {dynamic_speech_chunk_secs!r})"
        )

    # -- asr ----------------------------------
    asr = ic["asr"]
    num_workers = asr.get("num_workers")
    if num_workers is not None and (int(num_workers) > 2 or int(num_workers) < 1):
        raise ValueError(
            f"input_config.asr.num_workers must be between 1-2 (got {num_workers!r})"
        )

    beam_size = asr.get("beam_size")
    if beam_size is not None and (int(beam_size) > 5 or int(beam_size) < 1):
        raise ValueError(
            f"input_config.asr.beam_size must be between 1-5 (got {beam_size!r})"
        )

    language_detection_num_probes = asr.get("language_detection_num_probes")
    if language_detection_num_probes is not None and (int(language_detection_num_probes) < 3):
        raise ValueError(
            f"input_config.asr.language_detection_num_probes should be >= 3 (got {language_detection_num_probes!r})"
        )

    language_detection_min_sec = asr.get("language_detection_min_sec")
    if language_detection_min_sec is not None and (float(language_detection_min_sec) < 6.0):
        raise ValueError(
            f"input_config.asr.language_detection_min_sec should be >= 6.0s (got {language_detection_min_sec!r})"
        )

    # -- embed ----------------------------------
    # min_embedding_duration_sec
    # soft_embedding_min_sec

    # -- clustering ----------------------------------
    clustering = ic["clustering"]

    algorithm = clustering.get("algorithm")
    if algorithm not in _VALID_ALGORITHMS:
        raise ValueError(
            f"input_config.clustering.algorithm={algorithm!r} is invalid; "
            f"must be one of {sorted(_VALID_ALGORITHMS)}"
        )

    num_speakers = clustering.get("num_speakers")
    if num_speakers is not None and int(num_speakers) < 1:
        raise ValueError(
            f"input_config.clustering.num_speakers must be >= 1 or null (got {num_speakers!r})"
        )

    max_k = clustering.get("max_k")
    if max_k is not None and int(max_k) < 1:
        raise ValueError(
            f"input_config.clustering.max_k must be >= 1 (got {max_k!r})"
        )

    min_cluster_size = clustering.get("min_cluster_size")
    if min_cluster_size is not None and int(min_cluster_size) < 2:
        raise ValueError(
            f"input_config.clustering.min_cluster_size must be >= 2 (got {min_cluster_size!r})"
        )

    min_turns_per_speaker = clustering.get("min_turns_per_speaker")
    if min_turns_per_speaker is not None and int(min_turns_per_speaker) < 2:
        raise ValueError(
            f"input_config.clustering.min_turns_per_speaker must be >= 2 (got {min_turns_per_speaker!r})"
        )

    k_selection_method = clustering.get("k_selection_method")
    if k_selection_method not in _VALID_METHODS:
        raise ValueError(
            f"input_config.clustering.k_selection_method={k_selection_method!r} is invalid; "
            f"must be one of {sorted(_VALID_METHODS)}"
        )

    # -- Mark In/Out ----------------------------------
    # If "use_partial" is enabled, "mark_in_out" must have valid duration
    use_partial = ic["use_partial"]
    mark_in_out = ic["mark_in_out"]
    if use_partial and not (isinstance(mark_in_out, list) and len(mark_in_out) == 2 and (mark_in_out[1] - mark_in_out[0]) > 10):
        raise ValueError(
            f"input_config.use_partial={use_partial!r} or "
            f"input_config.mark_in_out={mark_in_out!r} is invalid; "
        )


def _validate_output_config(oc: dict) -> None:
    channel_mode = oc.get("channel_mode")
    if channel_mode not in _VALID_CHANNEL_MODES:
        raise ValueError(
            f"output_config.channel_mode={channel_mode!r} is invalid; "
            f"must be one of {sorted(_VALID_CHANNEL_MODES)}"
        )
