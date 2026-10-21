"""
Input schema validation and normalization for audiostem-separation.

Single source of truth for all user-configurable defaults.
Imports nothing from this project — no circular import risk.

Event shape
───────────
{
  "uuid": "...",
  "input": {
    "s3uri": "s3://bucket/key",
    "input_config": {                 ← optional; user overrides for model/
      "model_id": "bandit_v2",        ← Fixed value
      "fixed_chunk_size_secs": 180,   ← rounded to nearest mult of 8
    }
  },
  "output": {
    "s3uri":         "s3://bucket/prefix",
    "output_config": {         ← optional; user overrides for output stems
      "speech_stem":           true,
      "speech_resampl_freq":   16000,

      "music_stem":            true,
      "music_resampl_freq":    48000,

      "sfx_stem":              true,
      "sfx_resampl_freq":      32000,

      "speech_safe":              false,
      "speech_safe_resampl_freq": 48000,
    }
  },
  "data":     {}   ← transient/intermediate state written by each state handler
}

validate_and_normalize() returns a new event dict with defaults merged into input_config
and output_config so all downstream state handlers can read from those locations without
re-computing defaults.
"""
from __future__ import annotations

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SR           = 48_000  # bandit_v2 requires 48 kHz
DEFAULT_CHUNK_SEC   = 180     # 3 minutes
CHUNK_ALIGMENT_SECS = 8       # Bandit v2 internal chunk alignment size — do not change
DEFAULT_SPEECH_FREQ = 16_000  # pyannote + Whisper
DEFAULT_SFX_FREQ    = 32_000  # PANNs CNN14 / HTS-AT
DEFAULT_MUSIC_FREQ  = 48_000  # native Bandit output — no resample needed
# speech_safe is consumed by playback and mixing (AD mixes, review/QC,
# dialogue-forward renders) — full fidelity by default, unlike the
# ASR-leaning speech stem default above.
DEFAULT_SPEECH_SAFE_FREQ = 48_000

_MODEL_ID = "bandit_v2"        # Bandit model id - not configurable

_VALID_SAMPLING_FREQS = frozenset((8000, 16000, 32000, 44100, 48000, 96000))

# ── Defaults ──────────────────────────────────────────────────────────────────

_INPUT_CONFIG_DEFAULTS: dict = {
    "model_id":             _MODEL_ID,
    "fixed_chunk_size_secs": DEFAULT_CHUNK_SEC,
}

_OUTPUT_CONFIG_DEFAULTS: dict = {
    "speech_stem": True,
    "speech_resampl_freq": 16000,

    "music_stem": True,
    "music_resampl_freq": 48000,

    "sfx_stem": True,
    "sfx_resampl_freq": 32000,

    # Gated mix substitution (Design record §5) — opt-in, additive:
    # audio_speech_safe.wav alongside the stems, never replacing them.
    "speech_safe": False,
    "speech_safe_resampl_freq": DEFAULT_SPEECH_SAFE_FREQ,
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

    return {
        "model_id":              d["model_id"],
        "fixed_chunk_size_secs": user.get("fixed_chunk_size_secs", d["fixed_chunk_size_secs"]),
    }


def _merge_output_config(user: dict) -> dict:
    """Merge user output_config over defaults."""
    d = _OUTPUT_CONFIG_DEFAULTS
    return {
        "speech_stem":         user.get("speech_stem", d["speech_stem"]),
        "speech_resampl_freq": user.get("speech_resampl_freq", d["speech_resampl_freq"]),

        "music_stem":          user.get("music_stem", d["music_stem"]),
        "music_resampl_freq":  user.get("music_resampl_freq", d["music_resampl_freq"]),

        "sfx_stem":            user.get("sfx_stem", d["sfx_stem"]),
        "sfx_resampl_freq":    user.get("sfx_resampl_freq", d["sfx_resampl_freq"]),

        "speech_safe":              user.get("speech_safe", d["speech_safe"]),
        "speech_safe_resampl_freq": user.get("speech_safe_resampl_freq",
                                             d["speech_safe_resampl_freq"]),
    }


def _validate_input_config(ic: dict) -> None:
    fixed_chunk_size_secs = ic.get("fixed_chunk_size_secs")
    if fixed_chunk_size_secs is not None and int(fixed_chunk_size_secs) < 60:
        raise ValueError(
            f"input_config.fixed_chunk_size_secs must be >= 60 (got {fixed_chunk_size_secs!r})"
        )


def _validate_output_config(oc: dict) -> None:
    # Refuse-before-spend: validated in state 1, BEFORE the
    # media-processing conversion is submitted — an all-disabled request
    # must never pay for a conversion it cannot use. (FixedChunking
    # keeps the same check as defense in depth.)
    if not any((oc.get("speech_stem"), oc.get("music_stem"),
                oc.get("sfx_stem"), oc.get("speech_safe"))):
        raise ValueError(
            "output_config disables every output — at least one of "
            "speech_stem, music_stem, sfx_stem, speech_safe must be enabled"
        )
    for name, sampl_name in (("speech_stem", "speech_resampl_freq"),
                             ("music_stem", "music_resampl_freq"),
                             ("sfx_stem", "sfx_resampl_freq"),
                             ("speech_safe", "speech_safe_resampl_freq")):
        stem = oc.get(name)
        if stem is not None and not isinstance(stem, bool):
            raise ValueError(
                f"output_config.{name}={stem!r} is invalid; "
                f"must be boolean"
            )

        resampl_freq = oc.get(sampl_name)
        if resampl_freq is not None and resampl_freq not in _VALID_SAMPLING_FREQS:
            raise ValueError(
                f"output_config.{sampl_name}={resampl_freq!r} is invalid; "
                f"must be one of {sorted(_VALID_SAMPLING_FREQS)}"
            )

