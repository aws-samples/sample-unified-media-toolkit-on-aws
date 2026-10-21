"""
Input schema validation and normalization for speech-enhancement.

Single source of truth for all user-configurable defaults.
Imports nothing from this project — no circular import risk.

Event shape
───────────
{
  "uuid": "...",
  "input": {
    "s3uri": "s3://bucket/key",
    "input_config": {                 ← optional
      "model_id": "deepfilternet3",   ← fixed value, not configurable
      "fixed_chunk_size_secs": 360,   ← rounded to nearest mult of 10
    }
  },
  "output": {
    "s3uri": "s3://bucket/prefix",
    "output_config": {                ← optional
      "output_resampl_freq": 16000,   ← enhanced WAV sample rate
      "atten_lim_db": null            ← cap on noise attenuation (dB),
                                        full-band post-mix guarantee;
                                        ~0 = bypass, >=100 = unlimited,
                                        null = model default (full)
    }
  },
  "data": {}   ← transient state written by each state handler
}

validate_and_normalize() returns a new event dict with defaults merged
into input_config and output_config so downstream handlers never
re-compute defaults.
"""
from __future__ import annotations

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SR = 48_000           # DeepFilterNet3 operates at 48 kHz
DEFAULT_CHUNK_SEC = 360      # 6 minutes (the original workflow's default)
CHUNK_ALIGNMENT_SECS = 10    # whole multiple of the model's STFT hop
DEFAULT_OUT_FREQ = 16_000    # downstream speech consumers (ASR et al.)

_MODEL_ID = "deepfilternet3"   # fixed — never configurable (family rule:
                               # the model is an implementation detail)

_VALID_SAMPLING_FREQS = frozenset((8000, 16000, 32000, 44100, 48000, 96000))
# atten_lim_db caps how much noise reduction is applied — enforced as a
# full-band post-mix of the original (upstream enhance() semantics):
# < 0.01 dB = bypass, >= 100 dB = "no limit".
_ATTEN_LIM_RANGE = (0.0, 100.0)

# ── Defaults ──────────────────────────────────────────────────────────────────

_INPUT_CONFIG_DEFAULTS: dict = {
    "model_id": _MODEL_ID,
    "fixed_chunk_size_secs": DEFAULT_CHUNK_SEC,
}

_OUTPUT_CONFIG_DEFAULTS: dict = {
    "output_resampl_freq": DEFAULT_OUT_FREQ,
    "atten_lim_db": None,
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
    Validate required fields and fill defaults into input_config and
    output_config. Raises ValueError on missing/invalid values.
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
        "input": {**event["input"], "input_config": input_config},
        "output": {**event["output"], "output_config": output_config},
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
    parse_s3uri(inp["s3uri"])

    out = event.get("output") or {}
    if not out.get("s3uri"):
        raise ValueError("output.s3uri is required")
    parse_s3uri(out["s3uri"])


def _merge_input_config(user: dict) -> dict:
    d = _INPUT_CONFIG_DEFAULTS
    return {
        "model_id": d["model_id"],
        "fixed_chunk_size_secs": user.get("fixed_chunk_size_secs",
                                          d["fixed_chunk_size_secs"]),
    }


def _merge_output_config(user: dict) -> dict:
    d = _OUTPUT_CONFIG_DEFAULTS
    return {
        "output_resampl_freq": user.get("output_resampl_freq",
                                        d["output_resampl_freq"]),
        "atten_lim_db": user.get("atten_lim_db", d["atten_lim_db"]),
    }


def _validate_input_config(ic: dict) -> None:
    secs = ic.get("fixed_chunk_size_secs")
    if secs is not None and int(secs) < CHUNK_ALIGNMENT_SECS:
        raise ValueError(
            f"input_config.fixed_chunk_size_secs must be >= "
            f"{CHUNK_ALIGNMENT_SECS} (got {secs!r})")


def _validate_output_config(oc: dict) -> None:
    freq = oc.get("output_resampl_freq")
    if freq is not None and freq not in _VALID_SAMPLING_FREQS:
        raise ValueError(
            f"output_config.output_resampl_freq={freq!r} is invalid; "
            f"must be one of {sorted(_VALID_SAMPLING_FREQS)}")

    atten = oc.get("atten_lim_db")
    if atten is not None:
        lo, hi = _ATTEN_LIM_RANGE
        if not isinstance(atten, (int, float)) or isinstance(atten, bool) \
                or not (lo <= float(atten) <= hi):
            raise ValueError(
                f"output_config.atten_lim_db={atten!r} is invalid; "
                f"must be a number in [{lo:g}, {hi:g}] or null")
