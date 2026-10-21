"""
StateExtractProcessedAudio — runs after the media-processing run completes.

Reads data.media_processing_output — the GATEWAY submit handle
{job_id, workflow_id, asset_s3uri, output_s3_prefix} + terminal `status`
that MP's callback waker delivered to the held submit state (status AS
DATA; the FAILED verdict is enforced HERE, since MP is a hard
dependency). Derives the wav key from MP's output contract
({output_s3_prefix}/speech_enhancement/audio_48k_mono.wav), probes the WAV
header, and writes data.audio_bucket / audio_key / audio_cfg /
media_processing_arn / media_processing_usage_key so the rest of the
pipeline proceeds identically to the WAV fast-path.

The legacy nested-execution shape (pre-2026-09-18 ResultSelector output)
was retired 2026-09-20; a non-handle shape raises.

After this state, data.media_processing_request and
data.media_processing_output are set to None to keep the Step Functions
event payload lean.
"""
from __future__ import annotations

import json
import logging
import traceback

from lib.exceptions import NoAudioStream

from lib.s3_utils import s3 as _s3
from lib.wav_utils import parse_wav_header

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _normalize_verdict(mp_out: dict) -> tuple:
    """(code, inner_message) for a FAILED handle, across BOTH completion
    shapes. The waker forwards error = the raising class name and cause =
    a Lambda error JSON (errorMessage is the sentence that matters,
    truncated). DEPLOY SKEW is why BOTH shapes are read: MP deployments
    at or after 2026-10-15 emit the one waker shape on every completion
    path, but an OLDER MP's absorbed-into-finished synchronous completion
    puts the whole cause JSON into `error` with no `cause` key — so a
    field that parses as that JSON yields errorType/errorMessage,
    whichever field it arrived in. Do not delete the dual parse until no
    pre-2026-10-15 MP deployment can call back. Older waker deploys carry
    neither field; then both halves are empty and the caller falls back
    honestly."""
    code = str(mp_out.get("error") or "")
    inner = ""
    parsed_a_dict = False
    for raw in (mp_out.get("cause"), code):
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            # errorMessage may be absent — the waker's compact cause keeps
            # whichever of errorType/errorMessage the execution carried,
            # and an exception raised with empty args has only the type
            parsed_a_dict = True
            inner = str(parsed.get("errorMessage") or "")
            if parsed.get("errorType"):
                code = str(parsed["errorType"])
            break
    if not inner and not parsed_a_dict:
        # only an UNPARSEABLE cause rides along raw (truncated); a parsed
        # dict with no errorMessage already gave everything it had in code
        inner = str(mp_out.get("cause") or "")[:300]
    return code, inner


def lambda_handler(event: dict, context) -> dict:
    log.info("StateExtractProcessedAudio start")
    try:
        data   = event.get("data", {})
        mp_out = data.get("media_processing_output")
        if not mp_out:
            raise ValueError("data.media_processing_output is missing")

        if "output_s3_prefix" not in mp_out:
            # the legacy nested-execution shape retired 2026-09-20 —
            # every run now parks on MP's gateway and is completed with
            # the submit handle
            raise ValueError(
                f"data.media_processing_output is not a gateway handle "
                f"(got: {mp_out})")
        # MP's waker completed the held state with the submit handle +
        # terminal status AS DATA — a FAILED verdict is ours to enforce,
        # and it is enforced BY NAME: a source with no audio stream raises
        # NoAudioStream (the stable code agents branch on), everything
        # else re-raises with MP's own error/cause passed through.
        status = mp_out.get("status")
        if status != "SUCCEEDED":
            code, inner = _normalize_verdict(mp_out)
            # this consumer's MP request is ALWAYS audio-only (the wav
            # speech_enhancement group), so NoCompatibleOutputGroups can
            # only ever mean "no audio stream" here — the class alone is
            # decisive, even when no message survived the completion path.
            # The wording guard remains only for the legacy ValueError
            # shape (MP deploys older than the exception rename).
            no_audio = (code == "NoCompatibleOutputGroups"
                        or (code == "ValueError"
                            and "No output_groups are compatible" in inner
                            and "no audio" in inner))
            if no_audio:
                # a terminal content fact with a stable code: the agent
                # probes first next time and skips the audio chain —
                # failing HERE (before any model state) is the family's
                # refuse-before-spend rule (2026-10-15, the speech-
                # diarization decision applied family-wide)
                raise NoAudioStream(
                    "the source has no audio stream — nothing to enhance"
                    + (f" ({inner})" if inner else ""))
            detail = ": ".join(x for x in (code, inner) if x) \
                or "the waker carried no error detail — see the MP run"
            raise RuntimeError(
                f"media-processing run {mp_out.get('workflow_id')} "
                f"ended {status}: {detail}")
        prefix = mp_out["output_s3_prefix"]
        audio_bucket, _, key_prefix = prefix[5:].partition("/")
        # deterministic per MP's output contract:
        # {prefix}/{group_name}/{name}.wav — both set by our request
        audio_key = f"{key_prefix}/speech_enhancement/audio_48k_mono.wav"
        usage_key = f"{key_prefix}/usage_tracking.json"
        media_processing_arn = mp_out.get("job_id")

        audio_cfg = _probe_audio_cfg(audio_bucket, audio_key)

        log.info(
            "Extracted: s3://%s/%s  sr=%d  ch=%d  dur=%.1fs",
            audio_bucket, audio_key,
            audio_cfg["sample_rate"], audio_cfg["num_channels"],
            audio_cfg["duration_sec"],
        )

        return {
            **event,
            "data": {
                **data,
                # Override audio_bucket, audio_key from media-processing workflow
                "audio_bucket": audio_bucket, "audio_key": audio_key,
                "audio_cfg": audio_cfg,
                "media_processing_arn": media_processing_arn,
                # usage tracking reads MP's own published bill from here
                "media_processing_usage_key": usage_key,
                # clear both fields to keep the SFN event payload lean
                "media_processing_request": None,
                "media_processing_output": None,
            },
        }

    except Exception as exc:
        log.error("StateExtractProcessedAudio failed — %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        raise


# ── Helpers ───────────────────────────────────────────────────────────────────

def _probe_audio_cfg(bucket: str, key: str) -> dict:
    """
    Fetch the WAV header (64 KB byte-range GET) and return audio_cfg dict.
    Raises on parse failure — the media-processing output is always expected
    to be a valid 48 kHz mono WAV and any error here is a pipeline bug.
    """
    resp = _s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-65535")
    raw  = resp["Body"].read()
    info = parse_wav_header(raw)
    return {
        "sample_rate":     info.sample_rate,
        "num_channels":    info.num_channels,
        "bits_per_sample": info.bits_per_sample,
        "duration_sec":    round(info.duration_sec, 3),
    }
