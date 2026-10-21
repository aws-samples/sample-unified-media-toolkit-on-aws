"""
StateCheckAndPreprocess — first state in the pipeline.

Inspects the source file and decides how to proceed:

Fast-path (data.needs_conversion = False):
  Source is a RIFF/WAVE PCM or IEEE-float file, mono, sample_rate >= 16000.
  Writes data.audio_key / audio_bucket / audio_cfg and proceeds directly to
  StateFixedChunkingForVAD via the "Needs conversion?" Choice state.

WAV-normalise path (data.needs_conversion = False, but audio was resampled):
  Source IS a WAV but wrong sample rate or not mono (e.g. 48 kHz stereo).
  wav_to_wav.convert() reads, resamples, and writes the normalised file to
  S3 in-Lambda using concurrent multipart upload.  No child state machine
  invoked.  Output is identical to the fast-path.

Media-processing path (data.needs_conversion = True):
  Source is not a WAV at all (MP4, MXF, MKV, …).
  Writes data.media_processing_request (gateway params); the held
  StateSubmitMediaProcessing submits it to media-processing's GATEWAY
  with a callback token, then routes through StateExtractProcessedAudio.

Decision logic
──────────────
  extension ≠ .wav                 → media-processing path
  .wav, header parse fails         → media-processing path
  .wav, mono, sr ≥ 16 kHz         → fast-path
  .wav, but not mono or sr < 16k  → WAV-normalise path (wav_to_wav)
"""
from __future__ import annotations

import json
import logging
import time
import traceback

from lib.input_schema import (
    validate_and_normalize,
    get_input_config,
    get_output_config,
    parse_s3uri,
)
from lib.s3_utils import put_json
from states.check_and_preprocess.check_helper import check_source, build_media_processing_request, resolve_mark_in_out
from states.check_and_preprocess import wav_to_wav

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    log.info("StateCheckAndPreprocess event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        # Reshape the input params
        normalized_event = validate_and_normalize(event)

        uuid = normalized_event["uuid"]
        input = normalized_event["input"]
        output = normalized_event["output"]
        data = normalized_event.get("data", {})
        input_config = get_input_config(normalized_event)
        output_config = get_output_config(normalized_event)

        audio_bucket,  audio_key = parse_s3uri(input["s3uri"])
        output_bucket, output_prefix = parse_s3uri(output["s3uri"])
        output_prefix = output_prefix.rstrip("/")

        log.info("Probing s3://%s/%s", audio_bucket, audio_key)
        result = check_source(audio_bucket, audio_key)
        is_compatible_wav = result["is_compatible_wav"]
        wav_info = result["info"]

        # Resolve the mark_in_out and use_partial settings based on the content duration
        use_partial = input_config["use_partial"]
        mark_in_out = input_config["mark_in_out"]
        if wav_info is not None:
            use_partial, mark_in_out = resolve_mark_in_out(
                input_config["use_partial"],
                input_config["mark_in_out"],
                wav_info.duration_sec,
            )

        mark_in, mark_out = mark_in_out

        if is_compatible_wav:
            if use_partial:
                dst_key = f"{output_prefix}/clipped/audio_16k_mono.wav"
                audio_cfg = wav_to_wav.convert(
                    src_bucket=audio_bucket,
                    src_key=audio_key,
                    dst_bucket=output_bucket,
                    dst_key=dst_key,
                    src_info=wav_info,
                    use_partial=use_partial,
                    mark_in_out=mark_in_out,
                )
                out_data = {
                    "needs_conversion": False,
                    "audio_bucket":     output_bucket,
                    "audio_key":        dst_key,
                    "audio_cfg":        audio_cfg,
                }
            else:
                audio_cfg = {
                    "sample_rate":     wav_info.sample_rate,
                    "num_channels":    wav_info.num_channels,
                    "bits_per_sample": wav_info.bits_per_sample,
                    "duration_sec":    round(wav_info.duration_sec, 3),
                }
                out_data = {
                    "needs_conversion": False,
                    "audio_bucket":     audio_bucket,
                    "audio_key":        audio_key,
                    "audio_cfg":        audio_cfg,
                }

            log.info(
                "Fast-path: compatible WAV (sr=%d, ch=%d, bits=%d, dur=%.1fs)"
                "  %s mark_in=%d mark_out=%d clipped_dur=%.1fs",
                wav_info.sample_rate, wav_info.num_channels, wav_info.bits_per_sample, wav_info.duration_sec,
                "trimmed" if use_partial else "no-trim",
                mark_in, mark_out, audio_cfg["duration_sec"],
            )

        elif wav_info is not None:
            # WAV file but wrong sample rate or not mono — normalise in-Lambda
            reason = result["reason"]
            dst_key = f"{output_prefix}/resampled/audio_16k_mono.wav"
            audio_cfg = wav_to_wav.convert(
                src_bucket=audio_bucket,
                src_key=audio_key,
                dst_bucket=output_bucket,
                dst_key=dst_key,
                src_info=wav_info,
                use_partial=use_partial,
                mark_in_out=mark_in_out,
            )
            log.info(
                "WAV-normalise path: %s  (sr=%d, ch=%d, dur=%.1fs) → s3://%s/%s"
                "  %s mark_in=%d mark_out=%d clipped_dur=%.1fs",
                reason, wav_info.sample_rate, wav_info.num_channels, wav_info.duration_sec,
                output_bucket, dst_key,
                "trimmed" if use_partial else "no-trim",
                mark_in, mark_out, audio_cfg["duration_sec"],
            )
            out_data = {
                "needs_conversion": False,
                "audio_bucket":     output_bucket,
                "audio_key":        dst_key,
                "audio_cfg":        audio_cfg,
            }

        else:
            # Not a WAV at all — hand off to media-processing via its
            # GATEWAY (held task token). No duration available here; pass
            # raw partial-clip values through — media-processing does its
            # own duration probe.
            reason = result["reason"]
            log.info("Media-processing path: %s", reason)
            out_data = {
                "needs_conversion":         True,
                "media_processing_request": build_media_processing_request(
                    src_bucket=audio_bucket,
                    src_key=audio_key,
                    use_partial=use_partial,
                    mark_in_out=mark_in_out,
                ),
            }

        workflow_params = {
            "uuid":   uuid,
            "input":  input,
            "output": output,
            "data": {
                **data,
                # Unpack all workflow configs
                "language":      input_config["language"],
                "use_partial":   use_partial,
                "mark_in_out":   mark_in_out,
                "model_cfg":     input_config["model"],
                "vad_cfg":       input_config["vad"],
                "asr_cfg":       input_config["asr"],
                "embed_cfg":     input_config["embed"],
                "clu_cfg":       input_config["clustering"],
                # output configuration
                "output_bucket": output_bucket,
                "output_prefix": output_prefix,
                **output_config,
                **out_data,
            },
        }

        # Store workflow params to output bucket
        workflow_key = f"{output_prefix}/workflow_params.json"
        put_json(output_bucket, workflow_key, workflow_params)

        elapsed = round(time.monotonic() - t0, 3)
        log.info("StateCheckAndPreprocess completed in %.3fs — needs_conversion=%s",
                 elapsed, out_data["needs_conversion"])
        return workflow_params

    except Exception as exc:
        log.error("StateCheckAndPreprocess failed — %s: %s",
                  type(exc).__name__, exc)
        traceback.print_exc()
        raise
