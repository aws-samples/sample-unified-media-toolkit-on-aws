"""
StateCheckAndPreprocess — first state in the pipeline.

Validates the event via input_schema, then inspects the source file and
decides which of three paths to take:

Fast-path (data.needs_conversion = False):
  Source is a RIFF/WAVE PCM or IEEE-float file, exactly 48 kHz, mono.
  Writes data.audio_bucket / audio_key / audio_cfg and continues directly to
  StateFixedChunkingForSpeechEnhancement via the "Needs conversion?" Choice state.

WAV-normalise path (data.needs_conversion = False, but conversion happened):
  Source IS a WAV but wrong sample rate, multi-channel, or both.
  wav_to_wav.convert() reads, resamples to 48 kHz mono int16, and writes the
  normalised file to S3 in-Lambda using concurrent multipart upload.
  No child state machine invoked. Output format is identical to the fast-path.

Media-processing path (data.needs_conversion = True):
  Source is not a WAV (MP4, MXF, MKV, …) or WAV header parse failed.
  Writes data.media_processing_request; the held state submits it to the
  media-processing child execution, then routes through StateExtractProcessedAudio.

Decision logic
──────────────
  extension ≠ .wav                      → media-processing path
  .wav, header parse fails              → media-processing path
  .wav, mono, sr == 48 kHz             → fast-path
  .wav, but not mono or sr ≠ 48 kHz   → WAV-normalise path (wav_to_wav)

Output  (data fields written by all paths)
──────────────────────────────────────────
  model_id, fixed_chunk_size_secs       ← from input_config
  output_resampl_freq, atten_lim_db     ← from output_config
  audio_bucket, audio_key             ← source of 48 kHz mono WAV
  output_bucket, output_prefix        ← destination for all pipeline outputs
  needs_conversion: bool
  audio_cfg: dict | None              ← set on fast-path and WAV-normalise path
  media_processing_request: dict | None ← set on media-processing path only
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
    parse_s3uri
)
from lib.s3_utils import put_json
from states.check_and_preprocess.check_helper import check_source, build_media_processing_request
from states.check_and_preprocess import wav_to_wav

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def lambda_handler(event: dict, context) -> dict:
    log.info("StateCheckAndPreprocess event keys = %s", sorted(event.keys()))
    t0 = time.monotonic()

    try:
        normalized_event = validate_and_normalize(event)
        uuid = normalized_event["uuid"]
        input = normalized_event["input"]
        output = normalized_event["output"]
        data = normalized_event["data"]
        input_config = get_input_config(normalized_event)
        output_config = get_output_config(normalized_event)

        audio_bucket,  audio_key = parse_s3uri(input["s3uri"])
        output_bucket, output_prefix = parse_s3uri(output["s3uri"])

        log.info("Probing s3://%s/%s", audio_bucket, audio_key)
        result = check_source(audio_bucket, audio_key)

        if result["is_compatible_wav"]:
            info = result["info"]
            log.info(
                "Fast-path: compatible WAV (sr=%d, ch=%d, bits=%d, dur=%.1fs)",
                info.sample_rate, info.num_channels, info.bits_per_sample,
                info.duration_sec,
            )
            audio_cfg = {
                "sample_rate":     info.sample_rate,
                "num_channels":    info.num_channels,
                "bits_per_sample": info.bits_per_sample,
                "duration_sec":    round(info.duration_sec, 3),
            }
            out_data = {
                "needs_conversion": False,
                "audio_cfg": audio_cfg,
            }

        elif result["info"] is not None:
            # WAV file but wrong sample rate or not mono — normalise in-Lambda
            info = result["info"]
            reason = result["reason"]
            output_key = f"{output_prefix}/resampled/audio_48k_mono.wav"
            log.info(
                "WAV-normalise path: %s  (sr=%d, ch=%d, dur=%.1fs) → s3://%s/%s",
                reason, info.sample_rate, info.num_channels, info.duration_sec,
                output_bucket, output_key,
            )
            audio_cfg = wav_to_wav.convert(
                src_bucket=audio_bucket,
                src_key=audio_key,
                dst_bucket=output_bucket,
                dst_key=output_key,
                src_info=info,
            )
            out_data = {
                "needs_conversion": False,
                "audio_cfg":        audio_cfg,
            }
            # Update audio_bucket and audio_key
            audio_bucket = output_bucket
            audio_key = output_key

        else:
            # Not a WAV at all — hand off to media-processing state machine
            reason = result["reason"]
            log.info("Media-processing path: %s", reason)
            media_processing_request = build_media_processing_request(
                src_bucket=audio_bucket,
                src_key=audio_key,
            )
            out_data = {
                "needs_conversion":         True,
                "media_processing_request": media_processing_request,
                "audio_cfg":                None,
            }

        model_id = input_config["model_id"]
        fixed_chunk_size_secs = int(input_config["fixed_chunk_size_secs"])

        output_resampl_freq = int(output_config["output_resampl_freq"])
        atten_lim_db = output_config.get("atten_lim_db")

        workflow_params = {
            "uuid":   uuid,
            "input":  input,
            "output": output,
            "data": {
                **data,
                # input/input_config
                "model_id":              model_id,
                "fixed_chunk_size_secs": fixed_chunk_size_secs,
                "audio_bucket":          audio_bucket,
                "audio_key":             audio_key,
                # output/output_config
                "output_bucket":         output_bucket,
                "output_prefix":         output_prefix,
                "output_resampl_freq":   output_resampl_freq,
                "atten_lim_db":          atten_lim_db,
                # audio_cfg / media_processing_request
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
