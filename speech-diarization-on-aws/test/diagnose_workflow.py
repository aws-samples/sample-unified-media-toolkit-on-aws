#!/usr/bin/env python3
"""
diagnose_workflow.py — Generic speech-diarization workflow diagnostic, from
an execution ARN alone. Runs in the CUSTOMER's account/region.

Not one fixed diagnosis: a small framework of independent CHECKS, each
inspecting the same shared context (execution history, S3 outputs) and
reporting zero or more Findings. Adding a new symptom means adding one
function to CHECKS — everything else (report formatting, log correlation,
audio clipping, packaging) is shared.

Checks shipped today
─────────────────────
  check_zero_word_groups     — "no words assigned to group" (lib/transcribe.py
                                batch_worker()); the original, most common
                                complaint this script was built for.
  check_speaker_clustering   — wrong speaker COUNT (single-speaker misfire,
                                elbow/combined K disagreement, unreliable
                                silhouette), wrong speaker ASSIGNMENT (diffuse
                                clusters, rapid same-window flip-flops, heavy
                                LOW/SKIPPED fallback), and over/under-
                                clustering (flat or ceiling-pinned K-sweep
                                curve) — all read directly from
                                embeddings/global_speakers.json, which the
                                pipeline already computes; no new
                                instrumentation needed.
  check_language_detection   — probe disagreement across candidate languages.
  check_missing_outputs      — an expected output file never got written.

Sources combined (unchanged from the original script's design)
────────────────────────────────────────────────────────────────
  1. Step Functions execution history (GetExecutionHistory) — pipeline
     status, per-state retry/failure counts, precise start/end timestamps
     for the exact Map iteration behind any region of interest.
  2. S3 outputs under the execution's output_prefix.
  3. (Best-effort) CloudWatch Logs Insights on the transcription Lambda's
     log group, scoped to the exact time window of an affected region.
  4. (Optional, --clip-audio) A local .wav clip of each region's time
     window, byte-range GET directly from data.audio_bucket/audio_key.
  5. (Optional, --package) A tar.gz bundling the report, execution history,
     and every S3 file actually consulted by a check that fired a finding —
     for handing off to someone who can't (or shouldn't) run this script
     themselves.

Because this runs in the customer's account, it needs (beyond what the
workflow Lambda role already has per README.md):
  - states:DescribeExecution, states:GetExecutionHistory on the speech-
    diarization state machine's executions (already granted to the gateway
    tool role; may need adding for whoever runs this script)
  - logs:StartQuery, logs:GetQueryResults, logs:DescribeLogGroups on the
    transcription Lambda's log group (optional — log correlation degrades
    gracefully without it)
  - s3:GetObject, s3:ListBucket on the output bucket (already granted)
  - s3:GetObject on data.audio_bucket (only needed for --clip-audio). In
    the media-processing path this is the same ProxyBucket already granted;
    in the WAV fast-path it may be the customer's IngestBucket, which the
    workflow Lambda role already has read-only s3:GetObject on per
    README.md.

Map-iteration-index caveat
───────────────────────────
dyn_chunk_idx (assigned by StateDynamicChunkingBySpeechTurns, 0-based in
build order) is assumed to equal the Map "Map transcribe processing"
iteration index ($$.Map.Item.Index), since the iterators array is never
reordered between that state and the Map. This holds in every normal run;
it would only break if iterators were resorted or filtered in between,
which nothing in this pipeline does.

Usage
-----
  python diagnose_workflow.py --execution-arn <arn> [--region us-east-1]

  # Run only specific checks (default: all)
  python diagnose_workflow.py --execution-arn <arn> --checks zero_word_groups,speaker_clustering

  # Skip the CloudWatch Logs Insights step (e.g. no logs:StartQuery permission)
  python diagnose_workflow.py --execution-arn <arn> --skip-logs

  # Override output location (execution failed before writing data.output_bucket,
  # or you want to point at a different location than the execution reports)
  python diagnose_workflow.py --execution-arn <arn> --output-s3uri s3://bucket/prefix

  # Also download a local .wav clip of each finding's audio window
  python diagnose_workflow.py --execution-arn <arn> --clip-audio --clip-dir ./clips

  # Bundle everything consulted into one tar.gz for a deeper offline look
  python diagnose_workflow.py --execution-arn <arn> --package ./diagnosis.tar.gz
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import tarfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import boto3
from botocore.exceptions import ClientError

# Languages with no word-boundary spaces — Whisper timestamps come from
# sub-word tokens, making soft-boundary word assignment less reliable.
# See lib/model_registry.py CHAR_BASED_LANGUAGES.
_CHAR_BASED_LANGUAGES = {"zh", "ja", "ko", "th"}

# Extra seconds of context around each region's own window when clipping
# audio — long enough to hear the surrounding speech/silence, short enough
# that clips stay small.
_CLIP_PAD_SEC = 2.0

_SCHEDULED_TYPES = {"TaskScheduled", "LambdaFunctionScheduled"}
_SUCCEEDED_TYPES = {"TaskSucceeded", "LambdaFunctionSucceeded"}
_FAILED_TYPES = {
    "TaskFailed", "LambdaFunctionFailed",
    "TaskTimedOut", "LambdaFunctionTimedOut",
    "TaskStartFailed", "LambdaFunctionStartFailed",
}

# CloudWatch Logs Insights query window padding around a Map iteration's
# known start/end — covers small clock skew between SFN and Lambda logs.
_LOG_WINDOW_PAD_SEC = 5

# Cluster cohesion below this (mean intra-cluster cosine similarity) is
# "acoustically diffuse" — see lib/cluster_score.py _compute_cluster_confidence
# docstring, which uses the same 0.70 threshold in its own comment.
_DIFFUSE_CLUSTER_CONFIDENCE = 0.70

# Rapid speaker flip-flop detector: consecutive sentences shorter than this,
# with a gap below this, alternating speakers, are more likely centroid-
# boundary jitter than genuine fast turn-taking.
_FLIPFLOP_MAX_SENTENCE_SEC = 2.0
_FLIPFLOP_MAX_GAP_SEC = 0.5
_FLIPFLOP_MIN_RUN = 3  # alternations in a row before flagging

# Fraction of total sentences assigned via LOW-centroid or SKIPPED-overlap
# fallback (rather than direct AHC clustering) above which the transcript
# is running on meaningfully weaker signal.
_HIGH_FALLBACK_FRACTION = 0.35


# =============================================================================
# ARGPARSE / TOP-LEVEL
# =============================================================================

def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not a valid S3 URI: {uri}")
    parts = uri[5:].split("/", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"S3 URI must include a key/prefix: {uri}")
    return parts[0], parts[1].rstrip("/")


def fmt_ts(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    h, total_ms = divmod(total_ms, 3_600_000)
    m, total_ms = divmod(total_ms, 60_000)
    s, ms = divmod(total_ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def epoch(ts) -> float:
    return ts.timestamp() if hasattr(ts, "timestamp") else float(ts)


# =============================================================================
# S3 HELPERS
# =============================================================================

def get_json(s3, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(resp["Body"].read())


def list_transcript_keys(s3, bucket: str, prefix: str) -> list[str]:
    """
    Per-chunk transcripts live under {prefix}/dynamic/transcripts/, NOT
    {prefix}/transcripts/ — StateDynamicChunkingBySpeechTurns rewrites each
    iterator's output_prefix to f"{output_prefix}/dynamic" before handing it
    to the transcription Map (see dynamic_chunking_by_speech_turns/
    lambda_function.py _shape_output()). The top-level {prefix}/transcripts/
    path holds only final_transcript.json/.vtt, written later by
    StateMergeSpeakersTranscripts using the original (un-suffixed) prefix.
    """
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=bucket, Prefix=f"{prefix}/dynamic/transcripts/transcript_"
    ):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                keys.append(obj["Key"])
    return sorted(keys, key=lambda k: int(k.rsplit("_", 1)[-1].split(".")[0]))


# =============================================================================
# WAV CLIPPING (byte-range GET — no ffmpeg/librosa/numpy)
# =============================================================================
#
# The pipeline's own audio (data.audio_bucket / data.audio_key) is always a
# plain 16kHz mono PCM WAV by the time any state past StateCheckAndPreprocess
# runs — see the Source Format Handling section of README.md. That means
# clipping a time window requires no decoding at all: parse the 44-byte
# header once with struct, byte-range GET the exact PCM span for [start,
# end), and prepend a freshly-built 44-byte header sized to that span. The
# clip's samples are bit-identical to the original file.

class _WavHeader:
    __slots__ = ("sample_rate", "num_channels", "bits_per_sample", "data_offset")

    def __init__(self, sample_rate: int, num_channels: int,
                 bits_per_sample: int, data_offset: int):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.bits_per_sample = bits_per_sample
        self.data_offset = data_offset

    @property
    def bytes_per_frame(self) -> int:
        return self.num_channels * (self.bits_per_sample // 8)


def _parse_wav_header(raw: bytes) -> _WavHeader:
    """Minimal RIFF/WAVE header parser — enough to locate the data chunk
    and compute byte offsets. Assumes standard (non-extensible) PCM, which
    is what StateCheckAndPreprocess always normalises source audio to."""
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file (magic: {raw[:4]!r}/{raw[8:12]!r})")

    pos = 12
    sample_rate = num_channels = bits_per_sample = data_offset = None
    while pos + 8 <= len(raw):
        chunk_id = raw[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        body = pos + 8
        if chunk_id == b"fmt ":
            num_channels = struct.unpack_from("<H", raw, body + 2)[0]
            sample_rate = struct.unpack_from("<I", raw, body + 4)[0]
            bits_per_sample = struct.unpack_from("<H", raw, body + 14)[0]
        elif chunk_id == b"data":
            data_offset = body
            break
        pos += 8 + chunk_size + (chunk_size & 1)

    if data_offset is None or sample_rate is None:
        raise ValueError("Could not locate fmt/data chunks in WAV header")
    return _WavHeader(sample_rate, num_channels, bits_per_sample, data_offset)


def _build_wav_header(n_bytes: int, header: _WavHeader) -> bytes:
    """Build a standard 44-byte mono/stereo PCM WAV header for n_bytes of
    audio data matching header's format."""
    byte_rate = header.sample_rate * header.bytes_per_frame
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + n_bytes,
        b"WAVE",
        b"fmt ", 16,
        1, header.num_channels,             # PCM integer
        header.sample_rate, byte_rate,
        header.bytes_per_frame, header.bits_per_sample,
        b"data", n_bytes,
    )


def clip_wav_window(
    s3, bucket: str, key: str,
    start_sec: float, end_sec: float,
) -> bytes:
    """Byte-range GET a [start_sec, end_sec) window from a PCM WAV in S3 and
    return a complete, standalone WAV file (header + that exact PCM span).
    start_sec is clamped to 0; no clamping against file end is attempted —
    S3 range GETs past EOF simply return up to the last byte."""
    head_resp = s3.get_object(Bucket=bucket, Key=key, Range="bytes=0-4095")
    header = _parse_wav_header(head_resp["Body"].read())

    start_sec = max(0.0, start_sec)
    start_frame = int(start_sec * header.sample_rate)
    end_frame = int(end_sec * header.sample_rate)

    byte_start = header.data_offset + start_frame * header.bytes_per_frame
    byte_end = header.data_offset + end_frame * header.bytes_per_frame - 1

    resp = s3.get_object(
        Bucket=bucket, Key=key, Range=f"bytes={byte_start}-{byte_end}")
    pcm = resp["Body"].read()

    return _build_wav_header(len(pcm), header) + pcm


# =============================================================================
# STEP FUNCTIONS — EXECUTION HISTORY
# =============================================================================

def get_execution_history(sfn, execution_arn: str) -> list[dict]:
    events: list[dict] = []
    kwargs = {"executionArn": execution_arn, "maxResults": 1000,
              "reverseOrder": False, "includeExecutionData": True}
    while True:
        resp = sfn.get_execution_history(**kwargs)
        events.extend(resp["events"])
        token = resp.get("nextToken")
        if not token:
            break
        kwargs["nextToken"] = token
    return events


def find_state_name(event: dict, by_id: dict[int, dict]) -> str | None:
    """Walk previousEventId up to the nearest TaskStateEntered — works inside
    Map iterations too (each iteration has its own TaskStateEntered chain)."""
    current_id = event.get("previousEventId")
    seen: set[int] = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        parent = by_id.get(current_id)
        if not parent:
            break
        if parent["type"] == "TaskStateEntered":
            return parent.get("stateEnteredEventDetails", {}).get("name")
        current_id = parent.get("previousEventId")
    return None


def find_map_iteration_index(event: dict, by_id: dict[int, dict]) -> int | None:
    """Walk previousEventId up to the nearest MapIterationStarted — gives the
    exact iteration (chunk) a Task event belongs to, correct even when many
    iterations run concurrently (MaxConcurrency=40)."""
    current_id = event.get("previousEventId")
    seen: set[int] = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        parent = by_id.get(current_id)
        if not parent:
            break
        if parent["type"] == "MapIterationStarted":
            return parent.get("mapIterationStartedEventDetails", {}).get("index")
        current_id = parent.get("previousEventId")
    return None


def get_arn_from_scheduled(event: dict) -> str:
    details = event.get("taskScheduledEventDetails", {})
    raw = details.get("parameters")
    if raw:
        try:
            return json.loads(raw).get("FunctionName", "")
        except (json.JSONDecodeError, AttributeError):
            pass
    return event.get("lambdaFunctionScheduledEventDetails", {}).get("resource", "")


def get_error_cause(event: dict) -> tuple[str | None, str | None]:
    for key in (
        "taskFailedEventDetails", "lambdaFunctionFailedEventDetails",
        "taskTimedOutEventDetails", "lambdaFunctionTimedOutEventDetails",
        "taskStartFailedEventDetails", "lambdaFunctionStartFailedEventDetails",
    ):
        details = event.get(key)
        if details:
            return details.get("error"), details.get("cause")
    return None, None


class ExecutionAnalysis:
    """Parses raw execution history once; answers the questions checks need
    without re-walking the event list for each."""

    def __init__(self, events: list[dict]):
        self.events = events
        self.by_id = {e["id"]: e for e in events}

        # state_name -> list of attempt dicts {status, ts, error, cause, map_index}
        self.state_attempts: dict[str, list[dict]] = {}
        # (state_name, map_index) -> {"start": ts, "end": ts, "status": str}
        self.iteration_windows: dict[tuple[str, int], dict] = {}
        # state_name -> lambda ARN (first one seen)
        self.state_lambda_arn: dict[str, str] = {}

        self._parse()

    def _parse(self) -> None:
        by_id = self.by_id

        # ── Map iteration windows (top-level, flat — index is in the event) ──
        for e in self.events:
            t = e["type"]
            if t == "MapIterationStarted":
                d = e["mapIterationStartedEventDetails"]
                key = (d["name"], d["index"])
                self.iteration_windows.setdefault(key, {})["start"] = epoch(e["timestamp"])
                self.iteration_windows[key]["status"] = "RUNNING"
            elif t == "MapIterationSucceeded":
                d = e["mapIterationSucceededEventDetails"]
                key = (d["name"], d["index"])
                self.iteration_windows.setdefault(key, {})["end"] = epoch(e["timestamp"])
                self.iteration_windows[key]["status"] = "SUCCEEDED"
            elif t in ("MapIterationFailed", "MapIterationAborted"):
                d = e.get("mapIterationFailedEventDetails") or e.get("mapIterationAbortedEventDetails")
                key = (d["name"], d["index"])
                self.iteration_windows.setdefault(key, {})["end"] = epoch(e["timestamp"])
                self.iteration_windows[key]["status"] = t.replace("MapIteration", "").upper()

        # ── Per-state task attempts (Scheduled/Succeeded/Failed), Map-aware ──
        scheduled_by_id: dict[int, dict] = {}
        for e in self.events:
            if e["type"] not in _SCHEDULED_TYPES:
                continue
            state_name = find_state_name(e, by_id)
            if not state_name:
                continue
            map_index = find_map_iteration_index(e, by_id)
            arn = get_arn_from_scheduled(e)
            if arn:
                self.state_lambda_arn.setdefault(state_name, arn)
            attempt = {
                "state_name": state_name,
                "map_index": map_index,
                "scheduled_ts": epoch(e["timestamp"]),
                "succeeded_ts": None,
                # A failed attempt's terminal timestamp is recorded for the same
                # reason the usage_tracking handlers record it: without it a
                # failure's DURATION is unrecoverable, and that duration is what
                # separates "ran to its 900 s timeout" from "raised in 2 s" —
                # the single most useful fact about a retried state. See
                # README.md (Gotchas: usage-tracking invariants).
                "failed_ts": None,
                "status": "SCHEDULED",
                "error": None,
                "cause": None,
            }
            scheduled_by_id[e["id"]] = attempt
            self.state_attempts.setdefault(state_name, []).append(attempt)

        def _find_scheduled_id(event: dict) -> int | None:
            current_id = event.get("previousEventId")
            seen: set[int] = set()
            while current_id and current_id not in seen:
                seen.add(current_id)
                if current_id in scheduled_by_id:
                    return current_id
                parent = by_id.get(current_id)
                if not parent:
                    break
                current_id = parent.get("previousEventId")
            return None

        for e in self.events:
            t = e["type"]
            if t in _SUCCEEDED_TYPES:
                sched_id = _find_scheduled_id(e)
                if sched_id is not None:
                    scheduled_by_id[sched_id]["succeeded_ts"] = epoch(e["timestamp"])
                    scheduled_by_id[sched_id]["status"] = "SUCCEEDED"
            elif t in _FAILED_TYPES:
                sched_id = _find_scheduled_id(e)
                if sched_id is not None:
                    error, cause = get_error_cause(e)
                    scheduled_by_id[sched_id]["status"] = "FAILED"
                    scheduled_by_id[sched_id]["failed_ts"] = epoch(e["timestamp"])
                    scheduled_by_id[sched_id]["error"] = error
                    scheduled_by_id[sched_id]["cause"] = cause

    def attempts_for(self, state_name: str, map_index: int | None = None) -> list[dict]:
        attempts = self.state_attempts.get(state_name, [])
        if map_index is None:
            return attempts
        return [a for a in attempts if a["map_index"] == map_index]

    def iteration_window(self, state_name: str, index: int) -> dict | None:
        return self.iteration_windows.get((state_name, index))

    def all_failures(self) -> list[dict]:
        return [a for attempts in self.state_attempts.values()
                for a in attempts if a["status"] == "FAILED"]


# =============================================================================
# KNOWN-CAUSE LOOKUP
# =============================================================================
# Patterns mined from this repo's own README.md "Gotchas" section — turns a
# raw error/cause string into a plain-English likely cause and suggested
# fix, without needing a support engineer to already know the codebase.

_KNOWN_CAUSES: list[tuple[str, str, str]] = [
    (
        "States.Timeout",
        "A Map iteration (likely StateRunWhisperAlignWtpsplitModels, the "
        "4-core Lambda) ran past its Lambda timeout.",
        "The dynamic speech chunk may be too long for one iteration. Try "
        "lowering input_config.vad.dynamic_speech_chunk_secs (default 360s), "
        "or check whether beam_size/model_variant is set unusually high.",
    ),
    (
        "vbx",
        "clustering.algorithm='vbx' was requested.",
        "VBx is implemented but permanently disabled "
        "(_VALID_ALGORITHMS = frozenset((\"ahc\",)) in lib/input_schema.py). "
        "Use algorithm='ahc' (the default).",
    ),
    (
        "align/cy/model.tar.gz",
        "The Welsh (cy) forced-alignment model could not be found on S3 or "
        "downloaded from CloudFront.",
        "StateDetectLanguageDownloadModels falls back to the English "
        "alignment model automatically for Latin-script languages — this is "
        "expected degraded behavior, not a failure, unless model_variant="
        "'turbo' was explicitly forced (see README.md gotcha).",
    ),
    (
        "OutOfMemory",
        "A Lambda ran out of memory.",
        "Check which Lambda variant (see README.md Lambda variants table) "
        "handled the failing state and whether the input (audio duration, "
        "number of speakers, embedding count) is unusually large for that "
        "variant's memory allocation.",
    ),
    (
        "Sandbox.Timedout",
        "The Lambda's execution environment was reclaimed mid-run, usually "
        "because the invocation itself hit the hard 15-minute ceiling.",
        "Same remediation as States.Timeout — look at whether the Map "
        "iteration's chunk size or model_variant is unusually expensive.",
    ),
]


def lookup_known_cause(error: str | None, cause: str | None) -> str | None:
    haystack = f"{error or ''} {cause or ''}"
    for needle, explanation, fix in _KNOWN_CAUSES:
        if needle.lower() in haystack.lower():
            return f"{explanation} Suggested fix: {fix}"
    return None


# =============================================================================
# CLOUDWATCH LOGS INSIGHTS (best-effort, generic)
# =============================================================================

def run_logs_insights_query(
    logs_client, log_group_name: str, start_epoch: float, end_epoch: float,
    query_string: str,
) -> list[dict] | None:
    """Run one Logs Insights query to completion. Returns parsed rows, or
    None on timeout/non-Complete status/missing permissions (caller decides
    how to report) — never raises for permission errors."""
    try:
        start_query = logs_client.start_query(
            logGroupName=log_group_name,
            startTime=int(start_epoch - _LOG_WINDOW_PAD_SEC),
            endTime=int(end_epoch + _LOG_WINDOW_PAD_SEC),
            queryString=query_string,
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        print(f"    [WARN] CloudWatch Logs query failed ({code}) — "
              f"skipping log correlation. Requires logs:StartQuery/"
              f"GetQueryResults on {log_group_name}")
        return None
    query_id = start_query["queryId"]

    for _ in range(20):
        time.sleep(1)
        result = logs_client.get_query_results(queryId=query_id)
        if result["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            break
    else:
        print("    [WARN] Logs Insights query timed out waiting for results")
        return None

    if result["status"] != "Complete":
        print(f"    [WARN] Logs Insights query status={result['status']}")
        return None

    return [{f["field"]: f["value"] for f in row} for row in result.get("results", [])]


def resolve_log_group_name(lambda_arn: str) -> str:
    """get_function_configuration ARNs aren't log group names directly;
    derive /aws/lambda/{function-name} from the function's ARN tail."""
    function_name = lambda_arn.rsplit(":", 1)[-1]
    return f"/aws/lambda/{function_name}"


# Two-stage query, reusable by any check that has a message pattern unique to
# one Lambda invocation among many concurrently-running Map iterations
# (MaxConcurrency up to 40): stage 1 finds the invocation's own @requestId
# from a distinguishing line; stage 2 pulls every relevant line from ONLY
# that invocation. Filtering by @requestId (not message text alone) is
# required whenever the interesting summary line has no per-item identifier
# of its own (e.g. a per-batch aggregate covering several groups).

def query_logs_by_request_id(
    logs_client,
    log_group_name: str,
    find_query: str,
    detail_query: str,
    start_epoch: float,
    end_epoch: float,
) -> list[dict] | None:
    find_rows = run_logs_insights_query(
        logs_client, log_group_name, start_epoch, end_epoch, find_query)
    if find_rows is None:
        return None
    request_id = next(
        (row.get("@requestId") for row in find_rows if row.get("@requestId")),
        None,
    )
    if not request_id:
        print("    [WARN] distinguishing log line not found in this window "
              "(or lacks @requestId) — skipping log correlation")
        return None
    return run_logs_insights_query(
        logs_client, log_group_name, start_epoch, end_epoch,
        detail_query % request_id,
    )


# =============================================================================
# FINDINGS / REGIONS OF INTEREST
# =============================================================================

@dataclass
class RegionOfInterest:
    """One time window worth a human's attention, tied back to the Map
    iteration that produced it (if any) so logs/clips can be correlated."""
    label: str
    start_sec: float
    end_sec: float
    dyn_chunk_idx: int | None = None
    map_state_name: str | None = None
    # Optional: a two-stage @requestId log query specific to this region's
    # check (see query_logs_by_request_id). find_query/detail_query use the
    # same %-placeholder convention as the module-level query constants.
    log_find_query: str | None = None
    log_detail_query_tmpl: str | None = None


@dataclass
class Finding:
    check: str
    severity: str  # "info" | "warn" | "error"
    message: str
    regions: list[RegionOfInterest] = field(default_factory=list)
    # Extra structured detail printed under the message but not otherwise
    # interpreted (e.g. the full k-sweep curve) — free-form text lines.
    detail_lines: list[str] = field(default_factory=list)


# =============================================================================
# DIAGNOSTIC CONTEXT — fetched/computed once, shared by every check
# =============================================================================

class DiagnosticContext:
    def __init__(
        self,
        sfn, s3, logs_client,
        execution_arn: str,
        desc: dict,
        hist: ExecutionAnalysis,
        output_bucket: str | None,
        output_prefix: str | None,
        final_data: dict,
    ):
        self.sfn = sfn
        self.s3 = s3
        self.logs_client = logs_client
        self.execution_arn = execution_arn
        self.desc = desc
        self.hist = hist
        self.output_bucket = output_bucket
        self.output_prefix = output_prefix
        self.final_data = final_data

        # S3 objects fetched by a check are cached here (also doubles as the
        # "what did we actually consult" manifest for --package).
        self._fetched_json: dict[str, dict | None] = {}
        self._fetched_keys_ordered: list[str] = []
        self.report_lines: list[str] = []

    def log(self, line: str = "") -> None:
        print(line)
        self.report_lines.append(line)

    def get_json_cached(self, key: str) -> dict | None:
        if key not in self._fetched_json:
            doc = get_json(self.s3, self.output_bucket, key)
            self._fetched_json[key] = doc
            self._fetched_keys_ordered.append(key)
        return self._fetched_json[key]

    def key_exists(self, key: str) -> bool:
        """Existence check with NO parsing — use this (not get_json_cached)
        for non-JSON outputs like final_transcript.vtt, or for any check
        that only cares whether a file was written, not its content."""
        try:
            self.s3.head_object(Bucket=self.output_bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    @property
    def workflow_params(self) -> dict | None:
        return self.get_json_cached(f"{self.output_prefix}/workflow_params.json")

    @property
    def final_transcript(self) -> dict | None:
        return self.get_json_cached(
            f"{self.output_prefix}/transcripts/final_transcript.json")

    @property
    def global_speakers(self) -> dict | None:
        gs_key = (self.final_data.get("global_speakers") or {}).get(
            "global_speakers_key")
        if not gs_key:
            gs_key = f"{self.output_prefix}/embeddings/global_speakers.json"
        return self.get_json_cached(gs_key)

    def transcript_chunk_keys(self) -> list[str]:
        return list_transcript_keys(self.s3, self.output_bucket, self.output_prefix)

    def transcript_chunk(self, key: str) -> dict | None:
        return self.get_json_cached(key)

    def resolved_language(self) -> str:
        ft = self.final_transcript
        if ft and ft.get("language"):
            return ft["language"]
        return self.final_data.get("language") or "unknown"

    def consulted_keys(self) -> list[str]:
        return list(self._fetched_keys_ordered)


# =============================================================================
# CHECK — zero-word groups
# =============================================================================

def _diagnose_zero_word_group(stat: dict, language: str) -> str:
    duration = stat.get("duration_sec", 0.0)
    num_turns = stat.get("num_turns", 0)

    notes = []
    if duration < 0.5:
        notes.append(
            "very short (<0.5s) — plausibly a VAD false positive "
            "(breath/noise) that Whisper correctly produced nothing for"
        )
    elif duration < 1.5:
        notes.append(
            "short utterance — could be a genuine but brief/quiet word "
            "Whisper failed to decode"
        )
    else:
        notes.append(
            "duration is not unusually short — worth listening to this "
            "window directly; may indicate word-assignment drift rather "
            "than a real Whisper skip"
        )

    if num_turns > 1:
        notes.append(f"group merges {num_turns} VAD turns — check for gaps within")

    if language.lower() in _CHAR_BASED_LANGUAGES:
        notes.append(
            f"language='{language}' is character-based — Whisper produces "
            "sub-word tokens for this script, so word-to-group soft-boundary "
            "assignment is less precise than for space-delimited languages "
            "(see lib/model_registry.py CHAR_BASED_LANGUAGES); this raises "
            "the odds of orphaned words for this language specifically"
        )

    return "; ".join(notes)


# Stage 1: find this group's own log line and read off its @requestId.
# Scoped to "group %d [" only — the batch-summary line has no group number,
# so it can't be found this way (that's the point of stage 2).
_ZWG_LOG_QUERY_FIND_REQUEST_ID = (
    "fields @timestamp, @requestId, @message"
    " | filter @message like /group %d \\[/"
    " | limit 5"
)

# Stage 2: pull every relevant line from that SAME Lambda invocation only.
# Filtering by @requestId (rather than message content alone) is required
# because the batch-summary line ("batch: X/Y group(s) had no words
# assigned  (orphaned_words=N)") is a per-BATCH aggregate covering every
# group Whisper transcribed in one call — it carries no group number, and
# up to MaxConcurrency=40 Map iterations run concurrently, each emitting its
# own such line. Filtering on message text alone would pull in every
# concurrently-running invocation's summary line, not just this group's own.
_ZWG_LOG_QUERY_BY_REQUEST_ID = (
    'fields @timestamp, @message'
    ' | filter @requestId = "%s" and'
    ' (@message like /no words assigned|orphaned word|'
    'batch_worker transcribe error|had no words assigned/)'
    " | sort @timestamp asc"
    " | limit 200"
)


def check_zero_word_groups(ctx: DiagnosticContext) -> list[Finding]:
    """Root-cause 'no words assigned to group' — the literal error set by
    lib/transcribe.py's batch_worker() when a speech group produced zero
    words. Reads every transcripts/transcript_{n}.json worker_stats."""
    findings: list[Finding] = []
    transcript_keys = ctx.transcript_chunk_keys()
    if not transcript_keys:
        return [Finding(
            "zero_word_groups", "error",
            "No transcripts/transcript_*.json files found under this "
            "prefix — nothing to analyze for this check.",
        )]

    zero_word_groups: list[dict] = []
    total_groups = 0
    for key in transcript_keys:
        doc = ctx.transcript_chunk(key)
        if doc is None:
            continue
        dyn_chunk_idx = doc.get("dyn_chunk_idx")
        worker_stats = doc.get("worker_stats", [])
        total_groups += len(worker_stats)
        for stat in worker_stats:
            if stat.get("error") == "no words assigned to group":
                zero_word_groups.append({**stat, "dyn_chunk_idx": dyn_chunk_idx})

    if not zero_word_groups:
        findings.append(Finding(
            "zero_word_groups", "info",
            f"No zero-word groups found ({total_groups} total groups "
            "checked). If the customer still reports missing words, the "
            "gap may instead be: (a) words assigned to the WRONG group "
            "(soft-boundary/orphan mis-assignment, not a zero-word error) "
            "— compare final_transcript.json sentence coverage against the "
            "audio directly, or (b) a sentence-splitting/merge issue "
            "downstream of word assignment.",
        ))
        return findings

    language = ctx.resolved_language()
    zero_word_groups.sort(key=lambda s: s.get("abs_start", 0.0))

    for stat in zero_word_groups:
        start = stat.get("abs_start", 0.0)
        end = stat.get("abs_end", 0.0)
        region = RegionOfInterest(
            label=f"zero-word group={stat.get('group_idx')}",
            start_sec=start, end_sec=end,
            dyn_chunk_idx=stat.get("dyn_chunk_idx"),
            map_state_name="Run whisper, align, sentence splitting models",
            log_find_query=_ZWG_LOG_QUERY_FIND_REQUEST_ID % stat.get("group_idx"),
            log_detail_query_tmpl=_ZWG_LOG_QUERY_BY_REQUEST_ID,
        )
        diagnosis = _diagnose_zero_word_group(stat, language)
        findings.append(Finding(
            "zero_word_groups", "warn",
            f"chunk={stat.get('dyn_chunk_idx')} group={stat.get('group_idx')} "
            f"[{fmt_ts(start)} → {fmt_ts(end)}] "
            f"duration={stat.get('duration_sec')}s "
            f"num_turns={stat.get('num_turns')} — {diagnosis}",
            regions=[region],
        ))

    findings.append(Finding(
        "zero_word_groups", "info",
        f"{len(zero_word_groups)} / {total_groups} total groups had zero "
        "words assigned. Jump to the listed timestamps in the source media "
        "to confirm whether each is silence/noise (expected) or actual "
        "missed speech (bug).",
    ))
    return findings


# =============================================================================
# CHECK — speaker clustering (count, assignment, over/under-clustering)
# =============================================================================

def _flipflop_runs(sentences: list[dict]) -> list[tuple[int, int]]:
    """Find runs of >= _FLIPFLOP_MIN_RUN consecutive sentences that are all
    short, closely-spaced, and alternate speaker_id — a signature of
    centroid-boundary jitter rather than genuine fast turn-taking. Returns
    [(start_sentence_id, end_sentence_id), ...]."""
    runs: list[tuple[int, int]] = []
    run_start_i = None
    i = 0
    n = len(sentences)
    while i < n - 1:
        a, b = sentences[i], sentences[i + 1]
        dur_a = float(a["end"]) - float(a["start"])
        gap = float(b["start"]) - float(a["end"])
        alternates = (a.get("speaker_id") and b.get("speaker_id")
                      and a["speaker_id"] != b["speaker_id"])
        is_flip = (dur_a <= _FLIPFLOP_MAX_SENTENCE_SEC
                   and gap <= _FLIPFLOP_MAX_GAP_SEC and alternates)
        if is_flip:
            if run_start_i is None:
                run_start_i = i
        else:
            if run_start_i is not None:
                run_len = i - run_start_i + 1
                if run_len >= _FLIPFLOP_MIN_RUN:
                    runs.append((sentences[run_start_i]["id"], sentences[i]["id"]))
            run_start_i = None
        i += 1
    if run_start_i is not None:
        run_len = (n - 1) - run_start_i + 1
        if run_len >= _FLIPFLOP_MIN_RUN:
            runs.append((sentences[run_start_i]["id"], sentences[n - 1]["id"]))
    return runs


def check_speaker_clustering(ctx: DiagnosticContext) -> list[Finding]:
    """Three sub-checks against embeddings/global_speakers.json — the
    pipeline's own clustering diagnostics, no new instrumentation needed:

      1. Wrong speaker COUNT   — single-speaker misfire, elbow/combined K
                                  disagreement, unreliable silhouette,
                                  heavy outlier removal.
      2. Wrong ASSIGNMENT      — acoustically diffuse clusters (low mean
                                  intra-cluster similarity), rapid same-
                                  window speaker flip-flops in the final
                                  transcript, heavy LOW/SKIPPED fallback.
      3. Over/under-clustering — the full per-K silhouette/CH curve
                                  (k_sweep_log) is flat or pinned at the
                                  max_k ceiling, meaning the algorithm
                                  couldn't confidently tell K apart.
    """
    gs = ctx.global_speakers
    if gs is None:
        return [Finding(
            "speaker_clustering", "error",
            "embeddings/global_speakers.json not found — did "
            "StateGlobalSpeakersClustering complete?",
        )]

    findings: list[Finding] = []
    diag = gs.get("clustering_diagnostics", {})
    n_spk = gs.get("num_global_speakers")
    n_total = gs.get("num_sentences_total") or 0
    n_high = gs.get("num_sentences_high") or 0
    n_low = gs.get("num_sentences_low") or 0
    n_skip = gs.get("num_sentences_skipped") or 0

    findings.append(Finding(
        "speaker_clustering", "info",
        f"Clustering summary: {n_spk} speaker(s) via "
        f"{gs.get('clustering_method')} — "
        f"{n_high} high / {n_low} low / {n_skip} skipped sentences "
        f"({n_total} total).",
    ))

    # ── 1. Wrong speaker COUNT ────────────────────────────────────────────
    reason = str(diag.get("selection_reason") or "")
    if diag.get("single_speaker_detected"):
        findings.append(Finding(
            "speaker_clustering", "info",
            "Single-speaker was detected (fast-path exit, no K enumeration "
            "ran). If the customer expects >1 speaker, this is the first "
            "thing to check — inspect dendrogram_final_height "
            f"({diag.get('dendrogram_final_height')}) and consider "
            "lowering clustering.single_speaker_threshold if speakers "
            "sound acoustically similar (e.g. same-gender narrators).",
        ))
    # The full k_sweep table is long (one row per K) and up to three
    # independent findings below may want to show it — attach it to only
    # the FIRST one that fires so it never prints twice in one report.
    sweep_table_shown = False

    def _sweep_detail(sweep_rows: list[dict]) -> list[str]:
        nonlocal sweep_table_shown
        if sweep_table_shown:
            return ["    (see k_sweep table above)"]
        sweep_table_shown = True
        return _format_sweep_log(sweep_rows)

    if "disagreement" in reason:
        findings.append(Finding(
            "speaker_clustering", "warn",
            "Elbow and combined K-selection methods DISAGREED on the "
            f"speaker count: {reason}. This is a genuine ambiguity in the "
            "data, not a bug — the k_sweep_log below shows every K "
            "considered; a human may judge a different K is correct.",
            detail_lines=_sweep_detail(diag.get("k_sweep_log") or []),
        ))
    if "unreliable" in reason:
        findings.append(Finding(
            "speaker_clustering", "warn",
            f"Best silhouette score was below the reliability threshold: "
            f"{reason}. Low confidence that ANY K is well-separated for "
            "this recording (e.g. speakers with similar voices, heavy "
            "cross-talk, or too little speech per speaker).",
        ))
    n_outliers = diag.get("n_outliers_removed") or 0
    if n_high and n_outliers / max(n_high, 1) > 0.10:
        findings.append(Finding(
            "speaker_clustering", "warn",
            f"{n_outliers}/{n_high} high-quality turns "
            f"({n_outliers / n_high:.0%}) were removed as outliers before "
            "clustering and reassigned afterward by nearest centroid — "
            "unusually high; the centroids may be built on a "
            "meaningfully-shrunken subset of the real acoustic variety.",
        ))

    # ── 3. Over/under-clustering (checked before #2 so the curve is
    #        available to reference from the assignment-quality findings) ──
    sweep = diag.get("k_sweep_log") or []
    clu_cfg = (ctx.workflow_params or {}).get("input", {}).get(
        "input_config", {}).get("clustering", {})
    max_k = clu_cfg.get("max_k")
    k_selected = diag.get("k_selected")
    if sweep and len(sweep) >= 3:
        sils = [c["sil"] for c in sweep if c.get("sil") is not None and c["sil"] > -1.0]
        if sils and (max(sils) - min(sils)) < 0.05:
            findings.append(Finding(
                "speaker_clustering", "warn",
                f"The silhouette curve across all K from 2 to {sweep[-1]['k']} "
                f"is nearly FLAT (range {min(sils):.4f}–{max(sils):.4f}) — "
                "the algorithm could not confidently distinguish between "
                "different speaker counts for this recording. Likely "
                "over- or under-clustering; a human listening pass is the "
                "most reliable way to find the true K here.",
                detail_lines=_sweep_detail(sweep),
            ))
        # k_selected is k_act (post-merge cluster count, always <= the raw
        # sweep k) so it can equal max_k only by coincidence — comparing it
        # to max_k directly misses the real "still climbing" signature,
        # which shows up as the LAST row of the sweep (the true enumeration
        # ceiling actually reached) scoring highest/near-highest. Detected
        # here instead: the last candidate's combined score is within 2%
        # of the curve's max — i.e. the ceiling row never lost to an
        # interior peak, so raising max_k might well have found one.
        combined_vals = [c["combined"] for c in sweep if c.get("combined") is not None]
        last_row = sweep[-1]
        if (combined_vals and last_row.get("combined") is not None
                and last_row["combined"] >= max(combined_vals) * 0.98):
            findings.append(Finding(
                "speaker_clustering", "warn",
                f"The K-sweep never found a peak — its LAST candidate "
                f"(k={last_row['k']}, k_act={last_row['k_act']}) scores "
                f"at or near the curve's best combined score "
                f"({last_row['combined']:.4f} vs max "
                f"{max(combined_vals):.4f}), meaning the enumeration "
                f"stopped at its ceiling (max_k={max_k}) still climbing "
                "rather than descending from a true peak. k_selected "
                f"({k_selected}) is likely an UNDER-count. Try raising "
                "clustering.max_k and re-running just "
                "StateGlobalSpeakersClustering with the same embeddings.",
                detail_lines=_sweep_detail(sweep),
            ))

    # ── 2. Wrong ASSIGNMENT ───────────────────────────────────────────────
    speaker_models = gs.get("speaker_models") or {}
    diffuse = [
        (spk, model.get("confidence"))
        for spk, model in speaker_models.items()
        if model.get("confidence") is not None
        and model["confidence"] < _DIFFUSE_CLUSTER_CONFIDENCE
    ]
    if diffuse:
        diffuse_str = ", ".join(f"{spk}={conf:.2f}" for spk, conf in diffuse)
        findings.append(Finding(
            "speaker_clustering", "warn",
            f"{len(diffuse)} cluster(s) are acoustically DIFFUSE "
            f"(mean intra-cluster cosine similarity < "
            f"{_DIFFUSE_CLUSTER_CONFIDENCE}): {diffuse_str}. A diffuse "
            "cluster likely lumps together turns from more than one real "
            "speaker — this is the most direct signal of wrong-speaker "
            "assignment (as opposed to wrong speaker COUNT).",
        ))

    if n_total and (n_low + n_skip) / n_total > _HIGH_FALLBACK_FRACTION:
        findings.append(Finding(
            "speaker_clustering", "warn",
            f"{n_low + n_skip}/{n_total} sentences "
            f"({(n_low + n_skip) / n_total:.0%}) were assigned via LOW-"
            "quality nearest-centroid or SKIPPED diarization-overlap "
            "fallback, rather than direct AHC clustering on HIGH-quality "
            "embeddings. A high fallback fraction means more of the "
            "transcript's speaker labels are running on weaker signal — "
            "often caused by many short utterances (see "
            "embed.min_embedding_duration_sec/soft_embedding_min_sec).",
        ))

    final_transcript = ctx.final_transcript
    if final_transcript:
        sentences = final_transcript.get("sentences", [])
        runs = _flipflop_runs(sentences)
        if runs:
            regions = [
                RegionOfInterest(
                    label=f"speaker flip-flop sentences {s_id}-{e_id}",
                    start_sec=float(sentences[s_id]["start"]),
                    end_sec=float(sentences[e_id]["end"]),
                )
                for s_id, e_id in runs[:10]  # cap regions to keep output readable
            ]
            findings.append(Finding(
                "speaker_clustering", "warn",
                f"{len(runs)} run(s) of rapid speaker alternation between "
                "short, closely-spaced sentences found in "
                "final_transcript.json — each alternation is a sentence "
                f"under {_FLIPFLOP_MAX_SENTENCE_SEC}s with a gap under "
                f"{_FLIPFLOP_MAX_GAP_SEC}s to the next, switching speaker_id "
                f"at least {_FLIPFLOP_MIN_RUN} times in a row. This pattern "
                "more often reflects centroid-boundary jitter than genuine "
                "fast turn-taking — listed windows are worth a direct "
                "listen.",
                regions=regions,
            ))

    return findings


def _format_sweep_log(sweep: list[dict]) -> list[str]:
    lines = []
    for c in sweep:
        combined = c.get("combined")
        combined_str = f"{combined:.4f}" if combined is not None else "n/a"
        lines.append(
            f"    k={c.get('k'):<3d} k_act={c.get('k_act'):<3d} "
            f"sil={c.get('sil'):.4f}  ch={c.get('ch'):.2f}  "
            f"combined={combined_str}"
        )
    return lines


# =============================================================================
# CHECK — language detection
# =============================================================================

def check_language_detection(ctx: DiagnosticContext) -> list[Finding]:
    final_data = ctx.final_data
    if not final_data:
        return []

    findings: list[Finding] = []
    probe_results = final_data.get("language_probe_results") or []
    findings.append(Finding(
        "language_detection", "info",
        f"language={final_data.get('language')} "
        f"confidence={final_data.get('language_confidence')} "
        f"method={final_data.get('language_method')} "
        f"num_probes={final_data.get('language_num_probes')}",
    ))
    if len(probe_results) > 1:
        distinct_langs = {p.get("lang") for p in probe_results}
        if len(distinct_langs) > 1:
            findings.append(Finding(
                "language_detection", "warn",
                f"Language probes DISAGREED across {distinct_langs} — "
                "majority vote may be unreliable for this content (common "
                "with code-switching, background music/noise on some "
                "probes, or very short probe windows). Consider setting "
                "input_config.language explicitly to skip detection.",
            ))
    return findings


# =============================================================================
# CHECK — missing outputs
# =============================================================================

# (relative key under output_prefix, human label, condition under which it's
# expected — None means "always expected on a SUCCEEDED execution")
_EXPECTED_OUTPUTS = [
    ("workflow_params.json", "workflow params snapshot", None),
    ("transcripts/final_transcript.json", "final transcript", None),
    ("transcripts/final_transcript.vtt", "final transcript VTT", None),
    ("embeddings/global_speakers.json", "global speaker clustering", None),
    ("usage_tracking.json", "usage/cost tracking", None),
]


def check_missing_outputs(ctx: DiagnosticContext) -> list[Finding]:
    if ctx.desc.get("status") != "SUCCEEDED":
        return [Finding(
            "missing_outputs", "info",
            f"Execution status is {ctx.desc.get('status')}, not SUCCEEDED — "
            "skipping the expected-outputs check (a non-terminal or failed "
            "execution is not expected to have every output).",
        )]

    findings: list[Finding] = []
    missing = []
    for rel_key, label, _cond in _EXPECTED_OUTPUTS:
        full_key = f"{ctx.output_prefix}/{rel_key}"
        if not ctx.key_exists(full_key):
            missing.append((full_key, label))

    if missing:
        for full_key, label in missing:
            findings.append(Finding(
                "missing_outputs", "error",
                f"Expected output missing: {label} (s3://{ctx.output_bucket}/"
                f"{full_key}) — execution reports SUCCEEDED but this file "
                "was never written. Check the corresponding state's Lambda "
                "logs directly; this usually indicates a silent write "
                "failure or a code path that returns success without "
                "writing.",
            ))
    else:
        findings.append(Finding(
            "missing_outputs", "info",
            "All expected output files are present.",
        ))
    return findings


# =============================================================================
# CHECK REGISTRY
# =============================================================================

CHECKS: dict[str, Callable[[DiagnosticContext], list[Finding]]] = {
    "zero_word_groups": check_zero_word_groups,
    "speaker_clustering": check_speaker_clustering,
    "language_detection": check_language_detection,
    "missing_outputs": check_missing_outputs,
}


# =============================================================================
# LOG / CLIP CORRELATION — generic, driven by RegionOfInterest
# =============================================================================

def correlate_region(
    ctx: DiagnosticContext,
    region: RegionOfInterest,
    skip_logs: bool,
    clip_audio: bool,
    clip_dir: str,
) -> None:
    ctx.log(
        f"    region: {region.label}  "
        f"[{fmt_ts(region.start_sec)} → {fmt_ts(region.end_sec)}]"
    )

    # ── Map iteration attempt history, if this region maps to one ─────────
    if region.dyn_chunk_idx is not None and region.map_state_name:
        attempts = ctx.hist.attempts_for(region.map_state_name, region.dyn_chunk_idx)
        if attempts:
            n_failed = sum(1 for a in attempts if a["status"] == "FAILED")
            ctx.log(f"      Map iteration attempts: {len(attempts)}  failed={n_failed}")
            for a in attempts:
                if a["status"] == "FAILED":
                    known = lookup_known_cause(a["error"], a["cause"])
                    ctx.log(f"        retry due to: {a['error']}  ({a['cause']})")
                    if known:
                        ctx.log(f"        [KNOWN CAUSE] {known}")

    # ── Audio clip (best-effort) ────────────────────────────────────────────
    if clip_audio:
        audio_bucket = ctx.final_data.get("audio_bucket")
        audio_key = ctx.final_data.get("audio_key")
        if audio_bucket and audio_key:
            clip_start = region.start_sec - _CLIP_PAD_SEC
            clip_end = region.end_sec + _CLIP_PAD_SEC
            safe_label = region.label.replace(" ", "_").replace("/", "-")
            clip_path = os.path.join(
                clip_dir, f"{safe_label}_{fmt_ts(region.start_sec).replace(':', '-')}.wav")
            try:
                wav_bytes = clip_wav_window(
                    ctx.s3, audio_bucket, audio_key, clip_start, clip_end)
                os.makedirs(clip_dir, exist_ok=True)
                with open(clip_path, "wb") as f:
                    f.write(wav_bytes)
                ctx.log(f"      Clip saved: {clip_path}  "
                        f"[{fmt_ts(max(0.0, clip_start))} → {fmt_ts(clip_end)}]  "
                        f"({len(wav_bytes):,} bytes)")
            except ClientError as exc:
                ctx.log(f"      [WARN] Failed to clip audio ({exc}) — skipping")
        else:
            ctx.log("      [WARN] data.audio_bucket/audio_key not found in "
                    "execution output — skipping clip")

    # ── CloudWatch Logs Insights (best-effort, only if the region supplied
    #     its own query pair — generic regions with no query are skipped) ──
    if skip_logs or not (region.log_find_query and region.log_detail_query_tmpl):
        return
    if region.dyn_chunk_idx is None or not region.map_state_name:
        return

    transcribe_lambda_arn = ctx.hist.state_lambda_arn.get(region.map_state_name)
    if not transcribe_lambda_arn:
        ctx.log("      [WARN] Could not resolve Lambda ARN for log correlation")
        return

    window = ctx.hist.iteration_window("Map transcribe processing", region.dyn_chunk_idx)
    if not window or "start" not in window or "end" not in window:
        ctx.log("      [WARN] Could not resolve iteration time window — "
                "skipping log correlation")
        return

    log_group_name = resolve_log_group_name(transcribe_lambda_arn)
    ctx.log(f"      Querying CloudWatch Logs Insights [{window['start']:.0f} "
            f"→ {window['end']:.0f}] ...")
    rows = query_logs_by_request_id(
        ctx.logs_client, log_group_name,
        region.log_find_query, region.log_detail_query_tmpl,
        window["start"], window["end"],
    )
    if rows:
        for row in rows:
            ts = row.get("@timestamp", "?")
            msg = row.get("@message", "").strip()
            ctx.log(f"        [{ts}] {msg}")
    elif rows is not None:
        ctx.log("        (no matching log lines found in this window)")


# =============================================================================
# PACKAGING
# =============================================================================

def write_package(
    ctx: DiagnosticContext,
    findings: list[Finding],
    package_path: str,
) -> None:
    """
    Bundle the report, raw execution history, and every S3 object actually
    consulted by a check into one tar.gz. Only pulls in files checks
    fetched — e.g. embeddings/*.json is included only if the
    speaker-clustering check ran and fired findings that read it.
    """
    manifest = {
        "execution_arn": ctx.execution_arn,
        "output_bucket": ctx.output_bucket,
        "output_prefix": ctx.output_prefix,
        "checks_run": sorted({f.check for f in findings}),
        "findings_summary": [
            {"check": f.check, "severity": f.severity, "message": f.message}
            for f in findings
        ],
        "s3_keys_included": ctx.consulted_keys(),
        "note": (
            "This package may contain transcript text and other content "
            "from the diagnosed media. Review before sharing outside your "
            "organization."
        ),
    }

    with tarfile.open(package_path, "w:gz") as tar:
        def _add_bytes(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))

        _add_bytes("manifest.json", json.dumps(manifest, indent=2, default=str).encode())
        _add_bytes("report.txt", "\n".join(ctx.report_lines).encode())

        try:
            history_bytes = json.dumps(ctx.hist.events, indent=2, default=str).encode()
            _add_bytes("execution_history.json", history_bytes)
        except TypeError:
            pass  # non-serializable event field (shouldn't happen); skip rather than fail packaging

        for key in ctx.consulted_keys():
            doc = ctx._fetched_json.get(key)
            if doc is None:
                continue
            # Keep the S3 key's own directory structure inside the archive,
            # relative to output_prefix, so it's obvious where each file
            # came from without needing the manifest open side-by-side.
            arcname = key
            if ctx.output_prefix and key.startswith(ctx.output_prefix + "/"):
                arcname = key[len(ctx.output_prefix) + 1:]
            _add_bytes(arcname, json.dumps(doc, indent=2, default=str).encode())

    size = os.path.getsize(package_path)
    ctx.log(f"\nPackage written: {package_path} ({size:,} bytes)")
    ctx.log(f"  {manifest['note']}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze(
    execution_arn: str,
    region: str | None,
    output_s3uri_override: str | None,
    skip_logs: bool,
    clip_audio: bool,
    clip_dir: str,
    checks_to_run: list[str],
    package_path: str | None,
) -> int:
    session_kwargs = {"region_name": region} if region else {}
    session = boto3.Session(**session_kwargs)
    sfn = session.client("stepfunctions")
    s3 = session.client("s3")
    logs_client = session.client("logs")

    print(f"Execution: {execution_arn}\n")

    # ── DescribeExecution ────────────────────────────────────────────────────
    desc = sfn.describe_execution(executionArn=execution_arn)
    status = desc["status"]
    start_date = desc["startDate"]
    stop_date = desc.get("stopDate")
    wall_sec = (stop_date - start_date).total_seconds() if stop_date else None

    print("Execution summary")
    print(f"  status     : {status}")
    print(f"  started    : {start_date}")
    print(f"  stopped    : {stop_date}")
    print(f"  wall_sec   : {wall_sec}")
    if status not in ("SUCCEEDED",):
        print(f"  error      : {desc.get('error')}")
        print(f"  cause      : {desc.get('cause')}")
        known = lookup_known_cause(desc.get("error"), desc.get("cause"))
        if known:
            print(f"  [KNOWN CAUSE] {known}")
    print()

    # ── Resolve output_bucket / output_prefix ───────────────────────────────
    output_bucket = output_prefix = None
    final_data: dict = {}

    if output_s3uri_override:
        output_bucket, output_prefix = parse_s3_uri(output_s3uri_override)
    elif status == "SUCCEEDED":
        try:
            payload = json.loads(desc.get("output") or "{}")
            final_data = payload.get("data", {})
            output_bucket = final_data.get("output_bucket")
            output_prefix = final_data.get("output_prefix")
        except json.JSONDecodeError:
            pass

    if not (output_bucket and output_prefix):
        print(
            "  [ERROR] Could not resolve output_bucket/output_prefix from the "
            "execution output. Pass --output-s3uri explicitly (e.g. if the "
            "execution is still RUNNING or failed before StateCheckAndPreprocess)."
        )
        # Continue anyway — execution history analysis below is still useful.

    # ── Execution history — per-state attempts / failures ───────────────────
    print("Fetching execution history...")
    events = get_execution_history(sfn, execution_arn)
    print(f"  {len(events)} events\n")

    hist = ExecutionAnalysis(events)

    print("Per-state attempt summary (retries indicate transient infra issues)")
    for state_name, attempts in hist.state_attempts.items():
        n_total = len(attempts)
        n_failed = sum(1 for a in attempts if a["status"] == "FAILED")
        n_succeeded = sum(1 for a in attempts if a["status"] == "SUCCEEDED")
        flag = "  <-- retries/failures present" if n_failed else ""
        print(f"  {state_name:<45s} attempts={n_total:<4d} "
              f"succeeded={n_succeeded:<4d} failed={n_failed:<4d}{flag}")
        if n_failed:
            for a in attempts:
                if a["status"] == "FAILED":
                    idx_note = f" map_index={a['map_index']}" if a["map_index"] is not None else ""
                    # ran_for distinguishes a timeout (near the Lambda's limit)
                    # from an immediate raise — same `error` string on both.
                    ran_note = (
                        f" ran_for={a['failed_ts'] - a['scheduled_ts']:.1f}s"
                        if a["failed_ts"] is not None else "")
                    print(f"      [{fmt_ts(a['scheduled_ts'] - epoch(start_date))}]"
                          f"{idx_note}{ran_note}  error={a['error']}  "
                          f"cause={a['cause']}")
                    known = lookup_known_cause(a["error"], a["cause"])
                    if known:
                        print(f"        [KNOWN CAUSE] {known}")
    print()

    ctx = DiagnosticContext(
        sfn, s3, logs_client, execution_arn, desc, hist,
        output_bucket, output_prefix, final_data,
    )

    if not (output_bucket and output_prefix):
        print("Skipping S3-based checks — no output location resolved.")
        return 1

    # ── Run checks ────────────────────────────────────────────────────────
    all_findings: list[Finding] = []
    for name in checks_to_run:
        check_fn = CHECKS.get(name)
        if not check_fn:
            print(f"  [WARN] Unknown check '{name}' — skipping")
            continue
        ctx.log("=" * 79)
        ctx.log(f"CHECK: {name}")
        ctx.log("=" * 79)
        try:
            findings = check_fn(ctx)
        except Exception as exc:  # noqa: BLE001 — one check's bug must not abort the rest
            ctx.log(f"  [ERROR] check '{name}' raised {type(exc).__name__}: {exc}")
            continue
        all_findings.extend(findings)

        for finding in findings:
            marker = {"info": "  ", "warn": "! ", "error": "X "}[finding.severity]
            ctx.log(f"{marker}[{finding.severity.upper()}] {finding.message}")
            for line in finding.detail_lines:
                ctx.log(line)
            for region_ in finding.regions:
                correlate_region(ctx, region_, skip_logs, clip_audio, clip_dir)
        ctx.log("")

    # ── Summary ───────────────────────────────────────────────────────────
    n_warn = sum(1 for f in all_findings if f.severity == "warn")
    n_error = sum(1 for f in all_findings if f.severity == "error")
    ctx.log("=" * 79)
    ctx.log(f"SUMMARY: {n_error} error-level, {n_warn} warn-level finding(s) "
            f"across {len(checks_to_run)} check(s)")
    ctx.log("=" * 79)

    if package_path:
        write_package(ctx, all_findings, package_path)

    return 1 if n_error else 0


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Generic speech-diarization workflow diagnostic, from a Step "
            "Functions execution ARN alone — combines execution history "
            "(retries, language detection, Map iteration timing), S3 "
            "outputs, and (best-effort) CloudWatch Logs Insights. Run in "
            "the account that owns the execution."
        ),
        epilog=f"Available checks: {', '.join(CHECKS)}",
    )
    parser.add_argument(
        "--execution-arn",
        required=True,
        help="Step Functions execution ARN for the speech-diarization state machine",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (default: from boto3 session / AWS_DEFAULT_REGION)",
    )
    parser.add_argument(
        "--output-s3uri",
        default=None,
        help=(
            "Override s3://bucket/prefix for output lookup. Normally derived "
            "automatically from the execution's output; only needed if the "
            "execution didn't complete or you want to point elsewhere."
        ),
    )
    parser.add_argument(
        "--checks",
        default=None,
        help=(
            f"Comma-separated subset of checks to run (default: all). "
            f"Available: {', '.join(CHECKS)}"
        ),
    )
    parser.add_argument(
        "--skip-logs",
        action="store_true",
        help="Skip the CloudWatch Logs Insights step (e.g. no logs:StartQuery permission)",
    )
    parser.add_argument(
        "--clip-audio",
        action="store_true",
        help=(
            "Download a local .wav clip (+/- %.0fs padding) of each finding's "
            "audio window, byte-range GET directly from "
            "data.audio_bucket/audio_key — no ffmpeg/librosa/numpy involved."
        ) % _CLIP_PAD_SEC,
    )
    parser.add_argument(
        "--clip-dir",
        default="./clips",
        help="Directory to save audio clips to when --clip-audio is set (default: ./clips)",
    )
    parser.add_argument(
        "--package",
        default=None,
        metavar="PATH",
        help=(
            "Write a tar.gz to PATH bundling the report, raw execution "
            "history, and every S3 output file actually consulted by the "
            "checks that ran — for handing off to someone who can't run "
            "this script themselves. May contain transcript text; review "
            "before sharing outside your organization."
        ),
    )
    args = parser.parse_args()

    checks_to_run = (
        [c.strip() for c in args.checks.split(",")] if args.checks
        else list(CHECKS)
    )

    try:
        return analyze(
            execution_arn=args.execution_arn,
            region=args.region,
            output_s3uri_override=args.output_s3uri,
            skip_logs=args.skip_logs,
            clip_audio=args.clip_audio,
            clip_dir=args.clip_dir,
            checks_to_run=checks_to_run,
            package_path=args.package,
        )
    except ClientError as exc:
        print(f"ERROR: AWS API call failed — {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
