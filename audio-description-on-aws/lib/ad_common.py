"""
Shared plumbing + EVERY tuned constant for audio-description — one home,
provenance on each (the runner's measured lessons travel with the number,
README.md Design record #6/#8). lib/ modules are PURE (parsed data in,
data out); the states do all I/O through the helpers here.

Run prefix layout (micro-workflow/audio-description/{workflow_id}/):
    _plan/plan.json          resolved pointers + prompts + config
    _plan/repair_state.json  repair-chain checkpoint (step cursor)
    character_map.json       understand: name map (+ naming policy)
    watch_notes.json         understand: synopsis/tone/beats/reveals/continuity
    windows.json             plan_windows: narration windows with duties
    orientation.json         plan_windows: per-boundary necessity scores
    cues.json                write_script..repair: window_id -> text
    placed.json              synth_fit..repair: fitted takes in the mix
    drops.json               write-through register of destroyed cues
    ad_gaps.json             oracle-reported unserved obligations
    redundancy.json          consecutive-pair overlap measurements
    cue_ledger.json          finalize: the reviewer's artifact
    takes/raw_NNN.wav        synthesized takes; takes/fit_NNN.wav fitted
    delivery/                srt/vtt/dapt/review.html (+ mix when asked)
"""
from __future__ import annotations

import json
import os

import boto3
from botocore.exceptions import ClientError

_s3 = boto3.client("s3")
_lambda = boto3.client("lambda")

PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
TARGET_NAME = os.environ.get("TARGET_NAME", "audio-description")
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
BEDROCK_GATEWAY_ARN = os.environ.get("BEDROCK_GATEWAY_ARN", "")
PROMPT_REGISTRY_ARN = os.environ.get("PROMPT_REGISTRY_ARN", "")
TTS_GATEWAY_ARN = os.environ.get("TTS_GATEWAY_ARN", "")

# env-overridable like every deployment-varying value here (a model
# deprecation or a region needing an inference-profile prefix must not
# require a code edit + TOOL_VERSION bump that re-identifies every run)
DEFAULT_MODEL_CHAIN = [m.strip() for m in os.environ.get(
    "DEFAULT_MODEL_CHAIN", "anthropic.claude-sonnet-4-6").split(",")
    if m.strip()]

# The FIVE registry prompt categories (README.md Design record #8;
# frame-inspection's credits-read is consumed, not owned, here).
# ad-repair is ONE category; the repair step's tag parameterizes the ask.
PROMPT_CATEGORIES = ("ad-name-map", "ad-watch-notes", "ad-writer",
                     "ad-critic", "ad-repair")

# ── window geometry (runner constants, measured provenance verbatim) ────────
WORD_MERGE = 0.35
PAD = 0.30
MIN_USABLE_SCENE = 0.7
MIN_USABLE_STD = 1.5
LONG_GAP = 20.0
MIN_SHOT_WIN = 2.5
SCENE_SEARCH_AFTER = 10.0
SCENE_SEARCH_AFTER_HIGH = 20.0   # high-necessity breaks get a longer horizon
MIN_GROUND_WORDS = 7             # place + subject + verb; below this an entry
                                 # cue can only stub ("Car.") — measured on
                                 # Meridian: a 0.72s/2-word window held the
                                 # office->moving-car entry while a 6-word
                                 # window sat 6s later, unused
MONTAGE_SCENE_MAX = 6.0          # a scene too short to hold its own
                                 # description; a RUN of them is ONE
                                 # orientation event (drama_b's cold open
                                 # cuts 20+ scenes in 100s)
MIN_PLACE_WORDS = 4              # bare place, no who/what
LATE_ENTRY_MAX = 45.0            # past this an "entry" no longer orients
ANTICIPATE_MAX = 6.0             # a pre-cut window may orient this far ahead
REVEAL_MIN_WORDS = 5             # apparatus + subject — lower than an
                                 # entry's because a reveal needs no verb
REVEAL_FOLLOW_MAX = 20.0         # how far a follow-through may sit from the
                                 # reveal; the PRIMARY never migrates
EARLY_NAMING = True              # Netflix style (user decision 2026-08-28):
                                 # trades PARITY for CLARITY; never applies
                                 # to a CONCEALED identity
NECESSITY_HIGH = 0.55            # judgment default, NOT corpus-fitted; the
                                 # score ships per boundary so a consumer
                                 # can re-threshold on its own material
MUSIC_BUDGET_SCALE = 1.0         # sparseness via instruction, never by
                                 # crushing budget (measured: budget-crush
                                 # dropped a van-collision to "A sunny
                                 # intersection.")

# Sung-vocal window policy (user decision 2026-08-31): CONFIGURABLE per
# request. "narrate" (default): windows open over sung vocals — the
# transcribed-lyrics case is already closed for free by the WORD-level gap
# map, and a hard veto starves windows on music-heavy programmes.
# "avoid": spans where the speech stem is active but carries NO transcript
# words (untranscribed singing/walla) are treated as speech and closed.
SUNG_VOCAL_POLICIES = ("narrate", "avoid")
SUNG_VOCAL_DEFAULT = "narrate"

# ── fit + synthesis (runner constants) ──────────────────────────────────────
TEMPO_CAP = 1.28
GRACE = 0.2
SPACING = 0.3
ONSET_PAD = 0.08                 # margin before the true next-word onset;
                                 # ONE home (was copy-pasted 5x, review
                                 # 2026-08-31)
SR = 48000
REASK_ROUNDS = 2                 # bounded, counted (runner rounds 1..2)
TTS_SHARD_ATTEMPTS = 2           # total tries per fan-out shard: a FAILED
                                 # tts run is resubmitted under a fresh
                                 # per-attempt idempotency token (the
                                 # original absorbs into the sticky FAILED
                                 # execution forever — review 2026-09-02);
                                 # bounded so a systemic tts outage raises
                                 # by name instead of looping
# The sibling-run pointer vocabulary, ONE home (review 2026-09-08: the
# gateway and validate_plan each hand-maintained a copy — the flat zip
# carries both handlers plus lib/, so both import from here).
REQUIRED_INPUT_URIS = ("shot_segmentation_s3uri", "shot_metadata_s3uri",
                       "speech_diarization_s3uri",
                       "conversation_analysis_s3uri")
# face-recognition OR cast-resolution (2026-09-08): the cast run now
# exports person-level appearance timelines, so it can be the sole
# people source — one identity input instead of two. The rule lives
# HERE because the gateway refuses at submit and validate refuses
# in-run; two spellings of it would drift.
EITHER_INPUT_URIS = ("face_recognition_s3uri", "cast_resolution_s3uri")
OPTIONAL_INPUT_URIS = ("audiostem_s3uri", "decisions_s3uri",
                       "cast_resolution_s3uri", "face_recognition_s3uri")


def people_source_error(present) -> str | None:
    """The one input rule beyond simple presence: at least one people
    source. `present` is the set of supplied pointer names."""
    if not any(k in present for k in EITHER_INPUT_URIS):
        return ("a people source is required: face_recognition_s3uri "
                "or cast_resolution_s3uri (the cast run must export "
                "person appearances — runs from 2026-09-08 on)")
    return None

# cast-resolution bindings entering the name-map record. The structural
# part is the NAMED-only filter (a 2h news asset ships 600+ persons,
# most unnamed silent faces the name map has no use for); the cap is a
# prompt-budget knob over what survives that filter — 40 named people
# covers every corpus asset's principal cast with room, and one line
# each keeps the record's adjudicated section smaller than the raw
# hints it anchors.
CAST_BINDINGS_CAP = 40

TTS_SHARD_SIZE = 90              # tts-on-aws refuses >100 transcripts per
                                 # batch (its measured cap); the runner
                                 # sharded at <=90 and the port must too —
                                 # shards fan out in PARALLEL through the
                                 # loop's Map state, one task token per
                                 # shard (2026-09-02; tts is Lambda-driven
                                 # with no service throttle to respect)

# The seconds->words rates, all three, side by side (review 2026-08-31:
# they were scattered as bare 2.3/2.2/2.0 across eight sites). WPS is the
# writer's budget rate; the repair cap sits just under it so a repair
# never promises words the fitter rejects; the reask cap is harder still
# because the take already measured over.
WPS = 2.3
WPS_REPAIR_CAP = 2.2
WPS_REASK = 2.0

# ── oracle constants (runner, measured provenance in the modules) ───────────
BEAT_WINDOW = (-10.0, 15.0)      # a cue this far from a beat still conveys it
ORIENT_WINDOW = 12.0             # later than this does not orient its cut
REDUNDANT_GAP_MAX = 15.0
REDUNDANT_CONTAIN = 0.6
REDUNDANT_ECHO_WORDS = 8
TAIL_BRIDGE_MAX = 5.0
TEMPO_MINOR = 1.35
TEMPO_MAJOR = 1.60

# Map sizing. Env-tunable per deployment, NEVER per request (the
# workflow_id-preimage rule).
WINDOWS_PER_CHUNK = int(os.environ.get("WINDOWS_PER_CHUNK", "18"))


def run_prefix(workflow_id: str) -> str:
    return f"micro-workflow/{TARGET_NAME}/{workflow_id}"


def read_json(bucket: str, key: str):
    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body.decode("utf-8"))


def read_json_if_exists(bucket: str, key: str):
    """None ONLY on a genuine missing key; everything else re-raises."""
    try:
        return read_json(bucket, key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def read_jsonl(bucket: str, key: str) -> list:
    body = _s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return [json.loads(line) for line in body.decode("utf-8").splitlines()
            if line.strip()]


def write_json(bucket: str, key: str, payload) -> None:
    _s3.put_object(Bucket=bucket, Key=key,
                   Body=json.dumps(payload, ensure_ascii=False,
                                   indent=1).encode("utf-8"),
                   ContentType="application/json")


def read_bytes(bucket: str, key: str) -> bytes:
    return _s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def write_bytes(bucket: str, key: str, body: bytes,
                content_type: str = "application/octet-stream") -> None:
    _s3.put_object(Bucket=bucket, Key=key, Body=body,
                   ContentType=content_type)


def list_keys(bucket: str, prefix: str) -> list:
    """Every key under a prefix, sorted (S3 lists lexicographically;
    sorting is restated so callers may rely on it)."""
    keys = []
    for page in _s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return sorted(keys)


def exists(bucket: str, key: str) -> bool:
    """HEAD-cheap existence probe (refuse-before-spend needs to name a
    missing artifact without downloading it)."""
    try:
        _s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "403"):
            return False
        raise


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// uri: {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed s3 uri: {uri!r}")
    return bucket, key


def invoke_gateway_tool(function_arn: str, tool_name: str, event: dict):
    """Invoke a sibling's gateway-tool Lambda exactly as AgentCore would —
    the submit brain stays the sibling's. An `{error: true}` response is
    returned, not raised."""
    import base64
    ctx = base64.b64encode(json.dumps(
        {"custom": {"bedrockAgentCoreToolName": tool_name}}
    ).encode("utf-8")).decode("ascii")
    resp = _lambda.invoke(FunctionName=function_arn, ClientContext=ctx,
                          Payload=json.dumps(event).encode("utf-8"))
    payload = json.loads(resp["Payload"].read().decode("utf-8"))
    if resp.get("FunctionError"):
        raise RuntimeError(f"{tool_name} raised: {json.dumps(payload)[:400]}")
    return payload
