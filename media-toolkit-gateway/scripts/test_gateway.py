#!/usr/bin/env python3
"""
Test the AgentCore Gateway for every micro-workflow target without any
agent implementation.  Talks directly to the MCP HTTP endpoint.

What it tests
─────────────
1. Fetch credentials bundle from Secrets Manager.
2. Obtain a Cognito M2M token (client_credentials flow).
3. Call tools/list  → verify auth works and every deployed tool is advertised
   (40 with every optional target enabled: four targets x 3 tools +
   face-recognition's 11 + hybrid-search's 6 + hybrid-search-admin's 3 +
   text-to-speech's 6 + media-outputs' 2).
4. (Optional) Call a start tool on one or more targets, then poll get_job_status
   until the job reaches a terminal state or the poll timeout expires. On
   SUCCEEDED, the outputs[] contract is checked, the first few file entries are
   batch-presigned via media-outputs___get_download_urls (alongside a deliberately
   invalid uri, to assert per-item failure is isolated), the singular
   media-outputs___get_download_url is checked for its flat {url} shape, and one
   URL is fetched over HTTP.
5. (Optional, --idempotency) Re-submit each eligible target with the SAME
   arguments and assert the resubmit contract: same workflow_id + job_id, a
   `message` saying the call was absorbed, no result fields, no second
   execution — then poll for the results the submit deliberately withheld.  One
   target is excluded on purpose rather than by omission — see "Idempotency"
   below.

Two targets are exercised in a reduced form, because their response shapes do
not fit the shared submit/poll/outputs[] contract the table-driven poll loop
assumes (this is deliberate, see media-toolkit-gateway/README.md):

  text-to-speech  synthesize_speech returns per-transcript outputs[] entries
                  keyed by transcript_index rather than {uri, mime, label}
                  triples, and takes no asset_s3uri, so it gets a dedicated
                  result check (batch_qc_summary rollup, per-clip uri).  It DOES
                  carry the shared workflow_id/output_s3_prefix as of
                  2026-08-07; only the outputs[] shape differs.
  hybrid-search   index_content returns prepared/skipped/indexed counts and
                  never an outputs[] list, so it gets a counts check instead.

Idempotency (--idempotency)
───────────────────────────
--idempotency-token is the caller's assertion "this is the same request I sent
before".  It is folded into workflow_id = uuid5(canonical SFN input +
TOOL_VERSION + token) and then DISCARDED — never forwarded to the state
machine, never stored, and never echoed back, so there is no round trip to
assert.  The service answers with its own name for the run, `workflow_id`,
which is what this harness compares across a resubmit.

The token NARROWS dedup rather than replacing it: the same token with different
parameters still starts a fresh run, so a reused token can never hand back
another asset's results.  The reverse is what --idempotency exercises — a
resubmit hits the existing run only if EVERY argument matches, which is why
build_args() is a pure function of the parsed CLI namespace.

Every start tool but hybrid-search's accepts a token; omitting it means
different things.  Four targets dedup on the parameter hash with or without one,
because their parameters name the asset being processed and therefore fully
determine the output.  face-recognition and text-to-speech are different —
neither is tied to one asset (a persistent face collection and a mutable voice
catalog respectively), so identical parameters can legitimately need to run
again, and an ABSENT token means "do not deduplicate me".  For those two,
--idempotency therefore requires --idempotency-token; main() refuses without it
rather than letting a correct fresh run read as a contract failure.  See
TOKEN_ACCEPTING_TARGETS and TOKEN_REQUIRED_FOR_IDEMPOTENCY.

Sending a token to a tool that does NOT honour one — a poll, a read, a search —
is not an error either: as of 2026-08-06 every gateway accepts the field and
logs a warning, on the grounds that a hallucinated parameter must not fail an
otherwise-correct request over a field that could not have changed the answer.
This harness still only sends it where it is honoured.

There is ONE resubmit behavior, asserted for every non-excluded target:

  RETURNS_EXISTING  audio-stem, speech-diarization, media-processing,
                    shot-segmentation, face-recognition, text-to-speech
                    ExecutionAlreadyExists is caught, NO second execution is
                    started, and the reply is the same HANDLE a fresh start
                    returns — same workflow_id + job_id, plus a `message`
                    saying the call was absorbed, plus the existing run's
                    `status` (and `error` when that status is a failure).
                    Results are NOT in it; get_job_status is where they live.

A submit is a handle on every path, which is what makes the shape checkable:
the reachable field set equals the declared one, so this harness asserts the
whole allowed set (SUBMIT_HANDLE_FIELDS) rather than probing for one known leak.
Until 2026-08-07 an absorbed resubmit of a finished run handed back its full
results, so a submit's shape depended on when it was called and no such
assertion was possible.

`message` is the only field that distinguishes an absorbed call from a fresh
one, and comparing job_ids cannot substitute for it: the execution name is
derived from the same input hash either way, so a fresh start of a vanished run
returns a byte-identical ARN.

text-to-speech used to be a second contract of its own, answering
{"status": "ALREADY_RUNNING"} with no results — a status word that described
the NAME COLLISION rather than the run (an original that succeeded hours ago
still answered ALREADY_RUNNING).  Fixed 2026-08-05: it now describes the
existing execution like every sibling, so the second contract is gone rather
than encoded.

A terminal failure is STICKY.  Step Functions reserves the execution name for
~90 days, so an identical resubmit keeps returning that same verdict for that
whole window; only a changed parameter, a different --idempotency-token, or a
new tool version re-runs it.  The harness reports it loudly, with the `error`
cause, since that cause is what decides between fixing a parameter and
supplying a new token.

One target is excluded by design — see IDEMPOTENCY_EXCLUDED: hybrid-search
indexes per-document rather than per-execution, and collapsing two accumulating
writes on an input hash would be wrong, not merely unnecessary.

Usage
─────
    # List tools only (no jobs started)
    python test_gateway.py --secret-name /media-toolkit-gateway/dev/gateway-credentials

    # Submit a media-processing job and poll to completion
    python test_gateway.py \
        --secret-name /media-toolkit-gateway/dev/gateway-credentials \
        --run media-processing \
        --asset-s3-uri s3://<your-ingest-bucket>/test/sample.mp4

    # Submit every target under one idempotency token and poll
    python test_gateway.py \
        --secret-name /media-toolkit-gateway/dev/gateway-credentials \
        --run all --idempotency-token my-run-2026-08-06 \
        --asset-s3-uri s3://<your-ingest-bucket>/test/sample.mp4

    # Re-run with the same token and assert each target's resubmit contract
    python test_gateway.py \
        --secret-name /media-toolkit-gateway/dev/gateway-credentials \
        --run all --idempotency-token my-run-2026-08-06 --idempotency \
        --asset-s3-uri s3://<your-ingest-bucket>/test/sample.mp4

    # Synthesize ONE speaker's whole side of a diarized conversation, chaining
    # speech-diarization's output into tts over the same asset
    python test_gateway.py \
        --secret-name /media-toolkit-gateway/dev/gateway-credentials \
        --run text-to-speech --tts-voice-id <voice_id> \
        --tts-transcript-s3-uri s3://<proxy>/micro-workflow/speech-diarization/<wid>/transcripts/final_transcript.json

    # ...or just the first 10 of that speaker's cues (cheap smoke test)
    python test_gateway.py \
        --secret-name /media-toolkit-gateway/dev/gateway-credentials \
        --run text-to-speech --tts-voice-id <voice_id> --tts-max-cues 10 \
        --tts-transcript-s3-uri s3://<proxy>/micro-workflow/speech-diarization/<wid>/transcripts/final_transcript.json

Environment
───────────
    AWS_REGION (default: eu-west-1)
"""

import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import boto3

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_REGION = "eu-west-1"
DEFAULT_POLL_INTERVAL_SEC = 15
DEFAULT_POLL_TIMEOUT_SEC = 3600  # 1 hour

# Tool names as the gateway exposes them: {target}___{tool}
TOOLS = {
    "audio-stem": [
        "audio-stem___separate_audio_stems",
        "audio-stem___get_job_status",
        "audio-stem___describe_tool",
    ],
    "speech-enhancement": [
        "speech-enhancement___enhance_speech",
        "speech-enhancement___get_job_status",
        "speech-enhancement___describe_tool",
    ],
    "speech-diarization": [
        "speech-diarization___diarize_speech",
        "speech-diarization___get_job_status",
        "speech-diarization___describe_tool",
    ],
    "media-processing": [
        "media-processing___process_media",
        "media-processing___quick_edit",
        "media-processing___capture_frames",
        "media-processing___get_job_status",
        "media-processing___describe_tool",
    ],
    "shot-segmentation": [
        "shot-segmentation___segment_shots",
        "shot-segmentation___get_job_status",
        "shot-segmentation___describe_tool",
    ],
    "face-recognition": [
        "face-recognition___recognize_faces",
        "face-recognition___get_job_status",
        "face-recognition___describe_tool",
        "face-recognition___create_collection",
        "face-recognition___list_collections",
        "face-recognition___list_faces",
        "face-recognition___get_face",
        "face-recognition___batch_update_faces",
        "face-recognition___batch_index_faces",
        "face-recognition___batch_delete_faces",
        "face-recognition___delete_collection",
    ],
    "hybrid-search": [
        "hybrid-search___index_content",
        "hybrid-search___search_content",
        "hybrid-search___create_collection",
        "hybrid-search___list_collections",
        "hybrid-search___get_job_status",
        "hybrid-search___describe_tool",
    ],
    "hybrid-search-admin": [
        "hybrid-search-admin___configure_search_backend",
        "hybrid-search-admin___delete_collection",
        "hybrid-search-admin___describe_tool",
    ],
    "text-to-speech": [
        "text-to-speech___synthesize_speech",
        "text-to-speech___register_voice",
        "text-to-speech___get_job_status",
        "text-to-speech___list_voice_collection",
        "text-to-speech___get_voice",
        "text-to-speech___describe_tool",
    ],
    "bedrock-inference": [
        "bedrock-inference___run_inference",
        "bedrock-inference___get_job_status",
        "bedrock-inference___list_models",
        "bedrock-inference___get_model",
        "bedrock-inference___describe_tool",
    ],
    "prompt-registry": [
        "prompt-registry___put_prompt",
        "prompt-registry___get_prompt",
        "prompt-registry___list_prompts",
        "prompt-registry___promote_prompt",
        "prompt-registry___delete_prompt",
        "prompt-registry___deprecate_prompt",
        "prompt-registry___describe_tool",
    ],
    # DEFINED identical to prompt-registry modulo the prefix (the bundle
    # pins the two targets' schemas byte-equal; its coverage test then
    # pins this list against the template — a literal because that test
    # reads TOOLS via ast.literal_eval, parse-never-import)
    "prompt-registry-admin": [
        "prompt-registry-admin___put_prompt",
        "prompt-registry-admin___get_prompt",
        "prompt-registry-admin___list_prompts",
        "prompt-registry-admin___promote_prompt",
        "prompt-registry-admin___delete_prompt",
        "prompt-registry-admin___deprecate_prompt",
        "prompt-registry-admin___describe_tool",
    ],

    "conversation-analysis": [
        "conversation-analysis___analyze_conversation",
        "conversation-analysis___get_job_status",
        "conversation-analysis___describe_tool",
    ],
    "shot-metadata-enrichment": [
        "shot-metadata-enrichment___enrich_shot_metadata",
        "shot-metadata-enrichment___get_job_status",
        "shot-metadata-enrichment___describe_tool",
    ],
    "program-structure-analysis": [
        "program-structure-analysis___analyze_program_structure",
        "program-structure-analysis___get_job_status",
        "program-structure-analysis___describe_tool",
    ],
    "audio-description": [
        "audio-description___generate_ad",
        "audio-description___get_job_status",
        "audio-description___describe_tool",
    ],
    "artwork-selection": [
        "artwork-selection___select_artwork",
        "artwork-selection___get_job_status",
        "artwork-selection___describe_tool",
    ],
    "cast-resolution": [
        "cast-resolution___resolve_cast",
        "cast-resolution___get_job_status",
        "cast-resolution___describe_tool",
    ],
    "media-outputs": [
        "media-outputs___get_download_url",
        "media-outputs___get_download_urls",
    ],
    "media-ingest": [
        "media-ingest___list_files",
    ],
    "media-uploads": [
        "media-uploads___start_upload",
        "media-uploads___get_upload_part_urls",
        "media-uploads___complete_upload",
        "media-uploads___abort_upload",
    ],
}

# Every GatewayTarget in cfnstack.yaml that carries a Condition, i.e. one an
# empty ARN parameter legitimately omits. Membership is what distinguishes
# "not deployed" (a deployment choice) from "deployed but incomplete" (always
# a failure), so it must track the template's Conditions exactly in BOTH
# directions: a conditional target missing here fails a valid partial deploy,
# and an unconditional one listed here lets a total outage pass as a skip.
# Pinned against the template by micro-workflow-bundle's
# test_harness_tools_map_matches_the_gateway_template.
OPTIONAL_TARGETS = {
    "shot-segmentation", "face-recognition", "hybrid-search",
    "hybrid-search-admin", "text-to-speech", "bedrock-inference",
    "prompt-registry", "prompt-registry-admin", "conversation-analysis",
    "shot-metadata-enrichment",
    "program-structure-analysis", "audio-description", "cast-resolution",
    "artwork-selection", "speech-enhancement", "media-ingest",
    "media-uploads",
}

START_TOOL = {
    "audio-stem":         "audio-stem___separate_audio_stems",
    "speech-diarization": "speech-diarization___diarize_speech",
    "media-processing":   "media-processing___process_media",
    "shot-segmentation":  "shot-segmentation___segment_shots",
    "face-recognition":   "face-recognition___recognize_faces",
    "text-to-speech":     "text-to-speech___synthesize_speech",
    "hybrid-search":      "hybrid-search___index_content",
    "audio-description":  "audio-description___generate_ad",
    "cast-resolution":     "cast-resolution___resolve_cast",
    "artwork-selection":   "artwork-selection___select_artwork",
}

# DERIVED from START_TOOL, not hand-kept beside it (review, 2026-09-06):
# every workflow target's poll tool is literally {target}___get_job_status
# (each name is cross-checked against TOOLS above), and the two
# audio-description/cast-resolution omissions the derivation replaces were
# exactly the drift a parallel hand-written table invites — a target that
# could start but never be polled, discovered only on a live --run.
STATUS_TOOL = {t: f"{t}___get_job_status" for t in START_TOOL}

# Targets that accept --run (media-outputs has no start/status tools)
WORKFLOW_TARGETS = list(START_TOOL)

# Targets whose SUCCEEDED result carries the shared workflow_id + outputs[]
# contract that verify_outputs() checks.  The two exceptions are deliberate
# (see the module docstring) and get their own result checks.
SHARED_CONTRACT_TARGETS = {
    "audio-stem", "speech-diarization", "media-processing",
    "shot-segmentation", "face-recognition",
}

# Targets whose start tool declares `idempotency_token` — every one that starts
# an execution named after an input hash.  `hybrid-search` is the only start
# target absent: index_content names its execution uuid4() on purpose (see
# IDEMPOTENCY_EXCLUDED), so a token there would have nothing to narrow.
#
# There is no echo to assert in either direction — the token is consumed into
# the hash and discarded, so `workflow_id` is the only identity a response
# carries.
TOKEN_ACCEPTING_TARGETS = {
    "audio-stem", "speech-diarization", "media-processing",
    "shot-segmentation", "face-recognition", "text-to-speech",
}

# The two targets where an ABSENT token means "do not deduplicate me" — the
# gateway generates one, so every call gets its own run.  Neither is tied to a
# single asset: face-recognition reads and writes a persistent, mutable face
# collection, and text-to-speech synthesizes against a mutable voice catalog, so
# identical parameters can legitimately need to run again.  The other four dedup
# on the parameter hash with or without a token, because their parameters name
# the asset and therefore fully determine the output.
#
# Consequence for this harness: --idempotency MUST pass a token to these two, or
# the resubmit correctly starts a fresh run and the assertion fails on right
# behavior.  A subset of TOKEN_ACCEPTING_TARGETS, kept separate because the two
# sets answer different questions ("is the field accepted?" vs "does omitting it
# change the outcome?") and only the second one gates the CLI.
TOKEN_REQUIRED_FOR_IDEMPOTENCY = {"face-recognition", "text-to-speech"}

# Resubmit behavior per target — asserted by --idempotency.  Derived from how
# each gateway tool names its SFN execution; see the module docstring.  One
# value on purpose: every uuid5-named target describes the existing run.
# Keep it a dict rather than a set so a future target with a genuinely
# different contract has somewhere to say so.
RETURNS_EXISTING = "returns-existing"    # uuid5 name, existing run described

IDEMPOTENCY_CONTRACT = {
    "audio-stem":         RETURNS_EXISTING,
    "speech-diarization": RETURNS_EXISTING,
    "media-processing":   RETURNS_EXISTING,
    "shot-segmentation":  RETURNS_EXISTING,
    "face-recognition":   RETURNS_EXISTING,
    "text-to-speech":     RETURNS_EXISTING,
}

# Deliberately NOT asserted by --idempotency, and not because they were missed.
#
#   hybrid-search     index_content names its execution uuid4(), so the
#                     execution layer is never idempotent by design — a
#                     resubmit always starts a new run.  Its real guarantee is
#                     per-DOCUMENT: doc_id = md5(collection:asset:doc_type:
#                     start:end:chunk) filters duplicates out before any
#                     Bedrock spend.  Measured on a 77-document resubmit:
#                     prepared=0, skipped=77, indexed=0.  Asserting that here
#                     would mean polling a whole second indexing run to
#                     re-verify a property the indexer's own test suite already
#                     covers.
# Excluding it on principle, NOT as an oversight: execution-level idempotency
# would be actively WRONG for an accumulating write to a shared collection —
# two calls with different manifests must both run, and collapsing them on an
# input hash would silently drop the second.
#
# face-recognition was excluded here until 2026-08-05, when its uuid echo was
# broken on the poll path.  The echo is gone entirely as of 2026-08-06, which
# removed that failure mode rather than fixing it, so face-recognition is
# asserted like the other four — and it benefits MOST from the assertion,
# because a duplicate run burns a semaphore slot, billed Rekognition calls, and
# can auto-index faces into a persistent collection.  --run passes
# max_faces_to_index=0 so a failed assertion cannot mutate a live collection.
#
# hybrid-search is still fully exercised by --run; only the resubmit assertion
# is skipped.
IDEMPOTENCY_EXCLUDED = {"hybrid-search"}

# The complete set of fields a submit may return — a HANDLE, not results.  This
# is asserted as an upper bound (nothing outside it), never as an equality: the
# absorbed-only fields appear on a resubmit and `current_state` disappears once
# the run has stopped, so no single call produces the whole set.
#
# Enumerated here rather than probed for one known leak, because the regression
# it guards is "a result field came back", and results are per-target and
# open-ended: an allow-list fails on any of them, a deny-list only on the ones
# somebody thought of.  Mirrors the submit OutputSchemas in cfnstack.yaml.
SUBMIT_HANDLE_FIELDS = {
    "job_id", "status", "workflow_id", "output_s3_prefix", "asset_s3uri",
    "current_state",            # RUNNING only — a stopped execution is in none
    "message", "error",         # absorbed-resubmit only
    "error_code",               # absorbed terminal failure only (2026-10-15)
}

# Two targets have a genuinely narrower handle, and declaring the wider set for
# them would let a real leak through.
#
#   text-to-speech  submits are not asset-shaped (a synthesis names transcripts
#                   and a voice; a registration names a reference WAV) and
#                   neither path resolves a current state.
#   hybrid-search   index_content is uuid4-named, so there is no absorbed case
#                   and no `message`/`error`; it names the collection and how
#                   much was submitted instead of an asset.  Its
#                   num_inline_documents is what was SUBMITTED — prepared /
#                   skipped, what was embedded vs deduped, come from the poll.
SUBMIT_HANDLE_FIELDS_PER_TARGET = {
    # output_s3_prefix joined the tts handle 2026-08-31 (callback
    # consumers need the outputs.json location before any poll)
    "text-to-speech": {"job_id", "status", "workflow_id",
                       "output_s3_prefix", "message", "error"},
    "hybrid-search":  {"job_id", "status", "collection_id",
                       "num_inline_documents", "documents_s3uri"},
}


def submit_fields(target: str) -> set[str]:
    """Every field `target`'s start tool is allowed to return."""
    return SUBMIT_HANDLE_FIELDS_PER_TARGET.get(target, SUBMIT_HANDLE_FIELDS)

# Mirrors TOKEN_RE in workflow_identity.py.  A token names no execution, but the
# gateways still validate it as an SFN name (for most tools the workflow_id
# derived from it IS the execution name) — catching it locally turns an
# INVALID_PARAMETER response into an immediate, obvious error.
TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")

# Fixed synthesis text for --run text-to-speech, and the FALLBACK when no
# --tts-transcript-s3-uri is given.  Held as a constant rather than derived per
# call so the idempotency resubmit hashes identically; three short lines keep
# the run to a few minutes of CPU synthesis.
TTS_TRANSCRIPTS = (
    "The gateway is reachable and the token was accepted.",
    "This clip was synthesized by the micro-workflow under test.",
    "Quality control compares the transcription against this sentence.",
)

# --tts-transcript-s3-uri feeds tts one real speaker's dialogue instead of the
# three canned lines above, by reading a completed speech-diarization run's
# `transcripts/final_transcript.json` and taking EVERY sentence carrying one
# `speaker_id`.  That is a materially stronger test than the canned lines: a
# real speaker's turns are short, unevenly punctuated, and include the
# fragments ("Nah, not this time.") that a hand-written sentence never
# produces, and the count is whatever the content happens to contain rather
# than a number chosen to pass.  It also chains two workflows over one asset,
# which is how an agent would actually use them.
#
# What is NOT derived from the transcript, deliberately: `target_durations_s`.
# Each sentence has a real `end - start`, so the 1:1 list tts accepts is right
# there — but lib/duration.py bounds stretching at +-10% and RAISES
# DurationOutOfRangeError past it, and a cloned voice reading someone else's
# line has no reason to land within 10% of the original's pace.  A raised
# clip records {"error": ...} and fails verify_tts_result, so wiring the
# durations in would make the test's verdict depend on how fast the source
# speaker talks.  Duration control needs its own test with its own tolerance.

# Mirrors states/check_and_preprocess/lambda_function.py.  Copied rather than
# imported: this script talks to a deployed gateway over HTTP and must run with
# no tts-on-aws checkout on sys.path.  The point is to reject an over-budget
# transcript list HERE, with the speaker's name and byte count in the message,
# instead of reading INVALID_PARAMETER back from a submit and having to guess
# which of the two bounds was hit.  If those constants move, this drifts
# silently in the SAFE direction only when it stays the more conservative copy
# — so re-check it if a submit is ever rejected that this let through.
TTS_MAX_TRANSCRIPTS = 100                  # MAX_TRANSCRIPTS_PER_BATCH
TTS_MAX_TEXT_CHARS = 1000                  # MAX_TEXT_CHARS
TTS_PAYLOAD_BUDGET_BYTES = int(256 * 1024 * 0.80)   # SFN_PAYLOAD_BUDGET_BYTES
TTS_TOKENS_PER_CHAR = 25.0 / 15.0
TTS_BYTES_PER_TOKEN = 6
TTS_BYTES_PER_TRANSCRIPT_OVERHEAD = 120
TTS_BYTES_PER_OUTPUT_OVERHEAD = 400       # BYTES_PER_OUTPUT_OVERHEAD
TTS_TEXT_COPIES_IN_AGGREGATE = 1          # TEXT_COPIES_IN_AGGREGATE
TTS_MAP_MAX_CONCURRENCY = 40              # MAP_MAX_CONCURRENCY
TTS_BYTES_PER_ITERATOR_ENVELOPE = 1100    # BYTES_PER_ITERATOR_ENVELOPE


def tts_payload_bytes(texts, n_candidates: int = 1) -> int:
    """The LLM -> Flow+HiFT handoff: text plus every candidate's token array.

    n_candidates is 1 because --run always passes an explicit `seed`, which
    forces single-shot synthesis whatever the style; best-of-3 would triple the
    token arrays.  Pass 3 if this harness ever stops pinning the seed.
    """
    return sum(len(t) + TTS_BYTES_PER_TRANSCRIPT_OVERHEAD
               + n_candidates * int(len(t) * TTS_TOKENS_PER_CHAR)
               * TTS_BYTES_PER_TOKEN
               for t in texts)


def tts_aggregate_bytes(texts) -> int:
    """The Map's aggregate ResultPath: no token arrays by then, one copy of each
    text, plus one envelope per concurrent iterator.

    Both bounds are checked because NEITHER dominates, and which one binds
    depends on exactly the thing a diarized transcript varies most — sentence
    length.  Per transcript the batch bound costs ~11*L + 120 bytes and this one
    ~L + 400 (plus a fixed 40-iterator envelope term), so the batch bound is
    tighter per transcript above L = 41.  tts's own docs quote
    its limits at 230 and 1000 chars, both in that regime — but a real speaker's
    turns are full of "That's right." and "Mob heads?", and a hundred of those
    are bounded HERE.  Checking only the documented-tighter one would let
    exactly the short-turn case through.

    Each text used to appear THREE times here (transcripts[].text, qc.expected,
    qc.transcribed), which put the crossover at L = 25 and capped a request at
    63 transcripts at MAX_TEXT_CHARS; verify_output._prune_qc_evidence now emits
    one copy.  The worst case this budgets for is a run where EVERY clip fails
    QC, since a failed clip keeps its qc.transcribed.

    The envelope term is 1100, not the 380 this copy carried until 2026-08-10:
    that figure came from a calibration envelope missing the `voice` record,
    which carries an unbounded `ref_transcript` and is the largest member.  See
    tts-on-aws/README.md §13c.  This copy also keeps the fixed 40 envelopes where
    the real gate now passes the true iterator count — that over-charges a small
    request, which is the safe direction for a client-side pre-check, and it
    changes no ceiling at the lengths that actually bind (n >= 40 there anyway).
    """
    return (TTS_MAP_MAX_CONCURRENCY * TTS_BYTES_PER_ITERATOR_ENVELOPE
            + sum(TTS_TEXT_COPIES_IN_AGGREGATE * len(t)
                  + TTS_BYTES_PER_OUTPUT_OVERHEAD for t in texts))

# Singular and plural are separate tools with deliberately different response
# shapes: get_download_url -> {url, expires_at}; get_download_urls ->
# {urls, count, expires_at}. Both are asserted below.
DOWNLOAD_TOOL = "media-outputs___get_download_url"
DOWNLOAD_BATCH_TOOL = "media-outputs___get_download_urls"

TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}

# ── helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(msg, flush=True)


def _http(method: str, url: str, headers: dict, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:  # nosec B310
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {raw}") from exc


def fetch_credentials(secret_name: str, region: str) -> dict:
    """Fetch gateway credentials; during the media-toolkit-gateway
    rename window the secret may still live at the legacy
    media-workflow-gateway path — fall back so harnesses work on
    both sides of the infra cutover."""
    sm = boto3.client("secretsmanager", region_name=region)
    try:
        value = sm.get_secret_value(SecretId=secret_name)
    except sm.exceptions.ResourceNotFoundException:
        legacy = secret_name.replace("media-toolkit-gateway",
                                     "media-workflow-gateway")
        if legacy == secret_name:
            raise
        value = sm.get_secret_value(SecretId=legacy)
    return json.loads(value["SecretString"])


def resolve_speaker_transcripts(s3_uri: str, region: str,
                                speaker_id: str | None = None,
                                max_cues: int | None = None):
    """Read a speech-diarization `final_transcript.json` and return
    `(speaker_id, [text, ...])` for ONE speaker — that speaker's sentences in
    transcript order, truncated to the first `max_cues` when one is given.

    Called exactly ONCE, from main(), and the result stashed on the parsed
    namespace.  It must not be called from build_args: build_args is a pure
    function of the namespace precisely so the idempotency resubmit sends
    byte-identical arguments, and an S3 read there would put a network fetch
    inside the hash preimage.  Two fetches that returned lists differing by one
    whitespace character would derive two workflow_ids and the resubmit would
    report a contract failure that is really a harness bug.

    Default speaker: the one with the MOST sentences.  Picked by count rather
    than by name because `SPEAKER_00` is not the lead in general — on the
    Meridian clip it holds 15 sentences to `SPEAKER_01`'s 73 — and the point of
    the flag is to exercise a real dialogue load.  Ties break on speaker_id so
    the choice stays deterministic across runs; a resubmit that picked a
    different speaker would silently change the request.

    `max_cues` truncates to the FIRST n sentences, which is why it is applied
    here and not in build_args: the speaker choice must still be made over the
    speaker's whole side (truncating first would let a 10-cue cap pick a
    different lead), and the payload bounds below must measure what is actually
    sent.  Taking a prefix rather than a sample keeps the cues contiguous
    dialogue and keeps the request reproducible — a random sample would derive
    a fresh workflow_id on the resubmit and fail the idempotency assertion on
    harness behavior rather than service behavior.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"--tts-transcript-s3-uri is not an s3 uri: {s3_uri}")
    bucket, _, key = s3_uri[len("s3://"):].partition("/")
    if not bucket or not key:
        raise ValueError(f"--tts-transcript-s3-uri needs a bucket and a key: "
                         f"{s3_uri}")
    body = boto3.client("s3", region_name=region).get_object(
        Bucket=bucket, Key=key)["Body"].read()
    doc = json.loads(body)

    sentences = doc.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError(f"{s3_uri} has no sentences[] — is it a "
                         f"speech-diarization final_transcript.json?")

    by_speaker: dict[str, list[str]] = {}
    for s in sentences:
        text = (s.get("text") or "").strip()
        if text:                      # a diarized turn can be empty; tts rejects ""
            by_speaker.setdefault(s.get("speaker_id") or "UNKNOWN",
                                  []).append(text)
    if not by_speaker:
        raise ValueError(f"{s3_uri} has sentences but none with text")

    if speaker_id is None:
        speaker_id = min(by_speaker, key=lambda k: (-len(by_speaker[k]), k))
    elif speaker_id not in by_speaker:
        raise ValueError(f"--tts-speaker-id {speaker_id!r} is not in {s3_uri}; "
                         f"available: {sorted(by_speaker)}")
    texts = by_speaker[speaker_id]
    available = len(texts)
    if max_cues is not None and max_cues < available:
        texts = texts[:max_cues]

    log(f"  transcript:   {s3_uri}")
    log(f"  language:     {doc.get('language')}  "
        f"speakers: {sorted(by_speaker)}")
    log(f"  speaker:      {speaker_id} "
        f"({available} of {len(sentences)} sentences)")
    if max_cues is not None:
        log(f"  cues:         {len(texts)} (--tts-max-cues {max_cues}"
            f"{'' if max_cues < available else ', not reached'})")

    # Both of tts's bounds, checked here so the message names the speaker.
    over = [t for t in texts if len(t) > TTS_MAX_TEXT_CHARS]
    if over:
        raise ValueError(
            f"speaker {speaker_id} has {len(over)} sentence(s) over "
            f"{TTS_MAX_TEXT_CHARS} chars (longest {max(map(len, over))}); tts "
            f"would reject the request")
    if len(texts) > TTS_MAX_TRANSCRIPTS:
        raise ValueError(
            f"speaker {speaker_id} has {len(texts)} sentences, over tts's "
            f"{TTS_MAX_TRANSCRIPTS}-per-batch cap. Pick a lighter speaker with "
            f"--tts-speaker-id (available: {sorted(by_speaker)})")
    batch_est = tts_payload_bytes(texts)
    agg_est = tts_aggregate_bytes(texts)
    for est, what in ((batch_est, "LLM->Flow batch"),
                      (agg_est, "Map aggregate")):
        if est > TTS_PAYLOAD_BUDGET_BYTES:
            raise ValueError(
                f"speaker {speaker_id}'s {len(texts)} sentences estimate "
                f"{est} bytes of {what} payload, over tts's "
                f"{TTS_PAYLOAD_BUDGET_BYTES}-byte budget (the 256 KiB Step "
                f"Functions limit at 80%). Pick a lighter speaker with "
                f"--tts-speaker-id (available: {sorted(by_speaker)})")
    log(f"  payload est:  {batch_est} batch / {agg_est} aggregate, of "
        f"{TTS_PAYLOAD_BUDGET_BYTES} budget "
        f"({sum(len(t) for t in texts)} chars of text)")
    return speaker_id, texts


def get_token(token_endpoint: str, client_id: str, client_secret: str, scope: str) -> str:
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "scope": scope,
    }).encode()
    req = urllib.request.Request(
        token_endpoint,
        data=body,
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:  # nosec B310
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        raise RuntimeError(f"Token request failed HTTP {exc.code}: {raw}") from exc
    return payload["access_token"]


def mcp_request(gateway_url: str, token: str, method: str, params: dict | None = None) -> dict:
    body = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body["params"] = params
    return _http(
        "POST",
        gateway_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        body=body,
    )


def call_tool(gateway_url: str, token: str, tool_name: str, arguments: dict) -> dict:
    resp = mcp_request(gateway_url, token, "tools/call", {
        "name": tool_name,
        "arguments": arguments,
    })
    if "error" in resp:
        raise RuntimeError(f"tools/call error: {resp['error']}")
    # MCP result is in resp["result"]["content"][0]["text"] as JSON string
    result = resp.get("result", {})
    content = result.get("content", [])
    if content and content[0].get("type") == "text":
        text = content[0]["text"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Not every reply on this path is the workflows' JSON error
            # contract: AgentCore's own schema validation rejects a bad
            # call with PLAIN TEXT, and some admin tools answer with a
            # bare string. Surface the text instead of crashing the
            # caller with a JSONDecodeError that hides the actual reason.
            return {"_raw_text": text}
    return result


# ── test phases ───────────────────────────────────────────────────────────────

def test_tools_list(gateway_url: str, token: str) -> bool:
    log("\n── tools/list ──────────────────────────────────────────────────────")
    # The gateway paginates tools/list (30/page) — follow nextCursor.
    advertised: set = set()
    cursor = None
    for _ in range(20):  # page cap: a repeated/echoed cursor must not hang the test
        params = {"cursor": cursor} if cursor else None
        resp = mcp_request(gateway_url, token, "tools/list", params)
        if "error" in resp:
            log(f"  FAIL: {resp['error']}")
            return False
        result = resp.get("result", {})
        advertised |= {t["name"] for t in result.get("tools", [])}
        prev, cursor = cursor, result.get("nextCursor")
        if not cursor or cursor == prev:
            break
    else:
        log("  FAIL: tools/list did not terminate within 20 pages")
        return False
    log(f"  Advertised tools ({len(advertised)}): {sorted(advertised)}")

    # Conditional CFN targets (empty ARN param omits them). A fully-absent
    # optional target is a deployment choice, not a failure; a PARTIALLY
    # advertised one is always a failure.
    expected = set()
    for target, names in TOOLS.items():
        if target in OPTIONAL_TARGETS and not (set(names) & advertised):
            log(f"  (optional target '{target}' not deployed — skipping)")
            continue
        expected |= set(names)
    missing = expected - advertised
    extra   = advertised - expected

    if missing:
        log(f"  MISSING: {sorted(missing)}")
    if extra:
        log(f"  EXTRA (unexpected, not a failure): {sorted(extra)}")

    ok = not missing
    log(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def build_args(target: str, args_ns) -> dict:
    """
    Build one target's start-tool arguments.  Kept separate from submit_job so
    the idempotency pass re-submits BYTE-IDENTICAL arguments — workflow_id is
    uuid5 over the canonical SFN input, so any drift between the two calls
    (a re-derived default, a reordered list) starts a fresh run and the test
    would report a false idempotency failure.
    """
    # output_s3_prefix is service-derived (micro-workflow/{target}/{workflow_id})
    args: dict = {}
    if target in SHARED_CONTRACT_TARGETS:
        args["asset_s3uri"] = args_ns.asset_s3_uri

    # Minimal non-default arguments to keep the job cheap/fast in testing
    if target == "audio-stem":
        args["music_stem"]  = True
        args["sfx_stem"]    = True
    elif target == "media-processing":
        # Single mp4 proxy only (default) — no extra groups
        pass
    elif target == "speech-diarization":
        args["beam_size"]        = 1   # fastest ASR setting
        args["speaker_channels"] = True
    elif target == "shot-segmentation":
        # Default hyperparameters are sufficient for testing
        pass
    elif target == "face-recognition":
        # Keep test runs cheap and side-effect free: no auto-indexing
        args["max_faces_to_index"] = 0
        if args_ns.collection_id:
            args["collection_id"] = args_ns.collection_id
    elif target == "text-to-speech":
        # No asset_s3uri — tts synthesizes text, it does not read the asset.
        # seed pins the SAMPLING so a resubmit's audio is comparable;
        # --idempotency-token (added below) is what makes the SUBMIT
        # idempotent. Both are needed: without the seed the audio would differ
        # run to run, and without the token there would be no second run to
        # compare against because the gateway would start a fresh one.
        args["voice_id"]    = args_ns.tts_voice_id
        # One real speaker's dialogue when --tts-transcript-s3-uri resolved it
        # (main() stashes the list on the namespace before any submit), else the
        # canned lines. Read from the namespace, never re-resolved, so the
        # resubmit hashes identically — see resolve_speaker_transcripts.
        args["transcripts"] = list(getattr(args_ns, "tts_transcripts", None)
                                   or TTS_TRANSCRIPTS)
        args["flow_steps"]  = 6
        args["seed"]        = 1234
    elif target == "hybrid-search":
        # documents_s3uri (a JSONL contract manifest under the proxy bucket)
        # must be supplied — the harness does not author contracts, since they
        # are derived from another workflow's real output.
        args["collection_id"]    = args_ns.collection_id or "default"
        args["documents_s3uri"] = args_ns.documents_s3_uri

    # The token folds into the workflow_id hash and is then discarded.  Only the
    # targets that declare it get one; `hybrid-search` is left out because its
    # execution is uuid4-named and a token would narrow nothing.  Sending it to a
    # target that ignores tokens would not fail — since 2026-08-06 every gateway
    # accepts the field and logs a warning rather than erroring — but passing it
    # only where it is honoured keeps the resubmit hash preimage honest.
    #
    # For face-recognition and text-to-speech it is REQUIRED by --idempotency,
    # not cosmetic: with no token the gateway generates one per call, so a
    # resubmit deliberately starts a fresh run and the contract assertion would
    # fail.  See TOKEN_REQUIRED_FOR_IDEMPOTENCY.
    if args_ns.idempotency_token and target in TOKEN_ACCEPTING_TARGETS:
        args["idempotency_token"] = args_ns.idempotency_token
    return args


def submit_job(gateway_url: str, token: str, target: str,
               args_ns, label: str = "submit") -> dict | None:
    """Call one target's start tool. Returns the full result dict (the
    idempotency pass needs workflow_id and status, not just job_id)."""
    log(f"\n── {label} {target} ─────────────────────────────────────────────────")

    args = build_args(target, args_ns)
    tool = START_TOOL[target]
    log(f"  Calling {tool}")
    log(f"  Args: {json.dumps(args, indent=4)}")

    try:
        result = call_tool(gateway_url, token, tool, args)
    except RuntimeError as exc:
        log(f"  FAIL: {exc}")
        return None

    log(f"  Result: {json.dumps(result, indent=4)}")
    job_id = result.get("job_id")
    if not job_id:
        log("  FAIL: no job_id in response")
        return None

    # A submit is a handle on EVERY path, not just the absorbed one. Checked
    # here as well as in the idempotency pass because --run alone is what most
    # people invoke, and a result field leaking into a fresh start would
    # otherwise go unnoticed until someone happened to pass --idempotency.
    leaked = set(result) - submit_fields(target)
    if leaked:
        log(f"  FAIL: submit returned non-handle field(s) {sorted(leaked)} — "
            f"a submit names a run, it does not report one; results belong to "
            f"{STATUS_TOOL[target]}")
        return None

    # Not a failure: re-running the harness under a token it already used
    # legitimately lands on the existing run. Worth saying out loud, because it
    # means --idempotency is comparing a resubmit against a resubmit.
    if result.get("message"):
        log(f"  NOTE: absorbed into an existing run — {result['message']}")

    log(f"  PASS — job_id: {job_id}")
    return result


def _cell(status: str, state: str, width: int) -> str:
    """One table cell: 'RUNNING: <state>' or a bare terminal status."""
    text = f"{status}: {state}" if state and status == "RUNNING" else status
    if len(text) > width:
        text = text[: width - 1] + "…"
    return text.ljust(width)


def poll_jobs(gateway_url: str, token: str, job_ids: dict[str, str],
              interval: int, timeout: int, args_ns) -> dict[str, dict | None]:
    """Poll all submitted jobs in one interleaved loop, printing one table row
    per cycle (one column per target).

    Returns {target: terminal result dict}, or None for a target that timed
    out. Each SUCCEEDED result is checked against its contract and the verdict
    stored under the "_verified" key, so a caller can tell "reached a terminal
    state" apart from "and the result was valid"."""
    targets  = list(job_ids)
    col      = max(24, *(len(t) for t in targets)) + 16
    pending  = dict(job_ids)
    cells    = {t: "SUBMITTED".ljust(col) for t in targets}
    results: dict[str, dict | None] = {}
    deadline = time.monotonic() + timeout
    attempt  = 0
    start    = time.monotonic()

    header = "  " + "elapsed".ljust(9) + "  ".join(t.ljust(col) for t in targets)

    def print_header() -> None:
        log(header)
        log("  " + "-" * (len(header.rstrip()) - 2))

    log("\n── poll ────────────────────────────────────────────────────────────")
    print_header()

    while pending and time.monotonic() < deadline:
        attempt += 1
        terminal_results: dict[str, dict] = {}

        for target, job_id in list(pending.items()):
            try:
                result = call_tool(gateway_url, token, STATUS_TOOL[target],
                                   {"job_id": job_id})
            except RuntimeError as exc:
                cells[target] = _cell("POLL_ERROR", "", col)
                log(f"  {target}: poll error: {exc}")
                continue

            status = result.get("status", "UNKNOWN")
            cells[target] = _cell(status, result.get("current_state", ""), col)
            if status in TERMINAL_STATUSES:
                terminal_results[target] = result
                del pending[target]

        elapsed = int(time.monotonic() - start)
        row = f"{elapsed // 60:3d}:{elapsed % 60:02d}"
        log("  " + row.ljust(9) + "  ".join(cells[t] for t in targets))

        # Details printed after the row so the table stays aligned
        for target, result in terminal_results.items():
            log(f"\n  {target} result: {json.dumps(result, indent=4)}")
            results[target] = result
            ok = result.get("status") == "SUCCEEDED"
            if ok:
                ok = verify_result(gateway_url, token, target, result,
                                   args_ns)
            log(f"  {'PASS' if ok else 'FAIL'} {target} "
                f"(terminal status: {result.get('status')})\n")
            result["_verified"] = ok

        if pending:
            if terminal_results:
                print_header()
            time.sleep(interval)

    for target in pending:
        log(f"  FAIL {target}: poll timed out after {timeout}s")
        results[target] = None
    return results


def verify_result(gateway_url: str, token: str, target: str,
                  result: dict, args_ns) -> bool:
    """Dispatch a SUCCEEDED result to the right contract check. Two of the seven
    workflow targets have deliberately different response shapes, so a single
    outputs[]-based check cannot cover them (see the module docstring).

    There is no token round-trip to check: as of 2026-08-06 the token is
    consumed into the workflow_id hash and discarded, so nothing echoes it.
    The predecessor check (verify_uuid_echo) was a test whose subject was the
    echo itself, and the echo was the entire blast radius of the one bug this
    design replaced."""
    if target in SHARED_CONTRACT_TARGETS:
        return verify_outputs(gateway_url, token, target, result)
    if target == "text-to-speech":
        # The expected clip count is whatever was SUBMITTED, which with
        # --tts-transcript-s3-uri is one speaker's real sentence count and not
        # len(TTS_TRANSCRIPTS). Re-deriving it through build_args keeps the one
        # source of truth rather than a second copy that can disagree.
        return verify_tts_result(
            gateway_url, token, result,
            len(build_args("text-to-speech", args_ns)["transcripts"]))
    if target == "hybrid-search":
        return verify_index_result(result)
    log(f"  FAIL: no result check defined for target {target!r}")
    return False


def verify_tts_result(gateway_url: str, token: str, result: dict,
                      expected_clips: int) -> bool:
    """text-to-speech's synthesis result: per-transcript outputs[] entries keyed
    by transcript_index rather than {uri, mime, label} triples, plus the
    batch_qc_summary rollup.

    The shared workflow_id/output_s3_prefix/usage_uri fields ARE asserted below
    as of 2026-08-07 — they were missing from this target until then, which was a
    plain gap rather than part of the outputs[] shape difference.  Asserting them
    here is what stops them regressing on the live gateway; the unit tests in
    tts-on-aws/test/test_units.py only cover the parser.
    """
    log("\n── verify outputs text-to-speech ───────────────────────────────────")

    outputs = result.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        log("  FAIL: no outputs[] in SUCCEEDED result")
        return False
    if len(outputs) != expected_clips:
        log(f"  FAIL: expected {expected_clips} clips, got {len(outputs)}")
        return False

    uris = []
    for i, entry in enumerate(outputs):
        # transcript_index is a GLOBAL index across the flattened batch list,
        # so with one batch it must be dense and in order.
        if entry.get("transcript_index") != i:
            log(f"  FAIL: expected transcript_index {i}, got: {entry}")
            return False
        if entry.get("error"):
            log(f"  FAIL: clip {i} produced no output: {entry['error']}")
            return False
        uri = entry.get("uri")
        if not uri or not uri.startswith("s3://"):
            log(f"  FAIL: clip {i} has no s3 uri: {entry}")
            return False
        uris.append(uri)
    log(f"  outputs[] OK ({len(outputs)} clips)")

    # QC is informational by design (whisper-base's transcription is not
    # deterministic across invocations, so a fixed threshold flips on
    # borderline clips). The ROLLUP's presence and self-consistency IS
    # asserted; a qc failure is reported without failing the test.
    qc = result.get("batch_qc_summary")
    if not isinstance(qc, dict):
        log("  FAIL: no batch_qc_summary rollup in SUCCEEDED result")
        return False
    if qc.get("passed", 0) + qc.get("failed", 0) != qc.get("total"):
        log(f"  FAIL: batch_qc_summary does not add up: {qc}")
        return False
    if (qc.get("failed", 0) > 0) != bool(qc.get("failed_indices")):
        log(f"  FAIL: failed count and failed_indices disagree: {qc}")
        return False
    if qc.get("all_passed") is not (qc.get("failed") == 0):
        log(f"  FAIL: all_passed disagrees with failed count: {qc}")
        return False
    log(f"  batch_qc_summary OK ({qc['passed']}/{qc['total']} passed"
        + (f", failed_indices={qc['failed_indices']} — informational, not a "
           f"test failure" if qc.get("failed") else "") + ")")

    # Sibling-parity fields. Only the outputs[] SHAPE is allowed to differ here;
    # these three were simply absent until 2026-08-07 (the usage file was written
    # and unreachable, and the poll returned no workflow_id at all).
    for field, why in (("workflow_id",
                        "names the run when only a job_id was kept"),
                       ("output_s3_prefix",
                        "where the clips were written"),
                       ("usage_uri",
                        "locates usage_tracking.json's per-state breakdown")):
        if not result.get(field):
            log(f"  FAIL: no {field} in SUCCEEDED result — {why}")
            return False
    if not str(result["output_s3_prefix"]).startswith("s3://"):
        log(f"  FAIL: output_s3_prefix is not an s3 uri: "
            f"{result['output_s3_prefix']!r}")
        return False
    # Every clip must live under the advertised prefix, or the prefix is
    # describing some other run's location.
    stray = [u for u in uris if not u.startswith(result["output_s3_prefix"])]
    if stray:
        log(f"  FAIL: {len(stray)} clip(s) outside output_s3_prefix "
            f"{result['output_s3_prefix']}: {stray[:2]}")
        return False
    log(f"  parity fields OK (workflow_id={result['workflow_id']}, "
        f"prefix={result['output_s3_prefix']})")

    return verify_downloads(gateway_url, token, uris)


def verify_index_result(result: dict) -> bool:
    """hybrid-search's index result: prepared/skipped/indexed counts and no
    outputs[] at all (rows live in the search backend, not as S3 artifacts)."""
    log("\n── verify counts hybrid-search ─────────────────────────────────────")

    if "outputs" in result:
        log(f"  FAIL: index_content must not return outputs[]: {result}")
        return False
    counts = {k: result.get(k) for k in ("prepared", "skipped", "indexed")}
    if any(not isinstance(v, int) for v in counts.values()):
        log(f"  FAIL: prepared/skipped/indexed must all be ints: {counts}")
        return False
    if not result.get("collection_id"):
        log("  FAIL: no collection_id in SUCCEEDED result")
        return False
    if counts["prepared"] + counts["skipped"] == 0:
        log(f"  FAIL: nothing was prepared OR skipped — the manifest resolved "
            f"to zero documents: {counts}")
        return False
    # indexed counts ROWS written and prepared counts rows built, so on a first
    # run they match; a shortfall means a Map branch dropped rows silently.
    if counts["indexed"] != counts["prepared"]:
        log(f"  FAIL: indexed != prepared — rows were built but not written: "
            f"{counts}")
        return False
    log(f"  counts OK (collection={result['collection_id']}, "
        f"prepared={counts['prepared']}, skipped={counts['skipped']}, "
        f"indexed={counts['indexed']})")
    return True


def verify_outputs(gateway_url: str, token: str, target: str, result: dict) -> bool:
    """Check the outputs[] contract on a SUCCEEDED result, then presign the
    first few file entries via both media-outputs download tools and fetch one
    URL over HTTP."""
    log(f"\n── verify outputs {target} ─────────────────────────────────────────")

    if not result.get("workflow_id"):
        log("  FAIL: no workflow_id in SUCCEEDED result")
        return False

    outputs = result.get("outputs")
    if not outputs:
        log("  FAIL: no outputs[] in SUCCEEDED result")
        return False

    for entry in outputs:
        uri, label = entry.get("uri"), entry.get("label")
        if not uri or not uri.startswith("s3://") or not label:
            log(f"  FAIL: malformed outputs entry: {entry}")
            return False
        is_prefix = uri.endswith("/")
        if is_prefix and "mime" in entry:
            log(f"  FAIL: prefix entry must not carry mime: {entry}")
            return False
        if not is_prefix and not entry.get("mime"):
            log(f"  FAIL: file entry missing mime: {entry}")
            return False
    log(f"  outputs[] OK ({len(outputs)} entries)")

    files = [e["uri"] for e in outputs if not e["uri"].endswith("/")]
    if not files:
        log("  No file entries to presign — skipping download check")
        return True
    return verify_downloads(gateway_url, token, files)


def verify_downloads(gateway_url: str, token: str, uris: list[str]) -> bool:
    """Presign the first few uris via BOTH media-outputs download tools and
    fetch one URL over HTTP."""
    # Batch-presign up to 3 file entries in one call, plus a deliberately
    # invalid uri to assert per-item failure does not sink the good entries.
    batch = list(uris[:3])
    bad = "s3://wrong-bucket/micro-workflow/nope.json"
    log(f"  Presigning {len(batch)} file(s) + 1 invalid uri in one call")
    try:
        presign = call_tool(gateway_url, token, DOWNLOAD_BATCH_TOOL, {
            "uris": batch + [bad],
            "expiry_seconds": 300,
        })
    except RuntimeError as exc:
        log(f"  FAIL: get_download_urls error: {exc}")
        return False

    urls = presign.get("urls")
    if not isinstance(urls, list) or len(urls) != len(batch) + 1:
        log(f"  FAIL: expected {len(batch) + 1} urls entries, got: {presign}")
        return False
    if [u.get("uri") for u in urls] != batch + [bad]:
        log(f"  FAIL: urls[] must echo uris in request order: {presign}")
        return False
    if not urls[-1].get("error"):
        log(f"  FAIL: invalid uri should have failed its own entry: {urls[-1]}")
        return False
    if presign.get("count") != len(batch):
        log(f"  FAIL: count should be {len(batch)}, got {presign.get('count')}")
        return False
    if any(u.get("error") for u in urls[:-1]):
        log(f"  FAIL: a valid uri failed alongside the invalid one: {urls}")
        return False
    log(f"  Batch presign OK ({presign['count']}/{len(urls)} signed, "
        f"invalid entry rejected individually)")

    # The singular tool is a frozen contract: a FLAT {url, expires_at}, with no
    # urls/count nesting. A caller reading result["url"] must keep working.
    log("  Presigning 1 file via the singular tool (flat {url} shape)")
    try:
        single = call_tool(gateway_url, token, DOWNLOAD_TOOL, {
            "s3uri_or_key": batch[0],
            "expiry_seconds": 300,
        })
    except RuntimeError as exc:
        log(f"  FAIL: get_download_url error: {exc}")
        return False

    if not isinstance(single.get("url"), str) or not single.get("expires_at"):
        log(f"  FAIL: singular tool must return flat url + expires_at: {single}")
        return False
    if "urls" in single or "count" in single:
        log(f"  FAIL: singular tool must NOT return the batch shape: {single}")
        return False
    log("  Singular presign OK (flat url, no urls/count)")

    url = single["url"]

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:  # nosec B310
            body = resp.read(1024)
    except urllib.error.HTTPError as exc:
        log(f"  FAIL: presigned GET returned HTTP {exc.code}")
        return False
    log(f"  Presigned GET OK ({len(body)} bytes read, expires_at={presign.get('expires_at')})")
    return True


def test_idempotency(gateway_url: str, token: str, args_ns,
                     first: dict[str, dict]) -> dict[str, bool]:
    """Re-submit every target with byte-identical arguments and assert its
    documented resubmit contract. `first` maps target -> the first submit's
    result dict. Returns {target: pass/fail}.

    A submit is a HANDLE, never results — so this pass asserts the shape first
    and then POLLS for anything it wants to read. Both halves are the point.

    Until 2026-08-07 an absorbed resubmit described the existing run and handed
    back its full results, so the submit's shape depended on WHEN it was called:
    a fresh start answered with five fields and a resubmit of a finished run
    with the whole result set. This harness encoded that by calling
    verify_result() on a submit response. The contract now says a submit returns
    the same fields either way and only `status` differs, which is why the extra
    field check below asserts the whole allowed set rather than probing for one
    leaked key — a regression that re-adds results has to fail here, not pass
    quietly on the fields that happen to still be present.

    `message` is the ONLY thing that distinguishes an absorbed call from a fresh
    one, and that is not a stylistic point: the execution name is derived from
    the same hash, so a fresh start of a vanished run would return a
    byte-identical `job_id`. Comparing ARNs cannot tell the two apart."""
    log("\n══ idempotency ═════════════════════════════════════════════════════")
    targets = [t for t in first if t not in IDEMPOTENCY_EXCLUDED]
    log("  Re-submitting identical arguments. Expected per target:")
    for target in targets:
        log(f"    {target:<20} {IDEMPOTENCY_CONTRACT[target]}")
    for target in first:
        if target in IDEMPOTENCY_EXCLUDED:
            log(f"    {target:<20} (excluded by design — see "
                f"IDEMPOTENCY_EXCLUDED)")

    results: dict[str, bool] = {}

    for target in targets:
        original = first[target]
        contract = IDEMPOTENCY_CONTRACT[target]
        again = submit_job(gateway_url, token, target, args_ns,
                           label="resubmit")
        if again is None:
            results[f"{target}/idempotency"] = False
            continue

        ok = True
        same_workflow = again.get("workflow_id") == original.get("workflow_id")
        same_job      = again.get("job_id") == original.get("job_id")
        status        = again.get("status")

        assert contract == RETURNS_EXISTING, contract   # the only contract
        if not same_workflow:
            log(f"  FAIL: workflow_id changed on resubmit: "
                f"{original.get('workflow_id')} -> {again.get('workflow_id')}")
            ok = False
        if not same_job:
            log(f"  FAIL: job_id changed on resubmit: "
                f"{original.get('job_id')} -> {again.get('job_id')}")
            ok = False

        # A submit is a handle. Anything beyond the declared handle fields is a
        # result leaking back into the submit path — the exact regression the
        # 2026-08-07 change removed.
        leaked = set(again) - submit_fields(target)
        if leaked:
            log(f"  FAIL: resubmit returned non-handle field(s) "
                f"{sorted(leaked)} — a submit names a run, it does not report "
                f"one; results belong to {STATUS_TOOL[target]}")
            ok = False

        # `message` is the only observable difference between an absorbed
        # resubmit and a fresh start, and its absence here is not cosmetic: it
        # means this call started a SECOND execution, which is the failure the
        # whole contract exists to prevent. The identical job_id above cannot
        # rule that out — the name is a hash of the same input either way.
        if not again.get("message"):
            log("  FAIL: no `message` on the resubmit — that field is the only "
                "signal the call was absorbed rather than started, so its "
                "absence means a second execution was launched")
            ok = False

        if status == "RUNNING":
            log("  Absorbed into the existing RUNNING run "
                f"(current_state={again.get('current_state') or 'unknown'})")
        elif status == "SUCCEEDED":
            # The submit told us the run finished and nothing more. Reading its
            # results is get_job_status's job, so poll for them — that round
            # trip IS the contract, not an inefficiency to optimize away.
            if "current_state" in again:
                log("  FAIL: current_state present on a SUCCEEDED resubmit — a "
                    "stopped execution is in no state")
                ok = False
            log("  Absorbed into the existing SUCCEEDED run; polling "
                "get_job_status for its results")
            try:
                polled = call_tool(gateway_url, token, STATUS_TOOL[target],
                                   {"job_id": again["job_id"]})
            except RuntimeError as exc:
                log(f"  FAIL: could not poll the absorbed run: {exc}")
                ok = False
            else:
                if polled.get("status") != "SUCCEEDED":
                    log(f"  FAIL: submit said SUCCEEDED, poll says "
                        f"{polled.get('status')!r}")
                    ok = False
                elif not verify_result(gateway_url, token, target, polled,
                                       args_ns):
                    ok = False
                elif ok:
                    log("  Existing run's results read back from "
                        "get_job_status (no new execution)")
        else:
            # A terminal failure is STICKY: Step Functions reserves the
            # execution name for 90 days, so this same request keeps returning
            # this same verdict for that whole window. Only a changed
            # parameter, a different --idempotency-token, or a new tool version
            # re-runs it — which is why a failed run is worth reporting loudly
            # rather than as a transient poll result.
            log(f"  FAIL: existing run is {status!r} — that verdict holds for "
                f"these exact arguments until the reserved execution name "
                f"expires (~90 days); change a parameter or the "
                f"--idempotency-token to re-run. cause: "
                f"{again.get('error') or '(none reported)'}")
            if not again.get("error"):
                log("  FAIL: no `error` on a terminal-failure resubmit — the "
                    "cause is what decides between fixing a parameter and "
                    "supplying a new token")
            ok = False

        results[f"{target}/idempotency"] = ok

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test the AgentCore Gateway directly (no agent required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--secret-name",  required=True,
                   help="Secrets Manager secret name (e.g. /media-toolkit-gateway/dev/gateway-credentials)")
    p.add_argument("--region",       default=DEFAULT_REGION)
    p.add_argument("--run",
                   help="Submit a job and poll to completion. One target, a "
                        "comma-separated list of them, or 'all'. Omit to list "
                        f"tools only. Targets: {', '.join(WORKFLOW_TARGETS)}. "
                        "A list is useful when one target's input is another's "
                        "output (tts reads speech-diarization's transcript), "
                        "since those cannot share a single pass.")
    p.add_argument("--asset-s3-uri",
                   help="Source media S3 URI (required for every --run target "
                        "except text-to-speech and hybrid-search)")
    p.add_argument("--idempotency-token",
                   help="Idempotency token for face-recognition and "
                        "text-to-speech (1-80 chars of A-Za-z0-9_.-). Folded "
                        "into workflow_id and discarded — never echoed back. "
                        "Reuse it with --idempotency to assert resubmit "
                        "behavior; the other targets reject it.")
    p.add_argument("--idempotency", action="store_true",
                   help="After the first pass completes, re-submit every "
                        "target with identical arguments and assert its "
                        "documented resubmit contract.")
    p.add_argument("--collection-id",
                   help="Face collection for face-recognition / search "
                        "collection for hybrid-search (each stack's default "
                        "is used when omitted)")
    p.add_argument("--documents-s3-uri",
                   help="hybrid-search JSONL contract manifest under the proxy "
                        "bucket (required for --run hybrid-search)")
    p.add_argument("--tts-voice-id",
                   help="Registered voice id (required for --run text-to-speech; "
                        "list them with text-to-speech___list_voice_collection)")
    p.add_argument("--tts-transcript-s3-uri",
                   help="A completed speech-diarization run's "
                        "transcripts/final_transcript.json. Synthesizes ALL "
                        "sentences of one speaker_id instead of the canned "
                        "lines — real dialogue, real sentence count. Only "
                        "affects --run text-to-speech.")
    p.add_argument("--tts-speaker-id",
                   help="Which speaker_id to synthesize from "
                        "--tts-transcript-s3-uri (default: the one with the "
                        "most sentences)")
    p.add_argument("--tts-max-cues", type=int,
                   help="Synthesize only the FIRST n of the chosen speaker's "
                        "sentences (default: all of them). The speaker is still "
                        "chosen over their whole side, so a cap cannot change "
                        "which speaker is the lead. Requires "
                        "--tts-transcript-s3-uri.")
    p.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL_SEC)
    p.add_argument("--poll-timeout",  type=int, default=DEFAULT_POLL_TIMEOUT_SEC)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Validate up front — a missing per-target argument must not surface as an
    # async workflow failure ten minutes into a --run all.  Order is preserved
    # as written so a caller can control submit order; duplicates collapse,
    # since submitting one target twice in a pass would race two identical
    # requests and the second would be absorbed by the first.
    if not args.run:
        targets = []
    elif args.run == "all":
        targets = list(WORKFLOW_TARGETS)
    else:
        targets, seen = [], set()
        for name in (n.strip() for n in args.run.split(",")):
            if not name or name in seen:
                continue
            if name not in WORKFLOW_TARGETS:
                print(f"--run: unknown target {name!r}; choose from "
                      f"{', '.join(WORKFLOW_TARGETS)} or 'all'",
                      file=sys.stderr)
                sys.exit(1)
            seen.add(name)
            targets.append(name)
    missing: list[str] = []
    if any(t in SHARED_CONTRACT_TARGETS for t in targets) and not args.asset_s3_uri:
        missing.append("--asset-s3-uri")
    if "text-to-speech" in targets and not args.tts_voice_id:
        missing.append("--tts-voice-id (required for text-to-speech)")
    if "hybrid-search" in targets and not args.documents_s3_uri:
        missing.append("--documents-s3-uri (required for hybrid-search)")
    if missing:
        print("missing required argument(s): " + ", ".join(missing),
              file=sys.stderr)
        sys.exit(1)
    if args.idempotency_token and not TOKEN_RE.match(args.idempotency_token):
        print("--idempotency-token must be 1-80 chars of A-Z a-z 0-9 _ . - "
              "(it names nothing, but the gateways validate it as an SFN "
              "name)", file=sys.stderr)
        sys.exit(1)
    if args.idempotency and not args.run:
        print("--idempotency requires --run", file=sys.stderr)
        sys.exit(1)
    if args.tts_speaker_id and not args.tts_transcript_s3_uri:
        print("--tts-speaker-id requires --tts-transcript-s3-uri",
              file=sys.stderr)
        sys.exit(1)
    if args.tts_max_cues is not None:
        if not args.tts_transcript_s3_uri:
            print("--tts-max-cues requires --tts-transcript-s3-uri (there is "
                  "nothing to cap in the canned lines)", file=sys.stderr)
            sys.exit(1)
        if args.tts_max_cues < 1:
            print("--tts-max-cues must be at least 1", file=sys.stderr)
            sys.exit(1)
    # For these two targets an absent token means "do not deduplicate me" — the
    # gateway generates one per call, so a resubmit correctly starts a fresh run.
    # Without a token the assertion would fail on correct behavior, so say why
    # up front instead.
    if args.idempotency:
        needs_token = sorted(set(targets) & TOKEN_REQUIRED_FOR_IDEMPOTENCY)
        if needs_token and not args.idempotency_token:
            print(f"--idempotency requires --idempotency-token for "
                  f"{', '.join(needs_token)}: an absent token means 'start a "
                  f"fresh run' there, so a resubmit would correctly NOT return "
                  f"the existing run", file=sys.stderr)
            sys.exit(1)

    # Resolved ONCE, before any submit, and stashed on the namespace: every
    # build_args call — first pass and resubmit — then reads the same list, so
    # the two derive the same workflow_id. Fatal on failure rather than falling
    # back to TTS_TRANSCRIPTS: a silent fallback would report PASS for the
    # canned lines while claiming to have tested a real speaker.
    args.tts_transcripts = None
    if args.tts_transcript_s3_uri and "text-to-speech" in targets:
        log("\n── resolving tts transcripts ───────────────────────────────────────")
        try:
            speaker, args.tts_transcripts = resolve_speaker_transcripts(
                args.tts_transcript_s3_uri, args.region, args.tts_speaker_id,
                args.tts_max_cues)
        except Exception as exc:
            print(f"could not resolve --tts-transcript-s3-uri: {exc}",
                  file=sys.stderr)
            sys.exit(1)

    log(f"Region:      {args.region}")
    log(f"Secret name: {args.secret_name}")

    # 1. Credentials
    log("\n── fetching credentials ────────────────────────────────────────────")
    creds = fetch_credentials(args.secret_name, args.region)
    log(f"  endpoint:       {creds['endpointUrl']}")
    log(f"  tokenEndpoint:  {creds['tokenEndpoint']}")
    log(f"  clientId:       {creds['clientId']}")
    log(f"  scope:          {creds['scope']}")

    # 2. Token
    log("\n── obtaining token ─────────────────────────────────────────────────")
    token = get_token(
        creds["tokenEndpoint"],
        creds["clientId"],
        creds["clientSecret"],
        creds["scope"],
    )
    log("  Token obtained (not printed for security)")

    gateway_url = creds["endpointUrl"]
    results: dict[str, bool] = {}

    # 3. tools/list
    results["tools/list"] = test_tools_list(gateway_url, token)

    # 4. Optional job runs
    if args.run:
        job_ids: dict[str, str] = {}
        submitted: dict[str, dict] = {}

        for target in targets:
            submit = submit_job(gateway_url, token, target, args)
            if submit:
                job_ids[target] = submit["job_id"]
                submitted[target] = submit
            else:
                results[f"{target}/submit"] = False

        if job_ids:
            poll_results = poll_jobs(gateway_url, token, job_ids,
                                     args.poll_interval, args.poll_timeout,
                                     args)
            for target, terminal in poll_results.items():
                results[f"{target}/poll"] = bool(
                    terminal and terminal.get("_verified"))

        # 5. Optional idempotency pass — only over targets whose first pass
        #    actually submitted, since the contract is about a RESUBMIT.
        if args.idempotency and submitted:
            results.update(test_idempotency(
                gateway_url, token, args, submitted))

    # Summary
    log("\n── summary ─────────────────────────────────────────────────────────")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        log(f"  {status}  {name}")
        if not ok:
            all_pass = False

    log("")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
