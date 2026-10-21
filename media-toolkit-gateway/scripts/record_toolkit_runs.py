#!/usr/bin/env python3
"""
Record GENUINE gateway request/response pairs for the toolkit-explainer page
(docs/toolkit-explainer/), by driving the full Meridian chain through the
deployed AgentCore Gateway and capturing every MCP call verbatim.

Scenario recorded (one chained session — the same story the page's graph
tells, so the s3 uris visibly thread from one tool's response into the next
tool's request):

  phase 0   describe_tool + catalog reads on every target   (instant, free)
  phase 1   media-processing + speech-diarization + shot-segmentation +
            audio-stem, all on Meridian, submitted together; then ONE
            byte-identical duplicate submit of shot-segmentation to capture
            the absorbed-resubmit handle; poll all four to SUCCEEDED
  phase 2   face-recognition (joined to shot-segmentation's run prefix) and
            conversation-analysis (diarization's final_transcript.json)
  phase 3   shot-metadata-enrichment (all three upstream prefixes, sync)
  phase 4   hybrid-search: create_collection, index_content built from
            enrichment's real shot_metadata.jsonl, then search_content
  phase 5   text-to-speech on real transcript cues (when a voice exists),
            media-outputs presigning one artifact

Captures land in /tmp/toolkit_captures/captures.jsonl — RAW, containing the
account id, bucket names and ARNs. They are sanitized by a separate pass
before anything is committed; this file NEVER writes inside the repo.

Resumable by design: every submit carries the same idempotency_token, so a
re-run's submits are absorbed into the existing executions and polling simply
continues. Asset: Netflix Open Content "Meridian" (safe for customer-facing
docs by policy — see the family GOTCHAS convention).

Usage:
    python3 record_toolkit_runs.py           # runs the whole session
Environment: AWS_REGION (default eu-west-1), credentials for the dev account.
"""

import json
import os
import sys
import time
import urllib.request

import boto3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_gateway import fetch_credentials, get_token, mcp_request  # noqa: E402


def call_tool(gateway_url: str, token: str, tool_name: str, arguments: dict) -> dict:
    """test_gateway.call_tool, but tolerant of non-JSON text replies — some
    admin tools answer with a plain string, which must be CAPTURED, not crash
    the session."""
    resp = mcp_request(gateway_url, token, "tools/call", {
        "name": tool_name, "arguments": arguments,
    })
    if "error" in resp:
        raise RuntimeError(f"tools/call error: {resp['error']}")
    content = resp.get("result", {}).get("content", [])
    if content and content[0].get("type") == "text":
        text = content[0]["text"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"_raw_text": text}
    return resp.get("result", {})

REGION = os.environ.get("AWS_REGION", "eu-west-1")
SECRET_NAME = "/media-toolkit-gateway/dev/gateway-credentials"  # nosec B105
CAPTURE_DIR = "/tmp/toolkit_captures"
CAPTURE_FILE = os.path.join(CAPTURE_DIR, "captures.jsonl")

ASSET_S3URI = os.environ["TOOLKIT_ASSET_S3URI"]  # demo source asset
IDEMPOTENCY_TOKEN = "toolkit-explainer-20260824"  # nosec B105
SEARCH_COLLECTION = "toolkit-demo"

POLL_INTERVAL_SEC = 20
POLL_TIMEOUT_SEC = 5400

DESCRIBE_TARGETS = [
    "media-processing", "speech-diarization", "shot-segmentation",
    "audio-stem", "face-recognition", "conversation-analysis",
    "shot-metadata-enrichment", "hybrid-search", "text-to-speech",
    "bedrock-inference", "prompt-registry",
]

STATUS_TOOL = {
    "media-processing":         "media-processing___get_job_status",
    "speech-diarization":       "speech-diarization___get_job_status",
    "shot-segmentation":        "shot-segmentation___get_job_status",
    "audio-stem":               "audio-stem___get_job_status",
    "face-recognition":         "face-recognition___get_job_status",
    "conversation-analysis":    "conversation-analysis___get_job_status",
    "shot-metadata-enrichment": "shot-metadata-enrichment___get_job_status",
    "hybrid-search":            "hybrid-search___get_job_status",
    "text-to-speech":           "text-to-speech___get_job_status",
}

TERMINAL = {"SUCCEEDED", "FAILED", "TIMED_OUT", "ABORTED"}


class Recorder:
    """call_tool with a verbatim request/response capture layer."""

    def __init__(self, gateway_url: str, token_factory):
        self.gateway_url = gateway_url
        self._token_factory = token_factory
        self._token = token_factory()
        self._token_born = time.time()
        self.seq = 0
        os.makedirs(CAPTURE_DIR, exist_ok=True)

    def _fresh_token(self) -> str:
        # Cognito M2M tokens live 1h; a chained session outlives that.
        if time.time() - self._token_born > 2700:
            self._token = self._token_factory()
            self._token_born = time.time()
        return self._token

    def call(self, tool: str, args: dict, phase: str, note: str = "",
             capture: bool = True) -> dict:
        t0 = time.time()
        error = None
        try:
            result = call_tool(self.gateway_url, self._fresh_token(), tool, args)
        except RuntimeError as exc:
            result = None
            error = str(exc)
        elapsed_ms = int((time.time() - t0) * 1000)
        if capture:
            self.seq += 1
            with open(CAPTURE_FILE, "a") as f:
                f.write(json.dumps({
                    "seq": self.seq, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "phase": phase, "note": note, "tool": tool,
                    "request": args, "response": result, "error": error,
                    "elapsed_ms": elapsed_ms,
                }) + "\n")
        if error:
            print(f"    !! {tool}: {error[:200]}")
        return result or {}

    def poll(self, target: str, job_id: str, phase: str) -> dict:
        """Capture the FIRST status reply (the RUNNING shape) and the terminal
        one; poll silently in between."""
        tool = STATUS_TOOL[target]
        first = self.call(tool, {"job_id": job_id}, phase,
                          note=f"{target}: first status poll")
        status = first.get("status")
        deadline = time.time() + POLL_TIMEOUT_SEC
        last = first
        while status not in TERMINAL and time.time() < deadline:
            time.sleep(POLL_INTERVAL_SEC)
            last = self.call(tool, {"job_id": job_id}, phase, capture=False)
            new_status = last.get("status")
            if new_status != status:
                print(f"    {target}: {status} -> {new_status}"
                      f" ({last.get('current_state', '')})")
                status = new_status
        # terminal (or timed-out-waiting) reply, captured verbatim
        final = self.call(tool, {"job_id": job_id}, phase,
                          note=f"{target}: terminal status ({status})")
        print(f"    {target}: terminal {final.get('status')}")
        return final


def output_uri(status: dict, substring: str) -> str | None:
    for entry in status.get("outputs", []):
        if substring in entry.get("uri", ""):
            return entry["uri"]
    return None


def s3_read_json(s3uri: str):
    bucket, key = s3uri[5:].split("/", 1)
    body = boto3.client("s3", region_name=REGION).get_object(
        Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    if s3uri.endswith(".jsonl"):
        return [json.loads(l) for l in body.splitlines() if l.strip()]
    return json.loads(body)


def main() -> None:
    creds = fetch_credentials(SECRET_NAME, REGION)
    rec = Recorder(
        creds["endpointUrl"],
        lambda: get_token(creds["tokenEndpoint"], creds["clientId"],
                          creds["clientSecret"], creds["scope"]),
    )
    print(f"Recording to {CAPTURE_FILE}")

    # ── phase 0: self-description + catalog reads ─────────────────────────
    print("phase 0: describe_tool + catalogs")
    for target in DESCRIBE_TARGETS:
        rec.call(f"{target}___describe_tool", {}, "0-describe",
                 note=f"{target} self-description")
    rec.call("bedrock-inference___list_models", {}, "0-describe")
    rec.call("prompt-registry___list_prompts", {}, "0-describe")
    rec.call("face-recognition___list_collections", {}, "0-describe")
    rec.call("hybrid-search___list_collections", {}, "0-describe")
    voices = rec.call("text-to-speech___list_voice_collection", {"limit": 5},
                      "0-describe", note="voice catalog")

    # ── phase 1: the four asset-direct submits, in parallel ───────────────
    print("phase 1: submit media-processing, diarization, shot-seg, audio-stem")
    submits = {
        "media-processing": ("media-processing___process_media",
                             {"asset_s3uri": ASSET_S3URI,
                              "idempotency_token": IDEMPOTENCY_TOKEN}),
        "speech-diarization": ("speech-diarization___diarize_speech",
                               {"asset_s3uri": ASSET_S3URI,
                                "idempotency_token": IDEMPOTENCY_TOKEN}),
        "shot-segmentation": ("shot-segmentation___segment_shots",
                              {"asset_s3uri": ASSET_S3URI,
                               "idempotency_token": IDEMPOTENCY_TOKEN}),
        "audio-stem": ("audio-stem___separate_audio_stems",
                       {"asset_s3uri": ASSET_S3URI, "music_stem": True,
                        "sfx_stem": True,
                        "idempotency_token": IDEMPOTENCY_TOKEN}),
    }
    handles = {}
    for target, (tool, args) in submits.items():
        handles[target] = rec.call(tool, args, "1-submit", note=f"{target} submit")
        print(f"    {target}: job {handles[target].get('job_id', '?')[-24:]}"
              f" {handles[target].get('status', '?')}")

    finals = {}
    for target in submits:
        job_id = handles[target].get("job_id")
        if job_id:
            finals[target] = rec.poll(target, job_id, "1-poll")

    # the idempotency demo — AFTER completion, where StartExecution raises
    # ExecutionAlreadyExists and the gateway answers with the absorbed handle
    # (`message` present, existing run's status, no results). A duplicate
    # while the run is still RUNNING is silently the same execution and
    # carries no message, so the completed case is the one worth showing.
    tool, args = submits["shot-segmentation"]
    dup = rec.call(tool, args, "1-submit",
                   note="shot-segmentation DUPLICATE submit — absorbed, no second run")
    print(f"    duplicate absorbed: message={dup.get('message', '(none)')[:80]}")

    shotseg_prefix = finals.get("shot-segmentation", {}).get("output_s3_prefix")
    shots_json_uri = output_uri(finals.get("shot-segmentation", {}), "shots.json")
    transcript_uri = output_uri(finals.get("speech-diarization", {}),
                                "final_transcript.json")
    print(f"    shot-seg prefix: {shotseg_prefix}")
    print(f"    shots.json:      {shots_json_uri}")
    print(f"    transcript:      {transcript_uri}")

    # ── phase 2: the first chaining hop ────────────────────────────────────
    print("phase 2: face-recognition + conversation-analysis")
    fr = rec.call("face-recognition___recognize_faces",
                  {"asset_s3uri": ASSET_S3URI,
                   "shot_segmentation_s3uri": shots_json_uri,
                   "idempotency_token": IDEMPOTENCY_TOKEN},
                  "2-submit", note="face-recognition joined to shot-seg run")
    ca = rec.call("conversation-analysis___analyze_conversation",
                  {"transcript_s3uri": transcript_uri,
                   "history_scope": "meridian",
                   "idempotency_token": IDEMPOTENCY_TOKEN},
                  "2-submit", note="conversation-analysis on the diarized transcript")
    fr_final = rec.poll("face-recognition", fr["job_id"], "2-poll") if fr.get("job_id") else {}
    ca_final = rec.poll("conversation-analysis", ca["job_id"], "2-poll") if ca.get("job_id") else {}
    facerec_prefix = fr_final.get("output_s3_prefix")

    # ── phase 3: enrichment consumes all three upstream runs ──────────────
    print("phase 3: shot-metadata-enrichment (sync)")
    sme_args = {
        "shot_segmentation_s3uri": shotseg_prefix,
        "face_recognition_s3uri": facerec_prefix,
        "transcript_s3uri": transcript_uri,
        "program_names": "Meridian",
        "overall_content_type": "Movies/Feature Films",
        "history_scope": "meridian",
        "use_batch": False,
        "idempotency_token": IDEMPOTENCY_TOKEN,
    }
    sme = rec.call("shot-metadata-enrichment___enrich_shot_metadata",
                   sme_args, "3-submit",
                   note="enrichment consuming three upstream run prefixes")
    sme_final = rec.poll("shot-metadata-enrichment", sme["job_id"], "3-poll") \
        if sme.get("job_id") else {}
    shots_jsonl_uri = output_uri(sme_final, "shot_metadata.jsonl")

    # ── phase 4: index the real enrichment output, then search it ─────────
    print("phase 4: hybrid-search index + search")
    rec.call("hybrid-search___create_collection",
             {"collection_id": SEARCH_COLLECTION,
              "active_scopes": ["scene"],
              "name": "Toolkit explainer demo"},
             "4-search",
             note="create demo collection (an existing one is a benign error)")
    if shots_jsonl_uri:
        shots = s3_read_json(shots_jsonl_uri)
        # enrichment docs deliberately carry NO times (times never pass
        # through a model) — the timestamps join from shot-segmentation's
        # own shots.json by shot_id, exactly as a consumer is meant to
        times = {s["shot_id"]: (s["start_time"], s["end_time"])
                 for s in s3_read_json(shots_json_uri).get("shots", [])}
        docs = []
        for shot in shots:
            desc = shot.get("description") or ""
            if not desc:
                continue
            shot_id = int(shot.get("shot_id", len(docs)))
            start_s, end_s = times.get(shot_id, (0, 0))
            docs.append({
                "asset_id": "meridian",
                "doc_type": "scene",
                "start_ms": int(1000 * float(start_s)),
                "end_ms": int(1000 * float(end_s)),
                "chunk_idx": shot_id,
                # replace this asset's rows from the earlier zero-ms pass
                "refresh": True,
                # scene schema: 'vector' is the ANN field the backend embeds,
                # 'text' is the FTS field (entity extraction rides on it)
                "content": [
                    {"kind": "text", "field": "vector", "value": desc},
                    {"kind": "text", "field": "text", "value": desc,
                     "extract_entities": True},
                ],
            })
        docs = docs[:100]
        idx = rec.call("hybrid-search___index_content",
                       {"collection_id": SEARCH_COLLECTION,
                        "documents": docs,
                        "asset_metadata": {"title": "Meridian",
                                           "genres": ["drama", "noir"],
                                           "content_category": "Movies/Feature Films"}},
                       "4-search",
                       note=f"index {len(docs)} real enrichment docs")
        if idx.get("job_id"):
            rec.poll("hybrid-search", idx["job_id"], "4-search")
        for query in ["a man alone looking out at the ocean from the rocks",
                      "detective interviewing someone in an office"]:
            hits = rec.call("hybrid-search___search_content",
                            {"collection_id": SEARCH_COLLECTION,
                             "search_term": query, "k": 3},
                            "4-search", note=f"natural-language search: {query!r}")
            print(f"    search {query!r}: {len(hits.get('results', []))} hits")

    # ── phase 5: tts on real cues + presign one artifact ──────────────────
    print("phase 5: tts + media-outputs")
    voice_list = voices.get("voices") or []
    en_voices = [v for v in voice_list if v.get("language") == "en"] or voice_list
    if en_voices and transcript_uri:
        transcript = s3_read_json(transcript_uri)
        cues = [s.get("text", "") for s in transcript.get("sentences", [])]
        cues = [c for c in cues if len(c.split()) >= 5][:3]
        if cues:
            voice_id = en_voices[0]["voice_id"]
            tts = rec.call("text-to-speech___synthesize_speech",
                           {"voice_id": voice_id, "transcripts": cues,
                            "seed": 1234, "flow_steps": 6,
                            "idempotency_token": IDEMPOTENCY_TOKEN},
                           "5-tts", note="re-voice three real Meridian cues")
            if tts.get("job_id"):
                rec.poll("text-to-speech", tts["job_id"], "5-tts")
    else:
        print("    (no registered voice or no transcript — tts synth skipped)")

    if shots_jsonl_uri:
        rec.call("media-outputs___get_download_url",
                 {"s3uri_or_key": shots_jsonl_uri, "expiry_seconds": 300},
                 "5-outputs", note="presign the enrichment product")

    print(f"\nDONE — {rec.seq} captures in {CAPTURE_FILE}")


if __name__ == "__main__":
    main()
