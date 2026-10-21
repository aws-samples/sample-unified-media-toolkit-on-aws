"""
Sibling submissions + result readers. The submit brain stays the
sibling's gateway (family rule: never StartExecution on a sibling
directly); completion is PUSH — callback_task_token resolves the calling
wait state, and the LOOP states re-enter themselves on it.

Result reading is S3-only (run bookkeeping goes to S3; a waitForTaskToken
callback payload is a signal, never a data channel) — every pointer a
reader needs was checkpointed at submit time.
"""
from __future__ import annotations

import json

from lib import ad_common as C
from lib import fitting
from lib import parse_partial_output


def bedrock_submit(records: list, task_token: str, plan: dict,
                   tag: str, use_batch: bool | None = None) -> dict:
    """Stage records under our own prefix and submit ONE bedrock run with
    push completion. Returns {records_s3uri, workflow_id,
    output_s3_prefix} for the checkpoint.

    use_batch: the call site's latency posture. True on the big one-shot
    legs (namemap/watch/writer/critic) — batch halves token cost and the
    26 h state timeouts already cover a batch job's worst case. None on
    small serial legs (repairs, reasks): bedrock-inference's dollar guard
    (§5.6) decides, and a padded 100-record batch job per tiny record
    would trade days of wall clock for sub-cent savings."""
    prefix = plan["output_prefix"]
    key = f"{prefix}/_records/{tag}.jsonl"
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                   for r in records)
    C.write_bytes(C.PROXY_BUCKET, key, body.encode("utf-8"),
                  "application/jsonl")
    payload = {
        "records_s3uri": f"s3://{C.PROXY_BUCKET}/{key}",
        "callback_task_token": task_token,
        "idempotency_token": f"{plan['workflow_id']}-{tag}",
        "model_chain": plan["model_chain"],
    }
    if use_batch is not None:
        payload["use_batch"] = use_batch
    resp = C.invoke_gateway_tool(
        C.BEDROCK_GATEWAY_ARN, "bedrock-inference___run_inference", payload)
    if resp.get("error"):
        raise RuntimeError(f"bedrock-inference refused {tag} "
                           f"({resp.get('errorCode')}): "
                           f"{resp.get('message')}")
    rec = {"records_s3uri": f"s3://{C.PROXY_BUCKET}/{key}",
           "workflow_id": resp["workflow_id"],
           "output_s3_prefix": resp["output_s3_prefix"]}
    _register(plan, "bedrock", tag, rec)
    return rec


def bedrock_outputs(submit_record: dict, expected_ids: list) -> list:
    """output_text per record_id, order preserved. A FAILED record raises
    — a repair built on a missing answer is worse than a failed run."""
    b, k = C.parse_s3_uri(submit_record["output_s3_prefix"].rstrip("/")
                          + "/response.jsonl")
    out = {}
    for line in C.read_bytes(b, k).decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "SUCCESS":
            raise RuntimeError(f"bedrock record {r.get('record_id')} "
                               f"failed: {json.dumps(r)[:300]}")
        out[r["record_id"]] = r["output_text"]
    return [out[rid] for rid in expected_ids]


def assert_tts_push_completion() -> None:
    """Refuse-before-spend probe (ValidateAndPlan calls this): tts's
    describe_tool advertises `push_completion` from alpha.20260831a — a
    deployment without it silently IGNORES callback_task_token and the
    loop states sleep to their 26h deadman. Probing the CAPABILITY here,
    before any bedrock spend, beats inferring it per-submit from a
    companion field two states later (review 2026-08-31); tts_submit's
    output_s3_prefix check below stays as the cheap per-submit
    belt-and-braces."""
    resp = C.invoke_gateway_tool(
        C.TTS_GATEWAY_ARN, "text-to-speech___describe_tool", {})
    if not resp.get("push_completion"):
        raise RuntimeError(
            "text-to-speech deployment predates callback support "
            "(describe_tool advertises no push_completion) — deploy "
            "tts-on-aws alpha.20260831a+ before running audio-description")


def tts_submit(texts: list, task_token: str, plan: dict, tag: str,
               seed: int = 1234, attempt: int = 0) -> dict:
    """Submit one synthesis batch with push completion. TTS normalization
    happens HERE (years, ALL-CAPS) so no caller can forget it.

    `attempt` widens the idempotency token: attempt 0 is the plain
    {workflow_id}-{tag}; a retry of a FAILED shard submits under -r{n}
    so it starts a FRESH run instead of absorbing into the sticky
    failure (review 2026-09-02). The attempt is recorded on the register
    row so the fit leg can tell a landed retry from a lost one."""
    # GATEWAY parameter shape, not the state machine's: transcripts is a
    # list of plain STRINGS, and flow_steps/seed are flat keys (a nested
    # "config" is silently ignored — first live run 2026-08-31: the
    # [{"text": ...}] dict shape was refused as INVALID_PARAMETER, and
    # seed/flow_steps had never been reaching the model).
    idem = f"{plan['workflow_id']}-{tag}" + \
        (f"-r{attempt}" if attempt else "")
    resp = C.invoke_gateway_tool(
        C.TTS_GATEWAY_ARN, "text-to-speech___synthesize_speech", {
            "voice_id": plan["voice_id"],
            "transcripts": [fitting.tts_normalize(t) for t in texts],
            "flow_steps": 10,
            "seed": seed,
            "callback_task_token": task_token,
            "idempotency_token": idem,
        })
    if resp.get("error") and not resp.get("workflow_id"):
        # a REFUSAL (bad parameters, no handle) — nothing was started
        raise RuntimeError(f"text-to-speech refused {tag} "
                           f"({resp.get('errorCode')}): "
                           f"{resp.get('message')}")
    # An absorbed resubmit of a FAILED run answers with the handle PLUS
    # its error — and the gateway has already persisted the verdict doc
    # and completed OUR token synchronously. Raising here would fail the
    # Map iteration whose task already resolved; register the handle and
    # let the fit leg's collect_fanout decide retry-vs-raise.
    if not resp.get("output_s3_prefix"):
        # output_s3_prefix shipped in the SAME tts version as callback
        # support, so its absence proves this deployment predates
        # callbacks — which would IGNORE our token and leave the wait
        # state sleeping toward its timeout. ValidateAndPlan probes the
        # capability itself (assert_tts_push_completion) before any
        # spend; this stays as the per-submit belt-and-braces.
        raise RuntimeError(
            "text-to-speech deployment predates callback support "
            "(no output_s3_prefix in the submit handle) — deploy "
            "tts-on-aws alpha.20260831a+ before running audio-description")
    rec = {"workflow_id": resp.get("workflow_id"),
           "job_id": resp.get("job_id"),
           "output_s3_prefix": resp.get("output_s3_prefix"),
           "count": len(texts), "attempt": attempt}
    _register(plan, "tts", tag, rec)
    return rec


def _register(plan: dict, kind: str, tag: str, rec: dict) -> None:
    """Write-through submit register, ONE ROW PER FILE
    (_plan/submits/{kind}-{tag}.json) so usage tracking re-derives from
    a finished prefix — loop checkpoints get overwritten, the register
    never does. Rows never share a document because the tts fan-out Map
    registers shards CONCURRENTLY (2026-09-02): the old single-list
    read-modify-write would lose whichever row read before another's
    write. A whole-file overwrite per (kind, tag) is also what makes a
    replay free — an absorbed-terminal submit resolves its token
    synchronously INSIDE the gateway call, so a phase can re-run and
    re-register (the old append double-counted usage, review
    2026-08-31); same key, same bytes, no accumulation."""
    C.write_json(C.PROXY_BUCKET,
                 f"{plan['output_prefix']}/_plan/submits/{kind}-{tag}.json",
                 {"kind": kind, "tag": tag, **rec})


def read_submit(prefix: str, kind: str, tag: str) -> dict:
    """One register row — a fit leg reading the shard the Map's
    iteration submitted."""
    return C.read_json(C.PROXY_BUCKET,
                       f"{prefix}/_plan/submits/{kind}-{tag}.json")


def read_submits(prefix: str) -> list:
    """Every register row, sorted by key (usage tracking's whole-run
    sweep)."""
    return [C.read_json(C.PROXY_BUCKET, k) for k in
            C.list_keys(C.PROXY_BUCKET, f"{prefix}/_plan/submits/")]


def shard_specs(base: str, order: list, seed: int = 1234) -> dict:
    """tag -> {order, seed, attempt} for every shard of one fan-out.
    ONE home for the tag/spec contract (SubmitTtsShard subscripts it and
    the register filenames embed the tag) — synth rounds and repair
    passes both build their fan-outs here."""
    return {f"{base}-s{i:02d}": {"order": shard, "seed": seed,
                                 "attempt": 0}
            for i, shard in enumerate(tts_shards(order))}


def collect_fanout(prefix: str, specs: dict) -> dict:
    """The fit/apply leg's TOLERANT sweep over a completed fan-out Map
    (review 2026-09-02: the strict read wedged every redrive when the
    fan-out itself was lost). Returns exactly one of:

      {"refan":  [tags]}  — a register row / outputs doc is missing, or
                            a bumped attempt never landed (the checkpoint
                            advanced but send_task_success or an
                            iteration was lost). Re-issue the SAME
                            fan-out: landed shards absorb at the gateway
                            and complete synchronously; lost ones submit.
      {"retry":  [tags]}  — FAILED shards with attempts left. The caller
                            bumps each spec's attempt, checkpoints, and
                            re-fans; the retry submits under a fresh
                            -r{n} idempotency token.
      {"outs": {tag: takes}} — every shard SUCCEEDED and complete.

    Raises by name when a FAILED shard has exhausted TTS_SHARD_ATTEMPTS
    — a .version bump (or a new request token) is the retry mechanism
    then, exactly like any other sticky terminal failure."""
    refan, retry, outs = [], [], {}
    for tag in sorted(specs):
        want = specs[tag].get("attempt", 0)
        row = C.read_json_if_exists(
            C.PROXY_BUCKET, f"{prefix}/_plan/submits/tts-{tag}.json")
        if row is None or row.get("attempt", 0) < want:
            refan.append(tag)
            continue
        doc = C.read_json_if_exists(
            *C.parse_s3_uri(row["output_s3_prefix"].rstrip("/")
                            + "/outputs.json"))
        if doc is None:
            refan.append(tag)
            continue
        if doc.get("status") != "SUCCEEDED" or "outputs" not in doc:
            if want + 1 < C.TTS_SHARD_ATTEMPTS:
                retry.append(tag)
                continue
            raise RuntimeError(
                f"tts shard {tag} failed on all {want + 1} attempt(s) "
                f"({doc.get('status')}): {doc.get('error')} — execution "
                f"{doc.get('execution_arn', row['output_s3_prefix'])}; "
                f"bump .version (or change the idempotency token) to "
                f"re-run")
        outs[tag] = tts_outputs(row, doc=doc)
    if refan:
        return {"refan": refan}
    if retry:
        return {"retry": retry}
    return {"outs": outs}


def tts_outputs(submit_record: dict, doc: dict | None = None) -> list:
    """One dict per take in submission order — bucket/key, tts's OWN
    duration_s (the fit is arithmetic; nothing re-measures audio), and
    qc — read from the tts run's outputs persisted at its prefix.

    outputs.json is the POLL shape, persisted at the run prefix by the
    tts callback waker (one flattener — the gateway's own parser — so
    the file and a live poll can never disagree): entries carry
    transcript_index, uri (s3://), duration_s, qc_passed, or `error`.

    COMPLETENESS IS CHECKED, not assumed (review 2026-08-31): callers
    zip these positionally against the submitted window order, so one
    silently omitted clip would shift every later pair — wrong audio
    registered for every subsequent window with nothing crashing. A
    per-clip `error` entry also raises: every AD take is load-bearing."""
    prefix = submit_record["output_s3_prefix"].rstrip("/")
    if doc is None:
        doc = C.read_json(*C.parse_s3_uri(prefix + "/outputs.json"))
    if doc.get("status") != "SUCCEEDED" or "outputs" not in doc:
        # the waker persists a FAILED/TIMED_OUT/ABORTED verdict to the
        # SAME pointer (review 2026-08-31: without it the failure lived
        # only in the callback payload — discarded by design — and this
        # reader died on a NoSuchKey naming neither tts nor the cause)
        raise RuntimeError(
            f"tts synthesis run did not succeed "
            f"({doc.get('status')}): {doc.get('error')} — "
            f"execution {doc.get('execution_arn', prefix)}")
    outs = sorted(doc["outputs"], key=lambda o: o["transcript_index"])
    expected = submit_record.get("count")
    indices = [o["transcript_index"] for o in outs]
    if (expected is not None and len(outs) != expected) \
            or indices != list(range(len(outs))):
        raise RuntimeError(
            f"tts outputs incomplete: {len(outs)} of {expected} clips, "
            f"indices {indices[:10]}... — a positional join over a gap "
            "would silently mis-assign audio to windows")
    failed = [o["transcript_index"] for o in outs if o.get("error")]
    if failed:
        raise RuntimeError(
            f"tts clips failed at indices {failed}: "
            f"{[o.get('error') for o in outs if o.get('error')][:3]}")
    result = []
    for o in outs:
        bucket, key = C.parse_s3_uri(o["uri"])
        result.append({"bucket": bucket, "key": key,
                       "duration_s": float(o["duration_s"]),
                       "qc": {"passed": o.get("qc_passed"),
                              "similarity": o.get("qc_similarity")}})
    return result


def tts_shards(order: list) -> list:
    """Shards of <= TTS_SHARD_SIZE window ids — tts-on-aws refuses >100
    transcripts per batch (the runner sharded at <=90; dropping the
    shard was the 2026-08-31 review's stopper: a typical episode's ~120
    cues failed the whole run at synthesis). Shards are INDEPENDENT and
    run in parallel through the fan-out Map (2026-09-02)."""
    n = C.TTS_SHARD_SIZE
    return [order[i:i + n] for i in range(0, len(order), n)]


def prune_cues(cues: dict, valid_ids: set) -> tuple:
    """Drop hallucinated window_ids (review 2026-08-31: one model line
    with an id outside windows.json crashed the run downstream with
    KeyError — after the tts spend). Returns (kept, extras)."""
    extras = sorted(k for k in cues if k not in valid_ids)
    return {k: v for k, v in cues.items() if k in valid_ids}, extras


def register_takes(prefix: str, updates: dict) -> dict:
    """takes.json: window_id -> {wav (s3 uri), duration_s} — the raw-take
    register every later refit/revive reads (write-through, like drops).
    The wav stays IN the tts run's prefix; recording the pointer beats
    copying megabytes we never open."""
    key = f"{prefix}/takes.json"
    takes = C.read_json_if_exists(C.PROXY_BUCKET, key) or {}
    takes.update({str(k): v for k, v in updates.items()})
    C.write_json(C.PROXY_BUCKET, key, takes)
    return takes


def parse_jsonl_cues(text: str) -> tuple:
    """window_id -> text from a model's JSONL answer; salvage-parsed
    (parse_partial_output), bad lines returned for the log.

    EVERY routine model deviation lands in `bad`, never raises (review
    2026-08-31): a null/non-numeric window_id, a JSON array on one line
    (parse_jsonl_line's strict rung has no dict check), and "text":
    null (key present, so .get's default never applies) each used to
    crash the whole parse — defeating the salvage design where a bad
    line is a normal per-record outcome."""
    cues, bad = {}, []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith(("```", "//")):
            continue
        obj, _tier = parse_partial_output.parse_jsonl_line(
            line, ("window_id",))
        if not isinstance(obj, dict):
            bad.append(line[:80])
            continue
        try:
            wid = int(obj["window_id"])
        except (KeyError, TypeError, ValueError):
            bad.append(line[:80])
            continue
        text_val = obj.get("text")
        cues[wid] = text_val if isinstance(text_val, str) else ""
    return cues, bad
