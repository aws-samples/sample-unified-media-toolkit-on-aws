---
name: upload-media
description: "Upload one or more LOCAL media files into the toolkit's ingest bucket via the media-uploads presigned-upload tools — single PUT for small files, parallel multipart for large (max 5 GiB per file), auto MIME, verify by listing, then chain analysis. Use when the user asks to upload, ingest, push, or add a local file to the ingest bucket / the toolkit. Requires no AWS credentials — bytes go client-to-S3 via presigned URLs."
metadata:
  version: 1.0.0
compatibility: Requires the media-toolkit MCP server with the media-uploads target attached (IngestBucket configured on the stack), local file access, and python3 for the parallel uploader.
---

# Upload media (local files → ingest bucket)

Move local files into the ingest bucket so the workflow tools can see
them. The tool contract, size cap, MIME rules, and 403-refresh semantics
live in
[../video-processing/references/media_uploads_agent.md](../video-processing/references/media_uploads_agent.md)
(read it first). This file carries the composition and the data-plane
helper: [references/uploader.py](references/uploader.py) — stdlib-only
parallel PUTs, no AWS credentials.

## Input

`$ARGUMENTS` is one or more local file paths, optionally with a
destination (`upload /path/promo.mov to uploads/promo.mov`). Without a
destination, key = `uploads/{basename}` — say so. Multiple files = one
`start_upload` each (the tool is deliberately one-file-per-call); upload
them through one worker pool, sequential `start_upload`s.

## Steps (per file)

1. **Stat the file** — exact `size_bytes` is required. Over 5 GiB:
   refuse up front (the cap is hard) and tell the user; do not try.
2. `media-uploads___start_upload(key, size_bytes)` — let MIME
   auto-derive unless the user named a type. On `KEY_EXISTS`, ASK before
   passing `overwrite: true` (it replaces source media); in a
   NON-INTERACTIVE run take the reference's safe branch instead of
   stalling — a distinct key (suffix BEFORE the extension) and a
   prominent report of the deviation.
3. Write the response into a plan file and run the helper. **Put the plan
   OUTSIDE the repo** (e.g. `/tmp/upload-<name>/plan.json`): it embeds
   the presigned URLs, which carry a live session credential in their
   query string — a write grant that must never land in a working tree. Create that
   directory `mkdir -m 700` (default /tmp permissions leave the token
   readable to every local user for the URL lifetime) and delete the
   plan file once the upload is verified. **Copy the
   response VERBATIM under `response` and reshape nothing**: each signed
   URL is ~1.8 KB, one mangled character yields an opaque 403
   indistinguishable from early expiry, and every re-typing into a
   different structure is another chance to mangle. The helper derives
   mode, urls, part_size and content_type from it:
   ```
   {"file": <abs local path>,
    "response": { ...the start_upload result, exactly as returned... },
    "workers": 6}   ← optional (6 is the helper's default); multipart only
   python3 <abs path to>/references/uploader.py <abs path to>/plan.json
   ```
   (Explicit per-mode shapes still work INSTEAD of `response` — never
   both: a per-mode key — `mode`/`content_type`/`url`/`part_size`/
   `part_count` — beside a `response` block is refused by name; top-level `parts` (the
   resume patch), `done`, `size_bytes` and `workers` always stay
   top-level — see the helper's docstring. `content_type` belongs only
   to `single`; parts deliberately carry no Content-Type.)
   - `mode: "single"` → the helper does ONE PUT with the returned
     Content-Type header. Done — skip to step 6. If it fails
     (`errors.single`, e.g. an early-expired URL's HTTP 403): there is
     no refresh tool for single mode — re-run from step 1 (RE-STAT: the
     URL is signed for the declared length, so a file that changed size
     is the other way a single PUT 403s) for a fresh URL (the object
     never landed, so no `overwrite` needed) and retry; no session
     exists, nothing to abort. A NON-HTTP `errors.single` (local I/O —
     unreadable file, a directory, disk error) twice in a row: stop and
     report the condition — a fresh URL cannot fix a local error. An `errors.plan` line
     (either mode) means the helper refused BEFORE any PUT — nothing was
     sent, no URL consumed. Read the message: a malformed plan (missing
     `content_type`, a mangled part set) is fixed and the helper re-run;
     but a refusal saying the FILE changed is not a plan edit — do what
     it says: `abort_upload` the session (multipart), then restart from
     step 1 with a re-stat.
   - `mode: "multipart"` → the helper PUTs all parts in parallel and
     prints the collected ETags.
4. **On failed parts** (`ok: false`): HTTP 403 means expired URLs —
   `get_upload_part_urls(key, upload_id, size_bytes, part_size,
   part_numbers=failed_parts)` (≤50 per call; size_bytes is the value you
   SENT to start_upload, while part_size must be the one its RESPONSE
   returned — an input part_size can be clamped, so sent and returned
   differ exactly where it changes the byte ranges). Paste the refresh
   result's `parts` list at the TOP LEVEL of the plan — on a
   response-shaped plan the helper merges them into `response.parts` by
   part_number, so never edit inside the `response` block — and copy the
   previous run's `etags` list under `"done"` verbatim (the failure JSON
   carries `etags` for the parts that DID land — same shape both
   directions, and keeping each row's `local_md5` is what lets the
   helper re-hash those ranges and refuse by name if the file was
   rewritten at the same length between runs), then re-run the helper.
   **A repeated 403 is usually STILL expiry, not a dead end** — a
   refreshed URL expires too (`expires_at` caps at 3600 s), so on a slow
   link parts queued behind the workers can outlive their refresh and
   403 again legitimately: compare the refresh response's `expires_at`
   with the clock, and keep refreshing as long as parts are landing. The
   terminal case is a part that 403s IMMEDIATELY on a just-issued URL
   (seconds old, nowhere near `expires_at`), twice in a row: with the
   plan guards passing, the signed length no longer matches reality —
   stop, `abort_upload`, restart from step 1 (re-stat, and a NEW plan:
   `done` rows belong to the aborted session — never carry them into a
   fresh `start_upload`). An expired URL on a LARGE part can also
   surface as a connection error (`URLError(BrokenPipeError…)`,
   `ConnectionResetError`) — S3 rejects the PUT and closes the socket
   while the client is still streaming — and the helper's 60 s socket
   timeout surfaces a stall as `TimeoutError('timed out')`: treat
   connection errors AND timeouts on part PUTs like 403s — refresh and
   retry first. Only errors that are none of those, twice in a row:
   stop,
   `abort_upload`, and report the condition — never leave a session
   dangling (incomplete parts bill until aborted).
5. For multipart: `complete_upload(key, upload_id, parts=...)` — pass
   the helper's success-output `parts` list VERBATIM (it is emitted
   paste-ready with exactly the `part_number`/`etag` fields the tool
   schema declares; never rebuild it from `etags` by hand). The object
   exists only now.
6. **Verify**: `media-ingest___list_files(prefix=key)` — `prefix` is
   not an exact match, so confirm the row whose `key` equals yours
   (check `count`) — and compare `size` to the local size; report the
   `s3uri`. Never read `last_modified` as evidence the upload landed —
   on a multipart object it is NOT a completion time (the reference's
   "Verify, then chain" rule has the measured trap).

   Size proves LENGTH, not content. For integrity (the user asked for
   "intact", or the transfer path is untrusted), the helper has already
   done the work: it hashes every byte in the same read stream and
   emits `local_md5` beside each part's `etag` (single mode: top-level
   `etag` + `local_md5`). **`etag == local_md5` on every row is the
   whole check** — per-part matches compose to whole-object integrity,
   so no composite-ETag math is needed for a file this run uploaded.
   Rows resumed via `done` re-emit a `local_md5` only when the helper
   re-hash-verified them this run (rows fed back without one are
   trusted etag-only); check unverified rows against the run that
   uploaded them. The match/mismatch
   asymmetry (what a match does and does not prove; why only a MISMATCH
   is ambiguous — corruption *or* SSE-KMS/SSE-C, a case for the
   operator) and why an object you did not just upload cannot be
   verified at all live in the reference's "Integrity for free" rule.

## Ending

Report per file: `s3uri`, size, content type, mode (single/multipart ×
part count), wall time. Offer — don't auto-run — the natural next step:
`analyze-video {ASSET}` (or a single workflow via the video-processing
router) on the new `s3uri`. If any multipart session was left incomplete
after an abort failure, say so explicitly and name the upload_id — the
bucket's lifecycle rule is the only remaining cleanup.
