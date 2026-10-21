"""Data-plane half of the upload-media skill: parallel HTTP PUTs of a
local file to presigned S3 URLs. NO AWS credentials, NO gateway calls —
the agent does the control plane (start_upload / get_upload_part_urls /
complete_upload) and hands this script a plan file.

Usage:
  python3 uploader.py <plan.json>

plan.json — PREFERRED shape: paste the start_upload response verbatim
under "response" and add nothing but the local path. One copy, no
per-field reshaping, nothing retyped into a different structure:
  {"file": "/path/to/local.mov",
   "response": { ...the start_upload result, exactly as returned... },
   "workers": 6}                                          # optional

On a response-shaped plan, top-level "parts" are PATCHES: fresh urls
from get_upload_part_urls, replacing their part_numbers inside
response.parts — paste the refresh result's "parts" list (or the whole
refresh result; its parts list is unwrapped) at top level and the merge
is done for you. Any OTHER per-mode key (mode, content_type,
url, part_size) beside a "response" block is refused by name: two
copies of one fact have no deterministic winner. Explicit shapes
(no "response" block) are still accepted; explicit multipart REQUIRES
size_bytes — it is that shape's mutation guard:
  single:    {"file": "/path/to/local.mov", "mode": "single",
              "url": "...", "content_type": "video/quicktime"}
  multipart: {"file": "/path/to/local.mov", "mode": "multipart",
              "part_size": 134217728, "size_bytes": 2147483648,
              "parts": [{"part_number": 1, "url": "..."}, ...],
              "workers": 6,
              "done": [{"part_number": 1, "etag": "..."}, ...]}  # optional

ALWAYS prints exactly one json line to stdout — error paths included.
BOTH shapes carry "etags" (the parts that DID succeed); that list is
what you feed back as "done" to resume:
  {"ok": true,  "etags": [{"part_number": n, "etag": "...",
                           "local_md5": "..."}, ...],
   "parts": [{"part_number": n, "etag": "..."}, ...]}   # multipart only:
                       # paste-ready for complete_upload(parts=...)
  {"ok": false, "etags": [...succeeded so far...],
   "failed_parts": [n, ...], "errors": {"n": "HTTP 403"}}
  (single mode: {"ok": true, "etags": [], "etag": "...",
                 "local_md5": "...", "size_bytes": n})

"local_md5" is the MD5 of the bytes this run read, hashed in the same
stream (no second pass): etag == local_md5 on every row is the
integrity check, no composite-ETag math needed. Rows resumed via "done"
re-emit a local_md5 only when it was re-hash-verified this run; rows
fed back without one are trusted and re-emitted etag-only.

"done" takes the SAME list shape "etags" is emitted in: after a partial
failure, feed the previous run's etags back as done (verbatim — keeping
local_md5 is what arms the rewrite check below) and supply fresh URLs
(from get_upload_part_urls) only for failed_parts — parts ∪ done must
still cover 1..part_count, and already-done parts are skipped. A done
list belongs to ONE upload_id — after an abort + fresh start_upload,
start a NEW plan without it (nothing from the old session landed). Done
rows carrying a local_md5 are RE-HASHED from disk (in parallel, still
strictly before any PUT): a file rewritten at the same length since
the previous run (which no size check can see) is refused by name
instead of welding old-bytes and new-bytes parts into a silently
corrupt object. Verified rows re-emit their local_md5, so the guard
stays armed on every later resume; done rows without one are trusted
and re-emitted etag-only.
HTTP 403 usually means a URL expired early (signing-credential
lifetime); it is retriable with fresh URLs — a refreshed URL can expire
too while parts queue behind the workers, so a REPEATED 403 is usually
still expiry (the skill's step 4 carries the discriminator). The
helper RE-STATS the file and uses that size for all byte-range math,
refusing when the plan pins a size the file no longer matches: exact
when size_bytes is known (the response's echo on newer deployments, or
the plan's own copy — which must be the value SENT to start_upload; a
re-stat after the file changed defeats the guard), the part_size x
part_count bucket otherwise. A
single-mode plan without size_bytes carries no pin, so there a mutated
file surfaces instead as the signed-length HTTP 403. Memory stays flat: parts stream from disk in
8 MiB chunks with an explicit Content-Length (which the presigned
signature pins). stdlib only.
"""
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

_CHUNK = 8 * 1024 * 1024
# per-socket-operation idle timeout (connect / stalled send / recv), NOT a
# per-part total: a healthy slow transfer keeps making ops, a wedged socket
# trips this instead of hanging the one-json-line contract forever
_TIMEOUT = 60


def _md5():
    # a content check against S3 ETags, not a security use — the flag is
    # what keeps FIPS-enforcing Python builds from refusing every run
    try:
        return hashlib.md5(usedforsecurity=False)
    except TypeError:  # Python < 3.9
        return hashlib.md5()  # nosec B324


def _pos_int(value, name):
    """(value, None) as a positive int, else (None, a named refusal)."""
    try:
        v = _as_int(value)
    except (TypeError, ValueError):
        v = 0
    if v <= 0:
        return None, f"{name} must be a positive integer"
    return v, None


def _as_int(value):
    """int() that refuses what plain int() silently accepts: bools and
    non-integral floats — 300.5 must refuse as a slip, never floor to a
    size the guards then compare in good faith."""
    if isinstance(value, bool):
        raise ValueError(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(value)
    return int(value)


class _BoundedReader:
    """File-like over [offset, offset+length) — lets urllib stream the
    part instead of holding workers x part_size bytes in RAM. Reads are
    capped at _CHUNK even for read(-1): an unsized read must never
    materialize a whole part (that would be the exact blow-up this
    class exists to prevent)."""

    def __init__(self, path, offset, length):
        self._f = open(path, "rb")
        self._f.seek(offset)
        self._left = length
        # hashed in the same read stream — integrity costs no second pass
        # (MD5 because S3 ETags are MD5s)
        self.md5 = _md5()

    def read(self, n=-1):
        if self._left <= 0:
            return b""
        n = _CHUNK if n is None or n < 0 else min(n, _CHUNK)
        data = self._f.read(min(n, self._left))
        self._left -= len(data)
        self.md5.update(data)
        return data

    def close(self):
        self._f.close()


def _put(url, body, length, content_type=None):
    try:
        req = urllib.request.Request(url, data=body, method="PUT")
        # explicit Content-Length: S3 rejects chunked transfer on presigned
        # PUTs, and the URLs are SIGNED for this exact length
        req.add_header("Content-Length", str(length))
        if content_type:
            req.add_header("Content-Type", content_type)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # nosec B310 — presigned https
            return (resp.headers.get("ETag") or "").strip('"')
    finally:
        body.close()    # deterministic fd release on every path —
        # including a mangled url that fails Request() itself


def _part_map(rows, what, need_url=False):
    """Normalize part_numbers to int (in place) and index the rows by
    them. Returns (map, error): a duplicate, a missing url, or a
    malformed row is a PLAN slip, named as such here rather than
    surfacing downstream as a transfer error (or an errors.fatal) the
    skill would mis-route."""
    if not isinstance(rows, list):
        return None, f"\"{what}\" must be a list of part rows"
    out = {}
    for r in rows:
        try:
            n = _as_int(r["part_number"])
        except (KeyError, TypeError, ValueError):
            return None, (f"malformed {what} entry — each row needs an "
                          f"integer part_number")
        if n in out:
            return None, (f"duplicate {what} entry for part {n} — keep "
                          f"one row per part")
        if need_url and not r.get("url"):
            return None, f"{what} entry for part {n} carries no url"
        r["part_number"] = n
        out[n] = r
    return out, None


def _rows(etags, md5s):
    # a local_md5 means this run read those bytes: uploaded parts, plus
    # "done" rows whose md5 was re-hash-verified; done rows that arrived
    # WITHOUT one were not re-read and re-emit etag-only
    return [{"part_number": n, "etag": etags[n],
             **({"local_md5": md5s[n]} if n in md5s else {})}
            for n in sorted(etags)]


def _fail(errors, failed, etag_rows=None):
    print(json.dumps({"ok": False, "etags": etag_rows or [],
                      "failed_parts": sorted(failed), "errors": errors}))
    return 1


def _resolve_shape(plan):
    """Fold a "response" block into the plan, in place. Returns an error
    string (for errors.plan) or None. Two shapes, never mixed: a response
    block is authoritative for mode/content_type/url/part_size/parts (and
    its part_count metadata); beside it, "parts" is the resume PATCH —
    fresh urls (from get_upload_part_urls) replacing their part_numbers
    inside response.parts, so the natural verbatim paste at top level is
    the correct resume edit — and "done"/"size_bytes"/"workers" stay
    top-level. Any OTHER per-mode key beside a response is ambiguous
    (which copy wins?) and refused rather than silently resolved."""
    resp = plan.get("response") or {}
    if isinstance(resp, str):
        # MCP results often arrive as JSON text; pasting that text under
        # "response" is a verbatim paste doing exactly what was asked —
        # parse it rather than refuse it
        try:
            resp = json.loads(resp)
        except ValueError:
            return ("\"response\" is a string that is not valid JSON — "
                    "paste the start_upload result itself")
    if not isinstance(resp, dict):
        return "\"response\" must be the start_upload result object"
    if not resp:
        # an empty/null response block is no response at all — drop the
        # key so the explicit shape's requirements (size_bytes!) apply
        # instead of being half-waived by a hollow marker
        plan.pop("response", None)
        return None
    clash = [k for k in ("mode", "content_type", "url", "part_size",
                         "part_count") if k in plan]
    if clash:
        return (f"plan carries both a \"response\" block and top-level "
                f"{', '.join(clash)} — keep exactly one shape (beside a "
                f"response, the only per-mode key read is \"parts\", the "
                f"resume patch; \"done\", \"size_bytes\" and \"workers\" "
                f"stay top-level)")
    patch = None
    if "parts" in plan:
        raw = plan.pop("parts")
        if isinstance(raw, str):
            # same tolerance as "response": MCP results arrive as JSON text
            try:
                raw = json.loads(raw)
            except ValueError:
                return ("the top-level \"parts\" patch is a string that is "
                        "not valid JSON — paste the refresh result itself")
        if isinstance(raw, dict) and isinstance(raw.get("parts"), list):
            raw = raw["parts"]      # the whole refresh result was pasted —
            # its parts list is the patch, verbatim either way
        patch, err = _part_map(raw, "patch", need_url=True)
        if err:
            return err
    for k in ("mode", "content_type", "url", "part_size", "parts",
              "part_count"):
        if k in resp:
            plan[k] = resp[k]
    # newer deployments echo size_bytes in the response, which upgrades
    # the mutation guard from bucket-granular to exact; a top-level copy
    # that disagrees with the echo is two copies of one fact — refused
    # by name, same rule as the clash keys above
    try:
        echo = _as_int(resp["size_bytes"]) if "size_bytes" in resp else None
    except (TypeError, ValueError):
        # a malformed echo is a RESPONSE defect the agent cannot fix —
        # treat it as absent; the bucket guard still covers this case
        echo = None
    if echo is not None:
        if "size_bytes" in plan:
            try:
                agree = _as_int(plan["size_bytes"]) == echo
            except (TypeError, ValueError):
                return "size_bytes must be an integer (bytes)"
            if not agree:
                return (f"plan size_bytes {plan['size_bytes']} disagrees "
                        f"with the response's echo {echo} — the URLs were "
                        f"signed for the response's value; drop the "
                        f"top-level copy")
        plan["size_bytes"] = echo
    if patch is not None:
        base, err = _part_map(plan.get("parts") or [], "response part")
        if err:
            return err
        stray = sorted(set(patch) - set(base))
        if stray:
            return (f"patch parts {stray[:5]} are not in the response's "
                    f"part plan — a transcription slip, not a transfer "
                    f"error")
        plan["parts"] = [patch.get(p["part_number"], p)
                         for p in plan.get("parts") or []]
    return None


def main(plan_path):
    with open(plan_path) as f:
        try:
            plan = json.load(f)
        except ValueError as exc:
            # step 4 has the agent hand-edit this file — a trailing comma
            # or truncated paste is a plan slip, not a fatal condition
            return _fail({"plan": f"plan file is not valid JSON — "
                                  f"{exc}"}, [])
    if not isinstance(plan, dict):
        return _fail({"plan": "the plan file must be a JSON object — see "
                              "the shapes in this helper's docstring"}, [])
    shape_err = _resolve_shape(plan)
    if shape_err:
        return _fail({"plan": shape_err}, [])
    path = plan.get("file")
    if not path:
        return _fail({"plan": "plan needs \"file\" — the absolute local "
                              "path to upload"}, [])
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return _fail({"plan": f"cannot stat {path} — "
                              f"{exc.strerror or exc}"}, [])

    # scalar plan fields are coerced HERE so a transcription slip
    # ("2 GiB", "six") refuses as errors.plan, never as errors.fatal
    declared = plan.get("size_bytes")
    if declared is not None:
        declared, err = _pos_int(declared, "size_bytes")
        if err:
            return _fail({"plan": err}, [])

    mode = plan.get("mode")
    if mode is None:
        return _fail({"plan": "plan carries no mode — paste the start_upload "
                              "result verbatim under \"response\", or use an "
                              "explicit mode/url/parts shape"}, [])
    if mode not in ("single", "multipart"):
        return _fail({"plan": f"unrecognized mode {mode!r} — expected "
                              f"'single' or 'multipart'"}, [])
    if mode == "single":
        if declared is not None and declared != size:
            return _fail({"plan": f"local file is {size} bytes but the plan "
                                  f"was made for {declared} — restart "
                                  f"with a fresh start_upload"}, [])
        if not plan.get("url"):
            return _fail({"plan": "mode 'single' needs url — the presigned "
                                  "PUT url start_upload returned"}, [])
        if not plan.get("content_type"):
            # the signature pins Content-Type; PUTting without the header
            # fails as a 403 indistinguishable from an expired URL
            return _fail({"plan": "mode 'single' needs content_type — the "
                                  "value start_upload returned (it is folded "
                                  "into the signature)"}, [])
        try:
            body = _BoundedReader(path, 0, size)
            etag = _put(plan["url"], body, size, plan.get("content_type"))
        except urllib.error.HTTPError as exc:
            return _fail({"single": f"HTTP {exc.code}"}, [])
        except Exception as exc:  # noqa: BLE001 — the contract is one json line
            return _fail({"single": repr(exc)[:120]}, [])
        print(json.dumps({"ok": True, "etags": [], "etag": etag,
                          "local_md5": body.md5.hexdigest(),
                          "size_bytes": size}))
        return 0

    raw_parts = plan.get("parts")
    if isinstance(raw_parts, str):
        try:
            raw_parts = json.loads(raw_parts)
        except ValueError:
            return _fail({"plan": "\"parts\" is a string that is not valid "
                                  "JSON — paste the structure itself"}, [])
    if isinstance(raw_parts, dict) and isinstance(raw_parts.get("parts"),
                                                  list):
        raw_parts = raw_parts["parts"]
    plan["parts"] = raw_parts
    if "part_size" not in plan or not plan.get("parts"):
        return _fail({"plan": "multipart plan needs part_size and parts — "
                              "paste the start_upload result verbatim under "
                              "\"response\""}, [])
    done_rows = plan.get("done") or []
    dmap, err = _part_map(done_rows, "done")
    if err:
        return _fail({"plan": err}, [])
    done = {}
    for n, d in dmap.items():
        if not d.get("etag"):
            return _fail({"plan": f"done entry for part {n} carries no etag "
                                  f"— copy the previous run's etags rows "
                                  f"verbatim"}, [])
        if "local_md5" in d and not d["local_md5"]:
            # a truncated paste must not silently disarm the rewrite guard
            return _fail({"plan": f"done entry for part {n} carries an "
                                  f"empty local_md5 — copy the row verbatim "
                                  f"or drop the key"}, [])
        done[n] = d["etag"]
    # from here on, refusals echo the done rows back: the contract says
    # the failure JSON's etags are what the agent feeds back as done, so
    # a plan refusal must never wipe landed-part resume state
    echo_rows = done_rows
    if "response" not in plan and plan.get("size_bytes") is None:
        # the explicit shape's size_bytes IS its mutation guard (and the
        # source of its coverage check) — a stray part_count must never
        # waive it, or the shape is strictly less safe than response
        return _fail({"plan": "explicit multipart plan needs size_bytes — "
                              "the exact value sent to start_upload"}, [])
    part_size, err = _pos_int(plan["part_size"], "part_size")
    if err:
        return _fail({"plan": err + " (bytes) — the value the "
                              "start_upload response returned"}, [],
                     echo_rows)
    # workers is read only here (multipart) — single mode ignores it,
    # matching the SKILL's "multipart only" note
    workers = plan.get("workers")
    workers, err = _pos_int(6 if workers is None else workers, "workers")
    if err:
        return _fail({"plan": err}, [])
    if workers > 32:
        return _fail({"plan": "workers must be 1..32 — parallelism beyond "
                              "that only exhausts fds and S3 throttle"}, [],
                     echo_rows)
    _, err = _part_map(plan["parts"], "part", need_url=True)
    if err:
        return _fail({"plan": err}, [])
    # part_count comes from the response when there is one; the explicit
    # shape derives it from its required size_bytes — either way the
    # coverage check below applies to BOTH shapes
    expected = None if declared is None else -(-declared // part_size)  # ceil
    if "part_count" in plan:
        want_parts, err = _pos_int(plan["part_count"], "part_count")
        if err:
            return _fail({"plan": err}, [])
    elif expected is not None:
        want_parts = expected
    else:
        return _fail({"plan": "cannot determine part_count — the response "
                              "block carries no part_count and the plan no "
                              "size_bytes"}, [], echo_rows)
    if expected is not None and want_parts != expected:
        # the two facts disagree — uploading want_parts ranges of a
        # differently-sized file silently truncates or 403s
        return _fail({"plan": f"size_bytes {declared} with part_size "
                              f"{part_size} means {expected} parts, but "
                              f"the plan carries {want_parts} — abort_upload "
                              f"this session, then restart with a fresh "
                              f"start_upload"}, [])
    have = {p["part_number"] for p in plan["parts"]} | set(done)
    # bounds + count ≡ equality with {1..want_parts}, without materializing
    # a set a mangled part_size could blow up to billions of entries
    if len(have) != want_parts or min(have) < 1 or max(have) > want_parts:
        return _fail({"plan": f"parts+done cover {len(have)} part numbers "
                              f"in {min(have)}..{max(have)} but the plan "
                              f"needs 1..{want_parts} — a transcription "
                              f"slip, not a transfer error"}, [], echo_rows)
    if declared is not None and declared != size:
        return _fail({"plan": f"local file is {size} bytes but the plan was "
                              f"made for {declared} — abort_upload this "
                              f"session, then restart with a fresh "
                              f"start_upload"}, [], echo_rows)
    if declared is None:
        # the response shape carries no size_bytes; the plan still pins the
        # size to one part_size bucket, so a file mutated after start_upload
        # is refused by name instead of 403ing on signed part lengths
        if not ((want_parts - 1) * part_size < size <= want_parts * part_size):
            return _fail({"plan": f"local file is {size} bytes but the "
                                  f"response planned {want_parts} parts of "
                                  f"{part_size} — the file changed since "
                                  f"start_upload; abort_upload this session, "
                                  f"then restart with a fresh "
                                  f"start_upload"}, [], echo_rows)

    # Resume integrity: a size check cannot see a file REWRITTEN at the
    # same length between runs, which would weld old-bytes parts (done)
    # to new-bytes parts into a silently corrupt object. Done rows that
    # carry a local_md5 (the previous run's etags, copied verbatim) are
    # re-hashed from the disk of today — a pure local comparison, valid
    # on any bucket encryption; rows without one are trusted as-is. The
    # re-hash runs in the same-size pool the PUTs get (a near-complete
    # resume would otherwise stall serially on gigabytes of disk), and
    # STILL strictly before any PUT: a refusal means nothing was sent.
    # Verified rows re-emit their local_md5, so the guard stays armed on
    # the next resume instead of disarming after the first.
    stop = []                          # set by refuse() below

    def rehash_done(d):
        n = d["part_number"]           # already int-coerced by _part_map
        offset = (n - 1) * part_size
        body = _BoundedReader(path, offset, min(part_size, size - offset))
        try:
            while body.read(_CHUNK):
                if stop:
                    return n, None     # refusal already printed — bail
        finally:
            body.close()               # fd released even on a read error
        return n, body.md5.hexdigest()

    verified = {}
    to_verify = [d for d in done_rows if d.get("local_md5")]
    if to_verify:
        # submit + as_completed so a mismatch cancels the queued rehashes
        # instead of reading the rest of the file for an answer already
        # decided; at most `workers` in-flight reads finish behind it
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(rehash_done, d): d for d in to_verify}
            def refuse(msg):
                stop.append(True)       # in-flight rehashes exit at the
                for other in futs:      # next chunk instead of pinning
                    other.cancel()      # the interpreter open for minutes
                return _fail({"plan": msg}, [], echo_rows)

            for fut in as_completed(futs):
                try:
                    n, got = fut.result()
                except Exception as exc:  # noqa: BLE001 — name it, as plan
                    return refuse(f"cannot re-read part "
                                  f"{futs[fut]['part_number']} of {path} — "
                                  f"{repr(exc)[:120]}")
                if got != futs[fut]["local_md5"]:
                    return refuse(f"part {n}'s bytes on disk no longer "
                                  f"match the done row's local_md5. If that "
                                  f"row was copied verbatim, the file "
                                  f"changed since the previous run — "
                                  f"abort_upload and restart with a fresh "
                                  f"start_upload; if it may have been "
                                  f"retyped, re-copy the previous run's "
                                  f"etags verbatim and re-run first")
                verified[n] = got
        finally:
            # wait=False: on a refusal the answer is already printed by
            # _fail; in-flight reads finish on interpreter exit, and the
            # success path has drained every future via as_completed
            ex.shutdown(wait=False)

    todo = [p for p in plan["parts"] if p["part_number"] not in done]

    def upload_part(p):
        n = p["part_number"]
        offset = (n - 1) * part_size
        length = min(part_size, size - offset)
        body = _BoundedReader(path, offset, length)
        return n, _put(p["url"], body, length), body.md5.hexdigest()

    etags, md5s, failed, errors = dict(done), dict(verified), [], {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut, p in [(ex.submit(upload_part, p), p) for p in todo]:
            try:
                n, etag, local_md5 = fut.result()
                etags[n] = etag
                md5s[n] = local_md5
            except urllib.error.HTTPError as exc:
                failed.append(p["part_number"])
                errors[str(p["part_number"])] = f"HTTP {exc.code}"
            except Exception as exc:  # noqa: BLE001 — report, agent decides
                failed.append(p["part_number"])
                errors[str(p["part_number"])] = repr(exc)[:120]

    if failed:
        return _fail(errors, failed, _rows(etags, md5s))
    print(json.dumps({"ok": True, "etags": _rows(etags, md5s),
                      # paste-ready for complete_upload(parts=...) — only
                      # the two fields its schema declares, nothing retyped
                      "parts": [{"part_number": n, "etag": etags[n]}
                                for n in sorted(etags)]}))
    return 0


if __name__ == "__main__":
    # the one-json-line contract holds on EVERY path — a missing argument,
    # an unreadable plan, or a file deleted since start_upload must come
    # back as ok:false, never a bare traceback the agent has no branch for
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "etags": [], "failed_parts": [],
                          "errors": {"fatal": repr(exc)[:160]}}))
        sys.exit(1)
