"""Deterministic `workflow_id` derivation — the answer to "is this the same work?".

COPIED, NOT IMPORTED.  This file is byte-identical in every gateway_tool/ that
derives a workflow_id.  The gateway zips are flat (`zip -j`, no `lib/`), so there
is nowhere shared to import from — the copies ARE the distribution mechanism,
exactly like `bucket_validator.py`.  `micro-workflow-bundle`'s
`TestWorkflowIdentityParity` is what keeps them honest; it compares whole files,
so **per-gateway variation must live in the CALL, never in this module.**

Nine hand-copied derivations had already drifted three ways before this module
existed, which is the reason it exists.

    workflow_id = uuid5(NAMESPACE_URL, canonical_json(payload − exclude)
                                       + "|" + tool_version
                                       + extra material)

THREE IDS, each meaning one thing.  The old naming collapsed the first two, and
that conflation cost a production bug and a full design detour:

  gateway INPUT   `idempotency_token`  the caller's assertion "this is the same
                                       request I sent before".  Optional.
                                       Consumed into the hash and DISCARDED.
  gateway OUTPUT  `workflow_id`        the service's name for the run.  The
                                       caller's handle for polling and outputs.
  SFN input /     `uuid`               THIS EXECUTION'S LOCAL ID.  Deliberately
  state `event`                        NOT renamed to workflow_id: it is
                                       sometimes a random child tag (see
                                       face-recognition's prepare_media_
                                       processing), so `uuid` is the broader and
                                       therefore truthful name.

The token rides in `extra_material`, never in the payload, so "never forwarded,
never stored" holds by construction rather than by remembering to pop it — the
same reason `exclude` beats assigning a field after the hash.

`tool_version` is APPENDED rather than folded into the dict so that a version bump
re-runs everything: changed shaping produces different work for the same request,
and returning the old result would attribute it to the new code.

Why `output.s3uri` is excluded (the canonical `exclude` entry): the output prefix
is DERIVED from the workflow_id --

    output.s3uri = f"s3://{bucket}/micro-workflow/{target}/{workflow_id}"

-- so including it would require knowing the workflow_id in order to compute the
input that computes the workflow_id.  The value does not exist at hash time.  This
is circularity, not a policy choice, and there is no randomness involved: a
caller-supplied `output_s3_prefix` is rejected/ignored by every gateway and is not
in any InputSchema, so set or unset, the same request yields the same workflow_id.

Passing it through `exclude` rather than relying on "assign s3uri AFTER hashing"
is deliberate: the old ordering was an invariant enforced only by statement order,
which is invisible to a reader and trivially broken by an edit that moves a line.
"""

import json
import re
import uuid as _uuid_mod

NAMESPACE = _uuid_mod.NAMESPACE_URL

# A token has to survive being a Step Functions execution name, because for most
# tools the workflow_id derived from it IS the execution name.
TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
TOKEN_ERROR = ("idempotency_token must be 1-80 chars of A-Z a-z 0-9 _ . -")  # nosec B105

# The pre-2026-08-06 spelling.  Rejected rather than accepted-as-an-alias: it
# used to mean BOTH "which asset" and "same request as before", and a caller
# still sending it cannot be assumed to mean the surviving half.  Nothing is
# deployed and there are no external consumers, so a dual-accept window would
# buy compatibility nobody needs and keep the ambiguity alive.
RETIRED_TOKEN_KEY = "uuid"  # nosec B105
RETIRED_TOKEN_ERROR = (
    "`uuid` is no longer accepted. Use `idempotency_token` to assert that this "  # nosec B105
    "request is a resend of a previous one; the service returns its own "
    "`workflow_id` for the run."
)


def canonical_json(payload) -> str:
    """The one canonical form.  Sorted keys, no whitespace.

    Every byte of this string is part of the hash preimage, so changing this
    function silently re-runs every request in flight at a fresh output prefix.
    Do not "tidy" it.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _strip(payload, exclude):
    """Return a deep copy of `payload` without the dotted paths in `exclude`.

    A missing path is not an error — `exclude` states an intent ("this field must
    never reach the hash"), and enforcing it only when the field happens to be
    present already would defeat the point.
    """
    if not exclude:
        return payload
    # Round-tripping through JSON deep-copies and, more importantly, proves the
    # payload is serializable BEFORE it reaches the hash.
    pruned = json.loads(canonical_json(payload))
    for path in exclude:
        *parents, leaf = path.split(".")
        target = pruned
        for key in parents:
            target = target.get(key) if isinstance(target, dict) else None
            if target is None:
                break
        if isinstance(target, dict):
            target.pop(leaf, None)
    return pruned


def read_idempotency_token(event):
    """Pull the caller's idempotency token off a gateway event.

    Returns `(token, error)` — exactly one is truthy, and `token` is None when
    the caller supplied nothing (which is legal for every tool that accepts a
    token at all).  Returning the error rather than raising keeps each gateway's
    own `_error()` shape, which differs between them, out of this module.

    The token is the caller's assertion "this is the same request I sent
    before".  It is consumed into the hash and DISCARDED: never forwarded to the
    state machine, never stored, never echoed back.  A caller that supplied one
    already has it, and echoing it is what produced the one bug this design
    replaced — a poll read `correlation_uuid` from the execution OUTPUT, a key
    no state writes, so a SUCCEEDED poll answered with the derived id while the
    submit answered with the caller's.  The echo was that bug's entire blast
    radius.

    Tools that do not honour a token must not call this — they accept the field
    and log instead; see `warn_idempotency_token`.  "Does not honour" is NOT the
    same as "starts no execution": four of them do start one (see that
    function's docstring), they just do not let a caller's token decide the
    outcome.
    """
    if RETIRED_TOKEN_KEY in event:
        return None, RETIRED_TOKEN_ERROR
    token = (event.get("idempotency_token") or "").strip()
    if not token:
        return None, None
    if not TOKEN_RE.match(token):
        return None, TOKEN_ERROR
    return token, None


def warn_idempotency_token(event):
    """Log line for a tool that does not honour a token, handed one.  Else None.

    A WARNING, deliberately not an error.  The caller here is an LLM, and a
    hallucinated parameter name is one of its routine failure modes — erroring
    would fail an otherwise correct and complete request over a field the tool
    was always going to ignore anyway.  The asymmetry decides it: accepting a
    token nobody can honour costs a log line that the one person debugging will
    read, while rejecting costs a working request.

    So this is not a validator and returns no error.  It exists to make "this
    tool ignores the field" auditable in CloudWatch rather than invisible, and
    it covers BOTH spellings — including the retired `uuid`, which for these
    tools is equally harmless and equally worth recording.

    **"Does not honour a token" is not "starts no execution", and the message
    must not claim it is.**  The callers split three ways, and only the first
    group starts nothing:

      * status polls, sync reads and searches — start no execution and must
        reflect current state on every call;
      * accumulating writes named `uuid4()` (`index_content`,
        `batch_index_faces`, `batch_delete_faces`, `delete_collection`) — these
        DO start executions, deliberately un-collapsible because two calls with
        different documents must both run; they dedup one layer down and
        content-addressed, on `doc_id` or the source ETag, which survives a
        re-run without the caller remembering anything;
      * `register_voice` — starts an execution AND dedups it, on the parameter
        hash alone, so an identical resubmit is absorbed.  It refuses a token
        because there is no dedup at the record layer beneath it, so a token
        could only ever be used to force a duplicate, permanently unremovable
        catalog entry.

    An earlier version of this said "this tool starts no execution, so every
    call runs" for all three groups.  That was false for five tools and, worse,
    argued against `register_voice`'s exemption in the one place a maintainer
    looks before changing it.  The wording is therefore deliberately about the
    TOKEN ("nothing here reads it") and never about what the tool does.

    Where the field IS honoured, use `read_idempotency_token`; that one does
    validate, because there a malformed token would become an execution name.
    """
    if RETIRED_TOKEN_KEY in event:
        return ("`uuid` is no longer a parameter of this tool and was ignored; "
                "it names nothing here and no response echoes it")
    if event.get("idempotency_token"):
        return ("idempotency_token was ignored: this tool does not deduplicate "
                "on a caller-supplied token. Whatever it deduplicates on — "
                "current state, a content hash, or its own parameters — is "
                "unaffected by the value you sent")
    return None


def compute_workflow_id(payload, *, tool_version, exclude=(),
                        extra_material=()) -> str:
    """Deterministic workflow_id.  Same inputs -> same string, in any process.

    payload         the SFN input as it will be sent, MINUS any field derived
                    from the workflow_id itself (see `exclude`).
    tool_version    appended to the preimage, never folded into the dict.
    exclude         dotted paths that must not reach the hash, e.g.
                    ("output.s3uri",).  Missing paths are ignored.
    extra_material  additional strings folded in after the version — used for
                    content-addressing (source ETags, a records digest) so that
                    identical parameters over CHANGED content re-run.

    The caller owns token folding: an idempotency token belongs in `payload`
    before this is called, under whatever key that gateway sends.  Keeping it out
    of this signature is what lets the file stay byte-identical.
    """
    material = canonical_json(_strip(payload, exclude)) + "|" + str(tool_version)
    for extra in extra_material or ():
        material += "|" + str(extra)
    return str(_uuid_mod.uuid5(NAMESPACE, material))
