"""
`workflow_id` — identity and idempotency (README.md §3), on the SHARED derivation.

```
workflow_id = workflow_identity.compute_workflow_id(
    request,                              # minus EXCLUDED_KEYS, via exclude=
    tool_version=TOOL_VERSION,
    extra_material=(records_digest[, idempotency_token]))
```

The derivation itself lives in `lib/workflow_identity.py` — the byte-identical
copied module every micro-workflow gateway uses (micro-workflow-bundle/README.md §C2), adopted here
2026-08-11 in place of this module's original hand-rolled uuid5. **That adoption
changed the hash preimage** (the shared canonical form is `ensure_ascii=True` and
the material order is `payload | tool_version | extras`, where the original was
`payload | digest | tool_version` with `ensure_ascii=False`), which is exactly the
change the golden-vector test exists to make loud: old and new code disagree about
the identity of every request, so the adoption shipped with a `.version` bump and
before any external caller depended on the old hash.

This module remains the *bedrock-inference-specific* layer over the shared one:
which keys are excluded, how records are digested, and where outputs land. The
gateway imports the flat `workflow_identity` copy directly (its zip root);
`StateValidateAndPlan` recomputes through THIS module — same exclude set, same
extras — so the two sides cannot disagree about the parameter hash.

Four deliberate choices, unchanged from the original design:

- **`mode` and `use_batch` are excluded.** Sync and batch for identical work are the
  same work — same records, same prompts, same model — and must collapse to one
  execution. Otherwise a caller that retries in batch after a sync timeout pays
  twice. The mode that actually ran is reported as `summary.inference_mode`.
- **`output` is excluded.** It is where the answer goes, not what the answer is —
  and it is *derived from* the id, so including it would be circular.
- **A digest of the records is included, not the records themselves.** Prompt content
  is therefore in the hash — the caller renders before submitting (§4.1), so an edited
  prompt changes the digest and changes the `workflow_id`. Hashing the uri alone
  would mean a corrected record set returns the previous set's result, because the
  same uri can be overwritten in place.
- **The digest is computed from the bytes here, NOT read from the S3 ETag.** The
  gateway tool must know `workflow_id` *before* the object exists (the prefix it
  writes to is derived from it), and an ETag is only the body MD5 for an
  unencrypted, single-part upload — under SSE-KMS a bucket-level encryption change
  would silently alter every `workflow_id` in the deployment.

`digest_records()` hashes the **canonicalized parsed** records rather than the raw
file, so indentation, key order, and `ensure_ascii` differences between the writer
and the reader cannot change the id — only the semantic content can.

The `idempotency_token` (micro-workflow-bundle/README.md §C2's three-id design) is the caller's
assertion "this is the same request I sent before". It narrows dedup rather than
replacing the parameter hash — same token + different records still re-runs — and
it is consumed into the hash and discarded: never forwarded to the state machine,
never stored, never echoed. **Absent token means "dedup on the parameter hash"**:
this stack's records are final rendered bytes and fully name the work (the
asset-shaped side of the contract's rule), so identical parameters are identical
work and expensive re-runs require an explicit act — a fresh token. That token IS
the rerun feature; there is no force flag.
"""
from __future__ import annotations

import hashlib
import json

from lib import workflow_identity

# Excluded from the hash. See the module docstring for why each one.
EXCLUDED_KEYS = ("output", "mode", "use_batch", "tool_version")


def canonical_json(payload: dict) -> str:
    """
    Stable serialization for the RECORDS DIGEST: sorted keys, no whitespace,
    `ensure_ascii=False`.

    Deliberately NOT the shared module's canonical form (that one is
    `ensure_ascii=True`): this function feeds `digest_records`, whose output is a
    hex string handed to the shared derivation as opaque `extra_material` — the two
    canonicalizations never mix. Kept exactly as originally shipped so the digest
    half of the preimage did not move when the derivation half did.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def hashable_request(request: dict) -> dict:
    """The request with the non-semantic keys removed. Reporting/debug helper —
    the shared module's `exclude=` does the stripping on the hash path itself."""
    return {key: value for key, value in (request or {}).items()
            if key not in EXCLUDED_KEYS}


def digest_records(records: list) -> str:
    """
    A stable digest of a parsed record list.

    Canonicalized per record, so only semantic content moves the digest: the gateway
    tool writes the file and `StateValidateAndPlan` reads it back through the same
    `parse_records`, and the two must agree without depending on how either one
    happened to serialize whitespace.

    `record_id` order **is** significant and is not sorted away: `response.jsonl` is in
    request order (§4.2), so two requests holding the same records in a different order
    produce differently-ordered outputs and are not the same work.
    """
    digest = RecordsDigest()
    for record in records or []:
        digest.update(record)
    return digest.hexdigest()


class RecordsDigest:
    """Incremental form of `digest_records` — same preimage bytes, one record
    at a time, so the gateway's streaming staged-records path can hash a file
    it never holds. `digest_records` delegates here: one preimage, two entry
    points, and the shared-identity golden vector pins them together."""

    def __init__(self):
        self._digest = hashlib.sha256()

    def update(self, record: dict) -> None:
        self._digest.update(canonical_json(record).encode("utf-8"))
        self._digest.update(b"\n")

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def compute_workflow_id(request: dict, *, records_digest: str,
                        tool_version: str,
                        idempotency_token: str | None = None) -> str:
    """
    Deterministic `workflow_id` through the shared derivation. Same inputs → same
    string, in any process — and in any SIBLING STACK, which is the point of the
    shared module.

    `tool_version` is a separate preimage term rather than folded into the dict so
    that a version bump re-runs everything: a change to the shaping in
    `lib/bedrock.py` produces different calls for the same request, and returning
    the old result would attribute it to the new code.

    `idempotency_token`, when supplied, is folded as trailing extra material and
    then discarded — it narrows dedup (fresh token → fresh run at a fresh prefix),
    which is the caller's rerun mechanism. `StateValidateAndPlan` recomputes
    WITHOUT a token (it never sees one, by design), so its recompute equals the
    execution name exactly when no token was used — see its docstring for how the
    comparison is worded because of that.
    """
    extras = [str(records_digest or "")]
    if idempotency_token:
        extras.append(idempotency_token)
    return workflow_identity.compute_workflow_id(
        request or {}, tool_version=str(tool_version or ""),
        exclude=EXCLUDED_KEYS, extra_material=tuple(extras))


def output_prefix(target_name: str, workflow_id: str) -> str:
    """`micro-workflow/{TARGET_NAME}/{workflow_id}` — the §3 layout."""
    return f"micro-workflow/{target_name}/{workflow_id}"
