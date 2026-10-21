"""
Custom::PromptSeeder — install the builtin prompts at stack create/update.

Ships in the SAME flat zip as the gateway dispatcher and goes through the
SAME save path (`lambda_function.save_prompt`), so a seeded artifact and a
user-put artifact cannot drift in shape — one parser, one pair-write, one
place stage/default rules live.

Semantics (README.md "Lifecycle" + Design record §3.4):

  * walks `prompts/{category}/{version}.md` packaged beside this module
    (flat zip: prompts/ keeps its directory structure at the zip root);
  * default mode `write-if-absent`, never overwrite — versions are immutable
    and user edits are new versions, so absence is a plain existence check
    and a stack update cannot clobber anything. An updated builtin ships as
    a NEW version name.
  * `SeedMode: overwrite-builtins` is THE DEV BACKDOOR (settled 2026-08-17,
    for rebuilding category defaults while the new workflows are developed):
    a shipped prompt whose pair already exists is REWRITTEN in place, but
    only when the existing artifact is `origin: "builtin"` AND its
    source_md5 differs from the shipped bytes. `origin: "user"` versions are
    never touched in any mode, and the API surface (put_prompt) stays fully
    immutable either way. At GA the parameter goes back to (and stays at)
    write-if-absent.
  * seeded prompts land `stage: "prod"`, `origin: "builtin"`;
  * the filename stem MUST equal the header's `Version:` — a mismatch is a
    packaging error and FAILS the stack loudly rather than seeding a prompt
    whose stored version disagrees with its path.

CUSTOM-RESOURCE DISCIPLINE (micro-workflow-bundle/README.md §C9, verbatim from the proven
Custom::ModelRegistrySeeder): no statement that can raise sits outside the
try, Delete ALWAYS succeeds and deliberately deletes nothing (the bucket is
shared and prompts outlive any one stack), and the failure path can always
resolve a physical id.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Repo layout keeps prompts/ at the ROOT with this module under
# gateway_tool/; the deployed zip is FLAT with prompts/ beside the modules.
_PROMPTS_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts"),
)
_PROMPTS_DIR = next((c for c in _PROMPTS_CANDIDATES if os.path.isdir(c)),
                    _PROMPTS_CANDIDATES[0])
_PHYSICAL_ID_FALLBACK = "prompt-seeder"


def lambda_handler(event: dict, context) -> dict:
    physical_id = _PHYSICAL_ID_FALLBACK
    try:
        physical_id = event.get("PhysicalResourceId") or physical_id
        request_type = event.get("RequestType") or ""
        props = event.get("ResourceProperties") or {}

        if request_type == "Delete":
            # Deliberately leaves every prompt in place — see the docstring.
            _respond(event, context, "SUCCESS", physical_id,
                     {"Message": "prompts left in place"})
            return {"handled": True, "action": "delete-noop"}

        bucket = props["Bucket"]
        seed_mode = (props.get("SeedMode") or "write-if-absent").strip()
        if seed_mode not in ("write-if-absent", "overwrite-builtins"):
            raise ValueError(
                f"SeedMode {seed_mode!r} is not one of "
                "write-if-absent | overwrite-builtins")
        physical_id = f"{_PHYSICAL_ID_FALLBACK}-{bucket}"

        import lambda_function

        seeded, updated, skipped = [], [], []
        for category in sorted(os.listdir(_PROMPTS_DIR)):
            cat_dir = os.path.join(_PROMPTS_DIR, category)
            if not os.path.isdir(cat_dir):
                continue
            for name in sorted(os.listdir(cat_dir)):
                if not name.endswith(".md"):
                    continue
                stem = name[:-3]
                with open(os.path.join(cat_dir, name), encoding="utf-8") as f:
                    text = f.read()
                # BEFORE the write: a mismatch must fail the stack without
                # having seeded anything under the wrong name.
                import prompt_md
                parsed = prompt_md.parse_prompt_md(text)
                version = parsed["version"]
                if version != stem:
                    raise ValueError(
                        f"prompts/{category}/{name}: header Version "
                        f"{version!r} != filename stem {stem!r} — packaging "
                        "error; nothing was written")
                try:
                    lambda_function.save_prompt(
                        bucket, category, text, stage="prod",
                        origin="builtin")
                    seeded.append(f"{category}/{version}")
                    continue
                except lambda_function.VersionExists:
                    pass
                # The pair exists. In overwrite-builtins mode ONLY, rewrite
                # it in place when it is ours (origin builtin) and the
                # shipped bytes actually changed; a user version is never
                # touched in any mode.
                if seed_mode != "overwrite-builtins":
                    skipped.append(f"{category}/{stem}")
                    continue
                existing = lambda_function._load_artifact(
                    bucket, category, version)
                if (existing is None
                        or existing.get("origin") != "builtin"
                        or existing.get("source_md5")
                        == parsed["source_md5"]):
                    skipped.append(f"{category}/{stem}")
                    continue
                lambda_function.save_prompt(
                    bucket, category, text, stage="prod",
                    origin="builtin", overwrite=True)
                updated.append(f"{category}/{version}")

        logger.info("mode=%s seeded %d, updated %d, skipped %d: %s / %s / %s",
                    seed_mode, len(seeded), len(updated), len(skipped),
                    seeded, updated, skipped)
        _respond(event, context, "SUCCESS", physical_id,
                 {"Message": f"seeded {len(seeded)}, "
                             f"updated {len(updated)}, "
                             f"skipped {len(skipped)} existing"})
        return {"handled": True, "seeded": len(seeded),
                "updated": len(updated), "skipped": len(skipped)}

    except Exception as exc:                                       # noqa: BLE001
        logger.exception("prompt seeding failed")
        _respond(event, context, "FAILED", physical_id,
                 reason=f"{type(exc).__name__}: {exc}")
        return {"handled": False, "error": str(exc)}


def _respond(event: dict, context, status: str, physical_id: str,
             data: dict | None = None, reason: str = "") -> None:
    """Best-effort CFN response; never raises (a raise here re-parks the
    stack for the full custom-resource timeout, which is the failure §9
    exists to prevent)."""
    try:
        body = json.dumps({
            "Status": status,
            "Reason": (reason or f"see CloudWatch log stream "
                       f"{getattr(context, 'log_stream_name', '')}")[:1024],
            "PhysicalResourceId": physical_id,
            "StackId": event.get("StackId"),
            "RequestId": event.get("RequestId"),
            "LogicalResourceId": event.get("LogicalResourceId"),
            "Data": data or {},
        }).encode("utf-8")
        request = urllib.request.Request(
            event["ResponseURL"], data=body, method="PUT",
            headers={"Content-Type": ""})
        urllib.request.urlopen(request, timeout=30)  # nosec B310
    except Exception:                                              # noqa: BLE001
        logger.exception("could not deliver the CFN response")
