"""
AgentCore Gateway tool dispatcher for the prompt-registry micro-workflow.

Seven tools, ALL synchronous — this stack starts no executions (plan decision
8: no state machine), so there is no submit handle, no workflow_id, and no
idempotency_token anywhere. A stray token is warned about and IGNORED on every
tool (micro-workflow-bundle/README.md §C11), and per that rule the warning talks about the TOKEN
("nothing here reads it"), never about what the tool does.

  put_prompt        → parse ONCE, lint names, write the immutable pair
  get_prompt        → the artifact (+ source_s3uri); version optional → default
  list_prompts      → categories/versions with lifecycle metadata
  promote_prompt    → one-way stage ratchet (dev → test → prod)
  delete_prompt     → dev/test only; frees the version name
  deprecate_prompt  → the prod retirement path; `default` is protected
  describe_tool     → version, description, output_contract, vocabulary

Storage (README.md "Layout"): `_promptregistry/{category}/{version}/` holding
exactly `source.md` (byte-frozen, NEVER rewritten) and `artifact.json`
(content fields frozen at put; ONLY the lifecycle fields — stage, deprecated,
promoted_at, deprecated_at — are ever rewritten, by promote/deprecate).

`default` is the category's always-present entry point: born prod (put forces the
stage), never deprecatable, never deletable; a version-LESS get resolves to
it, but a NAMED missing version is PROMPT_NOT_FOUND — never a silent fallback
(v5 substituted default.md silently; that is the invisible-behavior-change
class this design kills).
"""
from __future__ import annotations

import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

import prompt_md
import vocabulary as vocabulary_mod

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_s3 = boto3.client("s3")

PROXY_BUCKET = os.environ.get("PROXY_BUCKET", "")
# The admin-surface switch (2026-09-08, the hybrid-search-admin
# pattern): ONE codebase, TWO functions. The standard agent-facing
# function runs "false" — categories whose DEFAULT version declares
# `Audience: workflow-internal` are invisible (list omits them, every
# category-taking tool refuses by name). The admin function runs
# "true" — the full surface; workflow children also point HERE (an
# internal caller is trusted by definition). Absent env = "true" so a
# not-yet-rewired deployment keeps serving its workflows mid-rollout;
# the template sets it explicitly on both functions.
SERVE_INTERNAL = os.environ.get("SERVE_INTERNAL", "true") == "true"
_CATEGORY_TOOLS = frozenset((
    "put_prompt", "get_prompt", "list_prompts", "promote_prompt",
    "delete_prompt", "deprecate_prompt"))
TOOL_VERSION = os.environ.get("TOOL_VERSION", "unknown")
# Must equal the GatewayTarget Name in media-toolkit-gateway/cfnstack.yaml.
TARGET_NAME = os.environ.get("TARGET_NAME", "prompt-registry")

REGISTRY_PREFIX = "_promptregistry"
STAGES = ("dev", "test", "prod")
_STAGE_ORDER = {"dev": 0, "test": 1, "prod": 2}
# The paste-accident bound (README.md "Why there is NO DynamoDB table"):
# not for S3's sake — a multi-megabyte "prompt" would ship to a model at
# real token cost. Largest real builtin is ~16 KB.
MAX_SOURCE_BYTES = 1_000_000

_TOOL_DELIMITER = "___"

# Cold-start loads, degrade-to-None with an ERROR log — the build script's
# zip-member guard is what makes the degraded state unreachable in a real
# deployment (same pattern as every sibling gateway).
try:
    with open(os.path.join(os.path.dirname(__file__),
                           "output_contract.json")) as _f:
        _OUTPUT_CONTRACT = json.load(_f)
except (OSError, json.JSONDecodeError) as _exc:
    logger.error("output_contract.json unavailable: %s", _exc)
    _OUTPUT_CONTRACT = None

# input_examples.json is the INPUT twin of the output contract: one minimal
# valid request per tool, served by describe_tool so an agent gets a
# known-good starting call instead of guessing shapes from prose. A unit
# test round-trips every example through this handler's own validation, so
# an example that drifts from the real contract fails the build. Same
# degradation contract as the output contract: never takes the CRUD tools
# down with it.
try:
    with open(os.path.join(os.path.dirname(__file__),
                           "input_examples.json")) as _f:
        _INPUT_EXAMPLES = json.load(_f)
except (OSError, json.JSONDecodeError) as _exc:
    logger.error("input_examples.json unavailable: %s", _exc)
    _INPUT_EXAMPLES = None

try:
    _VOCAB = vocabulary_mod.load_vocabulary()
    with open(vocabulary_mod._VOCAB_CANDIDATES[0]
              if os.path.exists(vocabulary_mod._VOCAB_CANDIDATES[0])
              else vocabulary_mod._VOCAB_CANDIDATES[1]) as _f:
        _VOCAB_RAW = json.load(_f)
except (OSError, json.JSONDecodeError, KeyError) as _exc:
    logger.error("vocabulary.json unavailable: %s", _exc)
    _VOCAB = None
    _VOCAB_RAW = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict, context) -> dict:
    event = event or {}
    logger.info("Event keys: %s", sorted(event.keys()))
    tool = _resolve_tool(context)
    logger.info("Tool: %r", tool)

    _warn_stray_token(event, tool)

    # ONE central gate: internal categories exist only on the admin
    # surface. Config check first (the gate GETs S3); scoped to the
    # tools that READ category, so a stray field on describe_tool is
    # warn-and-ignore like every other stray field (micro-workflow-bundle/README.md §C11).
    # put_prompt is additionally gated inside (a NEW category can be
    # born internal, which this pre-check cannot see).
    category = (event.get("category") or "").strip()
    if (category and not SERVE_INTERNAL and tool in _CATEGORY_TOOLS
            and PROXY_BUCKET and _category_internal(category)):
        return _error(
            f"category {category!r} is internal — it is served on the "
            "admin surface (prompt-registry-admin) only",
            "INTERNAL_CATEGORY")

    if tool == "put_prompt":
        return _put_prompt(event)
    if tool == "get_prompt":
        return _get_prompt(event)
    if tool == "list_prompts":
        return _list_prompts(event)
    if tool == "promote_prompt":
        return _promote_prompt(event)
    if tool == "delete_prompt":
        return _delete_prompt(event)
    if tool == "deprecate_prompt":
        return _deprecate_prompt(event)
    if tool == "describe_tool":
        return _describe_tool()

    return _error(f"Unknown tool: {tool!r}", "UNKNOWN_TOOL")


def _resolve_tool(context) -> str:
    try:
        raw = context.client_context.custom["bedrockAgentCoreToolName"]
        if _TOOL_DELIMITER in raw:
            return raw.split(_TOOL_DELIMITER, 1)[1]
        return raw
    except (AttributeError, KeyError, TypeError):
        logger.warning("bedrockAgentCoreToolName not found in the Lambda context")
        return ""


def _warn_stray_token(event: dict, tool: str) -> None:
    """micro-workflow-bundle/README.md §C11: a hallucinated parameter must not fail a request over
    a field that could not have changed the answer — and the message talks
    about the TOKEN, never about what the tool does."""
    for spelling in ("idempotency_token", "uuid"):
        if event.get(spelling) is not None:
            logger.warning("%s: `%s` was ignored — nothing here reads it",
                           tool, spelling)


def _error(message: str, code: str) -> dict:
    logger.warning("[%s] %s", code, message)
    return {"error": True, "errorCode": code, "message": message}


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _prefix(category: str, version: str) -> str:
    return f"{REGISTRY_PREFIX}/{category}/{version}"


def _artifact_key(category: str, version: str) -> str:
    return f"{_prefix(category, version)}/artifact.json"


def _source_key(category: str, version: str) -> str:
    return f"{_prefix(category, version)}/source.md"


def _artifact_internal(artifact) -> bool:
    """ONE spelling of the audience rule. Prefers the field save_prompt
    materializes; falls back to the aux header for artifacts written
    before 2026-09-08."""
    if artifact is None:
        return False
    if "internal" in artifact:
        return bool(artifact["internal"])
    return str((artifact.get("aux") or {}).get("audience")
               or "").startswith("workflow-internal")


# Category internality = the DEFAULT version's audience (the seeded
# builtin defines the category's nature). Cached for the Lambda
# lifetime; misses are NEVER cached (a category created later via
# admin put_prompt must not inherit a stale False, and garbage names
# must not grow the dict). Serve paths ALSO check per artifact, so a
# default-less category cannot smuggle internal versions past the gate.
_INTERNAL_CACHE: dict = {}


def _category_internal(category: str) -> bool:
    if category not in _INTERNAL_CACHE:
        artifact = _load_artifact(PROXY_BUCKET, category, "default")
        if artifact is None:
            return False
        _INTERNAL_CACHE[category] = _artifact_internal(artifact)
    return _INTERNAL_CACHE[category]


def _load_artifact(bucket: str, category: str, version: str):
    try:
        body = _s3.get_object(Bucket=bucket,
                              Key=_artifact_key(category, version))["Body"]
        return json.loads(body.read())
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def _write_artifact(bucket: str, category: str, version: str,
                    artifact: dict, if_none_match: bool = False) -> None:
    kwargs = {}
    if if_none_match:
        # S3 conditional write: the atomic half of write-if-absent. The HEAD
        # check before it is the friendly error; this is the race-proof one.
        kwargs["IfNoneMatch"] = "*"
    _s3.put_object(
        Bucket=bucket, Key=_artifact_key(category, version),
        Body=json.dumps(artifact, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json", **kwargs)


def _variable_names(variables: list) -> list:
    """Every root/leaf name across the declared variables, first-seen order."""
    seen: dict = {}

    def walk(node):
        seen.setdefault(node["root"], None)
        for leaf in node.get("leafs", []):
            walk(leaf)

    for variable in variables or []:
        walk(variable)
    return list(seen)


def _lint_names(variables: list) -> list:
    """The vocabulary warnings for put_prompt — advisory, NEVER blocking."""
    if _VOCAB is None:
        return ["vocabulary.json unavailable — variable names not linted"]
    warnings = []
    for name in _variable_names(variables):
        kind, detail = vocabulary_mod.classify(name, _VOCAB)
        if kind == "alias":
            warnings.append(f"'{name}': legacy spelling — canonical is "
                            f"'{detail}'")
        elif kind == "novel":
            if detail:
                warnings.append(f"'{name}': novel name — did you mean "
                                f"'{detail}'? (see vocabulary.json)")
            else:
                warnings.append(f"'{name}': novel name — not in "
                                "vocabulary.json; amend the vocabulary if "
                                "this is deliberate")
    return warnings


def save_prompt(bucket: str, category: str, text: str, *,
                stage: str = "dev", origin: str = "user",
                now_ms: int | None = None, overwrite: bool = False) -> dict:
    """
    The ONE save path — put_prompt and the Custom::PromptSeeder both go
    through here, so a seeded artifact and a user-put artifact cannot drift
    in shape. Raises prompt_md.ParseError / ValueError with a message fit to
    return; the callers translate to their error shapes.

    `overwrite=True` is the SEEDER'S dev backdoor ONLY (PromptSeedMode
    `overwrite-builtins`): it skips the exists-refusal and writes the pair
    unconditionally. put_prompt NEVER passes it — the API surface stays
    fully immutable; eligibility (origin builtin, source actually changed)
    is the seeder's check, made before calling here.
    """
    if not category or not prompt_md.NAME_RE.match(category):
        raise ValueError(
            f"category must match [A-Za-z0-9_.-]{{1,64}}, got {category!r}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError(
            f"prompt source is {len(text.encode('utf-8'))} bytes — over the "
            f"{MAX_SOURCE_BYTES} sanity bound (a multi-megabyte prompt is "
            "almost certainly a paste accident, and it would ship to a model "
            "at real token cost)")

    parsed = prompt_md.parse_prompt_md(text)
    version = parsed["version"]
    if not prompt_md.NAME_RE.match(version):
        raise ValueError(
            f"version must match [A-Za-z0-9_.-]{{1,64}}, got {version!r}")

    notes = []
    if version == "default" and stage != "prod":
        # `default` is the category's always-present entry point — born prod.
        stage = "prod"
        notes.append("version 'default' is always prod — the requested "
                     "stage was overridden")

    if not overwrite and _load_artifact(bucket, category, version) is not None:
        raise VersionExists(
            f"{category}/{version} already exists — versions are immutable; "
            "an edited prompt is a NEW version name (dev/test versions may "
            "be delete_prompt'ed to free the name)")

    now = int(now_ms if now_ms is not None else time.time() * 1000)
    artifact = {
        "category": category,
        "version": version,
        "origin": origin,
        "comment": parsed["comment"],
        "system": parsed["system"],
        "user_template": parsed["user_template"],
        "variables": parsed["variables"],
        "source_md5": parsed["source_md5"],
        "aux": parsed["aux"],
        # machine-readable visibility, ONE rule (_artifact_internal
        # falls back to the aux header for pre-2026-09-08 artifacts)
        "internal": str((parsed["aux"] or {}).get("audience")
                        or "").startswith("workflow-internal"),
        "created_at": now,
        "stage": stage,
        "deprecated": False,
        "promoted_at": None,
        "deprecated_at": None,
    }

    # source.md first, artifact.json second: artifact presence IS the
    # version-exists check, so a torn pair (source only) is invisible and
    # retryable rather than half-registered.
    _s3.put_object(Bucket=bucket, Key=_source_key(category, version),
                   Body=text.encode("utf-8"), ContentType="text/markdown")
    try:
        _write_artifact(bucket, category, version, artifact,
                        if_none_match=not overwrite)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "PreconditionFailed":
            raise VersionExists(
                f"{category}/{version} was created concurrently — versions "
                "are immutable; pick a new version name") from exc
        raise

    return {"artifact": artifact, "notes": notes,
            "warnings": _lint_names(parsed["variables"])}


class VersionExists(ValueError):
    pass


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _put_prompt(event: dict) -> dict:
    if not SERVE_INTERNAL:
        # a NEW category can be born internal — the central gate only
        # sees categories that already exist. Parse the submitted text
        # (whole-document parse; headers are what we read); a malformed
        # text falls through to save_prompt's real error.
        try:
            probe = prompt_md.parse_prompt_md(event.get("text") or "")
            born_internal = _artifact_internal(probe)
        except prompt_md.ParseError:
            born_internal = False   # save_prompt names the real error
        if born_internal:
            return _error(
                "this prompt declares Audience: workflow-internal — "
                "internal prompts are managed on the admin surface "
                "(prompt-registry-admin) only", "INTERNAL_CATEGORY")
    if not PROXY_BUCKET:
        return _error("PROXY_BUCKET env var must be set on this function",
                      "CONFIG_ERROR")
    category = (event.get("category") or "").strip()
    text = event.get("text") or ""
    stage = (event.get("stage") or "dev").strip().lower()
    try:
        result = save_prompt(PROXY_BUCKET, category, text, stage=stage,
                             origin="user")
    except VersionExists as exc:
        return _error(str(exc), "VERSION_EXISTS")
    except prompt_md.ParseError as exc:
        return _error(str(exc), "INVALID_PROMPT")
    except ValueError as exc:
        code = "PROMPT_TOO_LARGE" if "sanity bound" in str(exc) \
            else "INVALID_PARAMETER"
        return _error(str(exc), code)

    artifact = result["artifact"]
    response = {
        "category": artifact["category"],
        "version": artifact["version"],
        "stage": artifact["stage"],
        "origin": artifact["origin"],
        "source_s3uri": (f"s3://{PROXY_BUCKET}/"
                         f"{_source_key(category, artifact['version'])}"),
        "variables": [v["token"] for v in artifact["variables"]],
        "warnings": result["warnings"],
        "created_at": artifact["created_at"],
    }
    if result["notes"]:
        response["message"] = "; ".join(result["notes"])
    return response


def _get_prompt(event: dict) -> dict:
    category = (event.get("category") or "").strip()
    if not category:
        return _error("category is required", "MISSING_PARAMETER")
    # An ABSENT version means default; a NAMED missing version is NOT_FOUND —
    # never a silent fallback.
    version = (event.get("version") or "default").strip()
    artifact = _load_artifact(PROXY_BUCKET, category, version)
    if artifact is None:
        return _error(f"no prompt at {category}/{version}", "PROMPT_NOT_FOUND")
    if not SERVE_INTERNAL and _artifact_internal(artifact):
        # backstop for a default-less category: internal VERSIONS are
        # invisible here even when the category gate could not see them
        return _error(
            f"category {category!r} is internal — it is served on the "
            "admin surface (prompt-registry-admin) only",
            "INTERNAL_CATEGORY")
    return {**artifact,
            "source_s3uri":
                f"s3://{PROXY_BUCKET}/{_source_key(category, version)}"}


def _list_versions(category: str) -> list:
    """Version metadata under one category — per-version GETs, the documented
    cost of keeping lifecycle metadata in the artifact."""
    versions = []
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=PROXY_BUCKET,
                                   Prefix=f"{REGISTRY_PREFIX}/{category}/",
                                   Delimiter="/"):
        for cp in page.get("CommonPrefixes") or []:
            version = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            artifact = _load_artifact(PROXY_BUCKET, category, version)
            if artifact is None:
                continue   # torn pair mid-put; invisible by design
            if not SERVE_INTERNAL and _artifact_internal(artifact):
                continue   # default-less-category backstop
            versions.append({key: artifact.get(key) for key in
                             ("version", "stage", "origin", "deprecated",
                              "comment", "created_at")})
    return versions


def _list_prompts(event: dict) -> dict:
    category = (event.get("category") or "").strip()
    if category:
        versions = _list_versions(category)
        if not versions:
            return _error(f"no prompts under category {category!r}",
                          "PROMPT_NOT_FOUND")
        categories = [{"category": category, "versions": versions}]
    else:
        categories = []
        paginator = _s3.get_paginator("list_objects_v2")
        names = []
        for page in paginator.paginate(Bucket=PROXY_BUCKET,
                                       Prefix=f"{REGISTRY_PREFIX}/",
                                       Delimiter="/"):
            for cp in page.get("CommonPrefixes") or []:
                names.append(cp["Prefix"].rstrip("/").rsplit("/", 1)[-1])
        for name in sorted(names):
            if not SERVE_INTERNAL and _category_internal(name):
                continue        # invisible, not refused: nothing to see
            categories.append({"category": name,
                               "versions": _list_versions(name)})
    return {"categories": categories,
            "total_versions": sum(len(c["versions"]) for c in categories)}


def _promote_prompt(event: dict) -> dict:
    category = (event.get("category") or "").strip()
    version = (event.get("version") or "").strip()
    to_stage = (event.get("to_stage") or "prod").strip().lower()
    if not category or not version:
        return _error("category and version are required", "MISSING_PARAMETER")
    if to_stage not in STAGES:
        return _error(f"to_stage must be one of {STAGES}", "INVALID_PARAMETER")

    artifact = _load_artifact(PROXY_BUCKET, category, version)
    if artifact is None:
        return _error(f"no prompt at {category}/{version}", "PROMPT_NOT_FOUND")

    current = artifact.get("stage", "dev")
    if _STAGE_ORDER[current] >= _STAGE_ORDER[to_stage]:
        # Idempotent: a retried promotion must not error. Covers `default`
        # (born prod) with no special case.
        return {"category": category, "version": version, "stage": current,
                "previous_stage": current, "promoted_at":
                    artifact.get("promoted_at"),
                "message": f"already {current} — this call changed nothing"}

    previous = current
    # Read-modify-write of ONLY the lifecycle fields; content stays
    # byte-identical (pinned by test).
    artifact["stage"] = to_stage
    artifact["promoted_at"] = int(time.time() * 1000)
    _write_artifact(PROXY_BUCKET, category, version, artifact)
    return {"category": category, "version": version, "stage": to_stage,
            "previous_stage": previous, "promoted_at": artifact["promoted_at"]}


def _delete_prompt(event: dict) -> dict:
    category = (event.get("category") or "").strip()
    version = (event.get("version") or "").strip()
    if not category or not version:
        return _error("category and version are required", "MISSING_PARAMETER")

    artifact = _load_artifact(PROXY_BUCKET, category, version)
    if artifact is None:
        return _error(f"no prompt at {category}/{version}", "PROMPT_NOT_FOUND")
    stage = artifact.get("stage", "dev")
    if stage == "prod":
        return _error(
            f"{category}/{version} is prod — prod versions can only be "
            "deprecated (deprecate_prompt); delete is for dev/test iteration",
            "PROD_UNDELETABLE")

    _s3.delete_object(Bucket=PROXY_BUCKET,
                      Key=_artifact_key(category, version))
    _s3.delete_object(Bucket=PROXY_BUCKET, Key=_source_key(category, version))
    return {"category": category, "version": version, "deleted": True,
            "stage_was": stage,
            "message": f"version name '{version}' is freed for reuse; any "
                       "history entries naming it now point only at their "
                       "runs' own request.jsonl"}


def _deprecate_prompt(event: dict) -> dict:
    category = (event.get("category") or "").strip()
    version = (event.get("version") or "").strip()
    if not category or not version:
        return _error("category and version are required", "MISSING_PARAMETER")
    if version == "default":
        return _error(
            "version 'default' is the category's always-present entry point — a "
            "version-less get_prompt resolves to it, so deprecating it would "
            "orphan every such read", "DEFAULT_PROTECTED")

    artifact = _load_artifact(PROXY_BUCKET, category, version)
    if artifact is None:
        return _error(f"no prompt at {category}/{version}", "PROMPT_NOT_FOUND")
    if artifact.get("deprecated"):
        return {"category": category, "version": version,
                "stage": artifact.get("stage"), "deprecated": True,
                "deprecated_at": artifact.get("deprecated_at"),
                "message": "already deprecated — this call changed nothing"}

    artifact["deprecated"] = True
    artifact["deprecated_at"] = int(time.time() * 1000)
    _write_artifact(PROXY_BUCKET, category, version, artifact)
    return {"category": category, "version": version,
            "stage": artifact.get("stage"), "deprecated": True,
            "deprecated_at": artifact["deprecated_at"]}


def _describe_tool() -> dict:
    resp = {
        "version": TOOL_VERSION,
        "description": (
            "Prompt registry: versioned, immutable prompt templates on S3, "
            "parsed once at save time. No state machine — every tool is "
            "synchronous CRUD. Versions carry a stage (dev/test/prod): "
            "dev/test delete freely, prod deprecates only, 'default' is each "
            "category's protected entry point."),
    }
    if _OUTPUT_CONTRACT is not None:
        resp["output_contract"] = _OUTPUT_CONTRACT
    if _INPUT_EXAMPLES is not None:
        resp["input_examples"] = _INPUT_EXAMPLES
    if _VOCAB_RAW is not None:
        # The reviewed variable vocabulary, whole: a few KB, static per
        # version, and exactly what a prompt-authoring agent needs before
        # writing its first {{token}}.
        resp["vocabulary"] = _VOCAB_RAW
    return resp
