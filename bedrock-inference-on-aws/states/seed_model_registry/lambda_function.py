"""
Custom::ModelRegistrySeeder — install the model registry at stack create/update
(README.md §P5.1).

Ends the implicit dependency on the m2c core stack's `CreateModelList`: this stack
seeds its **own** `availablemodels.json`, from `models/availablemodels_master.json`
packaged in this zip. ONE master list, filtered per deployment — a model ships iff
the deploy geo is in its `geos` — so the three hand-maintained per-geo lists the
core stack carries cannot drift here; region availability is data on each entry.

At seed time the master is RESOLVED for this deployment:

  * entries whose `geos` excludes the deploy geo are dropped (Llama 4 and Nova
    Premier disappear from an EU artifact mechanically, which is the region story:
    `resolve_model`'s loud ModelNotRegistered is the answer an EU deployment gives
    for Llama 4, at plan time, before a token is spent);
  * `inference_region: "geo"` (and `batch_job` routing with it — batch follows
    `inference_region`, there is no separate batch region) resolves to the deploy
    geo, so the installed artifact carries only concrete values:
    `global` | `us`|`eu`|`apac` | a region literal;
  * the envelope stamps `registry_version` (this stack's `.version`) and `geo`,
    so "which list is installed" is answerable from the artifact itself.

The write is SKIPPED when the installed artifact already carries this
`registry_version` — a stack update that did not bump `.version` must not clobber
an operator-edited registry. A bumped version always overwrites: the registry is
seeded configuration, and edits that must survive upgrades belong in the master.

CUSTOM-RESOURCE DISCIPLINE (micro-workflow-bundle/README.md §C9, and the two live incidents behind it):
a custom resource that raises before responding leaves CloudFormation waiting a
full hour and orphans the stack. So no statement that can raise sits outside the
try, Delete ALWAYS succeeds (it deliberately does NOT delete the artifact — the
registry may be shared with a replacement stack, and a missing registry bricks
every in-flight execution), and the failure path can always resolve a physical id.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request


def _bedrock_vocab():
    """The gateway-tool zip this module ships in carries lib/ at its root
    (the dispatcher imports lib.identity the same way), so the contract
    vocabularies are IMPORTED, never mirrored. Lazy, resolved inside the
    handler's try — the custom-resource discipline is that nothing able to
    raise sits at module level, or an import failure skips _respond and
    hangs the stack for the full CFN timeout. lib/bedrock.py itself is
    stdlib-only, so the import cannot drag boto3 in."""
    from lib import bedrock
    return bedrock

logger = logging.getLogger()
logger.setLevel(logging.INFO)

GEOS = ("us", "eu", "apac")

# Region-prefix → geo vocabulary. The template derives Geo from the deploy
# region's first hyphen-segment, and for every APAC region that segment is
# "ap" (ap-southeast-1, ap-northeast-1, ...) while the geo vocabulary — the
# inference-profile prefix, `geos` in the master, `inference_region` — says
# "apac". Without this alias an APAC deploy failed the whole stack on the
# validator below. Normalized HERE rather than in the template because
# Fn::FindInMap accepts only Ref/FindInMap as keys, not Fn::Select, so a
# template-side translation would mean hand-enumerating every APAC region.
_GEO_ALIASES = {"ap": "apac"}


def _normalize_geo(raw) -> str:
    """The Geo property → a member of GEOS, or ValueError. Pure; unit-tested
    directly (an APAC deploy is exactly the case a us/eu test bed never runs)."""
    geo = _GEO_ALIASES.get(str(raw or "").lower(), str(raw or "").lower())
    if geo not in GEOS:
        raise ValueError(
            f"Geo must be one of {GEOS} or a region prefix that aliases to one "
            f"({_GEO_ALIASES}), got {raw!r} — it is derived from the deploy "
            "region's prefix, so anything else means the template's "
            "!Select/!Split moved, or this stack was deployed to a region "
            "family without a model list (ca-, sa-, me-, af-, il-), which "
            "should fail exactly like this")
    return geo

# Resolved relative to this file: states/seed_model_registry/ -> the zip root.
_MASTER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "availablemodels_master.json")

_PHYSICAL_ID_FALLBACK = "model-registry-seeder"


def lambda_handler(event: dict, context) -> dict:
    physical_id = _PHYSICAL_ID_FALLBACK
    try:
        physical_id = event.get("PhysicalResourceId") or physical_id
        request_type = event.get("RequestType") or ""
        props = event.get("ResourceProperties") or {}

        if request_type == "Delete":
            # Deliberately leaves the artifact in place — see the module docstring.
            _respond(event, context, "SUCCESS", physical_id,
                     {"Message": "registry left in place"})
            return {"handled": True, "action": "delete-noop"}

        bucket = props["Bucket"]
        key = props["RegistryKey"]
        geo = _normalize_geo(props.get("Geo"))
        version = str(props.get("Version") or "")

        physical_id = f"{_PHYSICAL_ID_FALLBACK}-{bucket}-{key}"

        import boto3
        s3 = boto3.client("s3")

        installed_version = _installed_version(s3, bucket, key)
        if installed_version == version and version:
            logger.info("registry at s3://%s/%s already at %s — skipping",
                        bucket, key, version)
            _respond(event, context, "SUCCESS", physical_id,
                     {"Message": f"already at {version}", "Models": "unchanged"})
            return {"handled": True, "action": "skip"}

        artifact = _resolve(_load_master(), geo, version)
        s3.put_object(
            Bucket=bucket, Key=key,
            Body=json.dumps(artifact, indent=2).encode("utf-8"),
            ContentType="application/json")
        logger.info("seeded %d model(s) for geo=%s at s3://%s/%s (%s)",
                    len(artifact["models"]), geo, bucket, key, version)
        _respond(event, context, "SUCCESS", physical_id,
                 {"Message": f"seeded {len(artifact['models'])} models",
                  "Models": str(len(artifact["models"]))})
        return {"handled": True, "action": "seed",
                "models": len(artifact["models"])}

    except Exception as exc:                                       # noqa: BLE001
        logger.exception("model registry seeding failed")
        _respond(event, context, "FAILED", physical_id,
                 reason=f"{type(exc).__name__}: {exc}")
        return {"handled": False, "error": str(exc)}


def _load_master() -> dict:
    with open(_MASTER_PATH, encoding="utf-8") as f:
        return json.load(f)


def _installed_version(s3, bucket: str, key: str) -> str:
    """The installed artifact's registry_version, or "" when absent/foreign.

    A bare-list artifact (the v5 core stack's shape) has no envelope and reports
    "" — so the first seed after cutover always overwrites it, which is the point
    of the cutover."""
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        installed = json.loads(body)
    except Exception:                                              # noqa: BLE001
        return ""
    if isinstance(installed, dict):
        return str(installed.get("registry_version") or "")
    return ""



def _require_capabilities(entry: dict) -> None:
    """The seed-time half of lib/bedrock.contract_findings — ONE battery of
    checks, raised here (failing the deploy, naming the entry) and warned
    at resolve time for artifacts that bypass the seeder. Runs BEFORE the
    geo filter so a bad entry fails every deploy, not only the geos that
    carry it. Born from the sonnet-5 -> sonnet-5-5 swap (a wrong
    thinking_default that 400'd every quiet SME request) — honestly: THAT
    exact slip was vocabulary-valid and only a live probe catches its like;
    what dies here is the adjacent class (missing keys, typos, quoted
    booleans, self-contradictory pairs) at the only moment the data
    changes."""
    problems = _bedrock_vocab().contract_findings(entry)
    if problems:
        mid = (entry.get("model_id") or entry.get("modelId")
               or "<missing model_id>")
        raise ValueError(f"registry entry {mid!r}: " + "; ".join(problems))

def _resolve(master: dict, geo: str, version: str) -> dict:
    """The master list -> this deployment's artifact. Pure; unit-tested directly."""
    models = []
    for entry in master.get("models") or []:
        _require_capabilities(entry)
        if geo not in (entry.get("geos") or GEOS):
            continue
        resolved = json.loads(json.dumps(entry))   # deep copy
        resolved.pop("geos", None)
        if resolved.get("inference_region") == "geo":
            resolved["inference_region"] = geo
        models.append(resolved)
    return {"registry_version": version, "geo": geo, "models": models}


def _respond(event: dict, context, status: str, physical_id: str,
             data: dict | None = None, reason: str = "") -> None:
    """cfn-response, stdlib only. Its own failure is logged, never raised — the
    caller is already on the failure path when this matters most."""
    url = event.get("ResponseURL")
    if not url:
        logger.warning("no ResponseURL on the event — direct invocation?")
        return
    body = json.dumps({
        "Status": status,
        "Reason": (reason or f"see CloudWatch log stream "
                   f"{getattr(context, 'log_stream_name', '')}")[:1024],
        "PhysicalResourceId": physical_id,
        "StackId": event.get("StackId") or "",
        "RequestId": event.get("RequestId") or "",
        "LogicalResourceId": event.get("LogicalResourceId") or "",
        "Data": data or {},
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=body, method="PUT",
            headers={"Content-Type": "", "Content-Length": str(len(body))})
        urllib.request.urlopen(req, timeout=30)  # nosec B310
    except Exception:                                              # noqa: BLE001
        logger.exception("could not send the CFN response")
