"""Custom resource: publish the client kit to the proxy bucket.

The kit (skills + configure.sh + get_gateway_token.sh + README) is
assembled at build time and BAKED INTO this Lambda's own package as
client_kit.zip — never copied from the dist bucket, because customers
deploying from our template may not have GetObject there. On stack
Create/Update this writes two objects to the proxy bucket:

    _toolkit/client-kit-{Version}.zip   (immutable, per tool version)
    _toolkit/client-kit.zip             (stable alias, overwritten)

A customer needs only their own AWS access:
    aws s3 cp s3://<proxy>/_toolkit/client-kit.zip .
Delete is a no-op (the objects are content the operator may keep).
"""
import json
import os
import urllib.request

import boto3

s3 = boto3.client("s3")

KIT_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "client_kit.zip")


def handler(event, context):
    print("Event:", json.dumps(event))
    request_type = event.get("RequestType")
    try:
        if request_type in ("Create", "Update"):
            _publish(event.get("ResourceProperties", {}))
        _send(event, context, "SUCCESS", {})
    except Exception as exc:                    # noqa: BLE001
        print("ERROR:", exc)
        _send(event, context, "FAILED", {}, reason=str(exc))


def _with_kit_env(body: bytes, secret_name: str, region: str) -> bytes:
    """Append kit.env — this deployment's identity — to the baked zip, so
    the published kit's scripts default to their OWN gateway's secret
    instead of a well-known name (which a derived GatewayName does not
    have). Fixed member timestamp => deterministic bytes => the ETag
    idempotency skip below keeps working across no-op republishes. No
    SecretName property (an older template invoking newer code) leaves
    the kit byte-identical to the baked one."""
    if not secret_name:
        return body
    import io
    import zipfile
    buf = io.BytesIO(body)
    with zipfile.ZipFile(buf, "a", zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("kit.env", date_time=(1980, 1, 1, 0, 0, 0))
        # ZipInfo defaults external_attr to 0 — some extractors read that
        # as mode 000, and an unreadable kit.env kills configure.sh at
        # the source line. 0644, explicitly.
        info.external_attr = 0o644 << 16
        zf.writestr(info,
                    "# written by this deployment's KitPublisher at publish\n"
                    f"KIT_SECRET_NAME={secret_name}\n"
                    f"KIT_REGION={region}\n")
    return buf.getvalue()


def _publish(props):
    import hashlib

    from botocore.exceptions import ClientError

    bucket = props["Bucket"]
    version = props.get("Version", "unknown")
    body = _with_kit_env(open(KIT_LOCAL, "rb").read(),
                         props.get("SecretName", ""),
                         os.environ.get("AWS_REGION", ""))
    # single-part put => ETag is the body md5; a matching ETag means
    # the published kit is already byte-identical and the write is
    # skipped (idempotent republish on no-op stack updates)
    etag = '"%s"' % hashlib.md5(body, usedforsecurity=False).hexdigest()
    for key in (f"_toolkit/client-kit-{version}.zip",
                "_toolkit/client-kit.zip"):
        try:
            existing = s3.head_object(Bucket=bucket, Key=key)["ETag"]
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code not in ("404", "NoSuchKey", "NotFound"):
                raise
            existing = None
        if existing == etag:
            print(f"unchanged s3://{bucket}/{key} — skipped")
            continue
        s3.put_object(Bucket=bucket, Key=key, Body=body,
                      ContentType="application/zip")
        print(f"published s3://{bucket}/{key} ({len(body)} bytes)")


def _send(event, context, status, data, reason=""):
    """Send response to the CFN pre-signed S3 URL."""
    props = event.get("ResourceProperties", {})
    body = json.dumps({
        "Status":             status,
        "Reason":             reason or status,
        # Stable PhysicalResourceId (bucket-scoped) so updates are
        # in-place, never replacement.
        "PhysicalResourceId": f"client-kit-{props.get('Bucket', 'na')}",
        "StackId":            event["StackId"],
        "RequestId":          event["RequestId"],
        "LogicalResourceId":  event["LogicalResourceId"],
        "Data":               data,
    }).encode()
    req = urllib.request.Request(
        url=event["ResponseURL"], data=body, method="PUT",
        headers={"Content-Type": "", "Content-Length": str(len(body))})
    with urllib.request.urlopen(req) as r:  # nosec B310
        print("CFN response status:", r.status)
