"""
CloudFormation custom resource: proxy bucket validator.

Verifies the ProxyBucket exists and is owned by this account, creating it
(with public access blocked) if absent. FAILs the stack on ownership or
region mismatch so misconfiguration surfaces at deploy time, not at the
first workflow run.

Deployed in the same zip as the gateway tool Lambda; wired as
Handler: bucket_validator.handler on the ProxyBucketValidatorFunction.
Runs on stack Create/Update; Delete is a no-op (never deletes the bucket).
PhysicalResourceId is the bucket name, so changing the ProxyBucket
parameter triggers a no-op Delete of the old resource — the old bucket
survives.

This file is byte-identical across all six micro-workflows that declare a
ProxyBucketValidatorFunction. It is not importable from a shared lib: the
gateway zip is flat (`zip -j`, no lib/), so the copies ARE the distribution
mechanism. test_units.py::TestBucketValidatorParity in the bundle suite
enforces that they stay identical — fix a bug here and copy it out to all
six, or the test fails.
"""
import json
import time
import urllib.request

import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")


def handler(event, context):
    # Nothing above the try may raise. An exception escaping this handler
    # means _send() never runs, CFN never receives a response, and the stack
    # sits in DELETE_IN_PROGRESS for a full hour before giving up — that is
    # exactly how tts-on-aws orphaned its own validator function twice
    # (KeyError: 'AccountId' on the Delete event, because the reads below
    # used to sit outside the try and Delete does not need them at all).
    # PhysicalResourceId must also be resolvable in the failure path, so it
    # falls back to the CFN-supplied one rather than a property lookup.
    props = event.get("ResourceProperties") or {}
    bucket = props.get("BucketName") or event.get("PhysicalResourceId") or "unknown"
    try:
        if event["RequestType"] in ("Create", "Update"):
            account = props["AccountId"]
            region = props.get("Region", "us-east-1")
            _validate(bucket, account, region)
        _send(event, context, "SUCCESS", bucket, "")
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        _send(event, context, "FAILED", bucket, f"{type(exc).__name__}: {exc}")


def _validate(bucket, account, region):
    for attempt in (1, 2):
        try:
            s3.head_bucket(Bucket=bucket, ExpectedBucketOwner=account)
            return
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get(
                "ResponseMetadata", {}).get("HTTPStatusCode", 0)
            if code in ("404", "NoSuchBucket") or status == 404:
                _create(bucket, region)
                return
            if code in ("403", "AccessDenied") or status == 403:
                raise Exception(
                    f"bucket {bucket} exists but is not owned by "
                    f"account {account} (or access denied)")
            if status == 301:
                raise Exception(
                    f"bucket {bucket} exists in a different region "
                    f"(expected {region})")
            if status >= 500 and attempt == 1:
                time.sleep(2)
                continue
            raise


def _create(bucket, region):
    kwargs = {"Bucket": bucket}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    try:
        s3.create_bucket(**kwargs)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        # parallel-stack race: another stack created it first
        if code != "BucketAlreadyOwnedByYou":
            raise
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        })


def _send(event, context, status, bucket, reason):
    body = json.dumps({
        "Status": status,
        "Reason": reason or f"See {context.log_stream_name}",
        "PhysicalResourceId": bucket,
        "StackId": event["StackId"],
        "RequestId": event["RequestId"],
        "LogicalResourceId": event["LogicalResourceId"],
        "Data": {},
    }).encode()
    req = urllib.request.Request(
        event["ResponseURL"], data=body, method="PUT",
        headers={"Content-Type": ""})
    # The pre-signed S3 PUT is the only thing standing between a finished
    # handler and an hour of DELETE_IN_PROGRESS, so a transient failure here
    # gets retried rather than surfacing as a silent hang.
    for attempt in (1, 2, 3):
        try:
            urllib.request.urlopen(req, timeout=15)  # nosec B310
            return
        except Exception as exc:
            print(f"[WARN] cfn response PUT attempt {attempt} failed: {exc}")
            if attempt == 3:
                raise
            time.sleep(2 * attempt)
