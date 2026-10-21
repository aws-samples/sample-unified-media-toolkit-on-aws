import io
import json
import time
import urllib.request
import zipfile
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")
ddb = boto3.client("dynamodb")


def _seed_default_collection(table_name, workflow_cfg):
    """
    Seed the 'default' collection record from workflow_config/hybrid-search.json
    (skip-if-exists — operator edits survive redeploys). Callers that omit
    collection_id resolve to this record at runtime; the global config file
    stays the SOURCE for engine settings (SLM models, prompt) only.
    """
    scopes = workflow_cfg.get("schemas") or []
    schema_map = {s: workflow_cfg.get(s, "") for s in scopes if workflow_cfg.get(s)}

    def paths_to_ddb(paths):
        return {"M": {p: {"M": {k: {"N": str(v)} for k, v in w.items()}}
                      for p, w in (paths or {}).items()}}

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    item = {
        "collection_id":         {"S": "default"},
        "name":                  {"S": "Default"},
        "active_scopes":         {"L": [{"S": s} for s in scopes]},
        "schema_map":            {"M": {k: {"S": v} for k, v in schema_map.items()}},
        "retrieval_paths":       paths_to_ddb(workflow_cfg.get("retrievalPaths")),
        "image_retrieval_paths": paths_to_ddb(workflow_cfg.get("imageRetrievalPaths")),
        "weights_locked":        {"BOOL": False},
        "created_by":            {"S": "deploy"},
        "created_at":            {"S": now},
        "updated_at":            {"S": now},
    }
    try:
        ddb.put_item(TableName=table_name, Item=item,
                     ConditionExpression="attribute_not_exists(collection_id)")
        return "seeded"
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return "exists"
        raise


def send(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="PUT",
                                 headers={"Content-Type": ""})
    urllib.request.urlopen(req)  # nosec B310


def _matches(filename, pattern):
    if pattern == "*":
        return True
    if pattern.startswith("*."):
        return filename.endswith(pattern[1:])
    return filename == pattern


def _copy_file(zf, zip_name, dest_bucket, dest_key,
               on_update, content_type, is_create):
    packaged_bytes = zf.read(zip_name)

    if is_create or on_update == "overwrite":
        body = packaged_bytes
    elif on_update == "skip":
        try:
            s3.head_object(Bucket=dest_bucket, Key=dest_key)
            return "skipped"
        except:
            pass
        body = packaged_bytes
    else:
        raise ValueError(f"Unknown on_update strategy: {on_update!r}")

    s3.put_object(Bucket=dest_bucket, Key=dest_key,
                  Body=body, ContentType=content_type)
    return "uploaded"


def handler(event, context):
    print(json.dumps(event, default=str))
    physical_id = event.get("PhysicalResourceId") or event["LogicalResourceId"]
    resp = {"StackId": event["StackId"], "RequestId": event["RequestId"],
            "LogicalResourceId": event["LogicalResourceId"],
            "PhysicalResourceId": physical_id}
    try:
        if event["RequestType"] == "Delete":
            send(event["ResponseURL"], {**resp, "Status": "SUCCESS", "Data": {}})
            return

        props = event["ResourceProperties"]
        src_bucket = props["SrcBucket"]
        src_key = props["SrcKey"]
        dest_bucket = props["DestBucket"]
        is_create = event["RequestType"] == "Create"
        file_specs = props["Files"]  # list of {src_prefix, dest_key_prefix, filter, on_update, content_type}

        # Name the object on failure: CloudFormation surfaces only this
        # message, and a bare "Access Denied" cannot be told apart from a
        # missing key (GetObject on an absent key answers 403 to a caller
        # without ListBucket on that bucket).
        try:
            zip_obj = s3.get_object(Bucket=src_bucket, Key=src_key)
            zip_data = zip_obj["Body"].read()
        except Exception as e:
            raise RuntimeError(
                f"cannot read deployment zip s3://{src_bucket}/{src_key} — "
                f"{type(e).__name__}: {e}") from e

        total_uploaded = 0
        total_skipped = 0

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            all_names = zf.namelist()
            for spec in file_specs:
                src_prefix = spec["src_prefix"]
                dest_prefix = spec["dest_key_prefix"]
                filt = spec.get("filter", "*")
                on_update = spec.get("on_update", "skip")
                content_type = spec.get("content_type", "application/octet-stream")

                candidates = [
                    n for n in all_names
                    if n.startswith(src_prefix)
                    and "/" not in n[len(src_prefix):]
                    and n != src_prefix
                ]
                candidates = [n for n in candidates if _matches(n[len(src_prefix):], filt)]

                if not candidates:
                    raise ValueError(f"No files matching '{src_prefix}{filt}' found in zip")

                for zip_name in candidates:
                    filename = zip_name[len(src_prefix):]
                    dest_key = f"{dest_prefix}/{filename}"
                    result = _copy_file(zf, zip_name, dest_bucket, dest_key,
                                        on_update, content_type, is_create)
                    if result == "uploaded":
                        total_uploaded += 1
                        print(f"PUT s3://{dest_bucket}/{dest_key} ({on_update})")
                    else:
                        total_skipped += 1
                        print(f"SKIP s3://{dest_bucket}/{dest_key}")

        # Seed the default collection AFTER file copies so the workflow
        # config the record derives from is guaranteed present in S3.
        seed_result = ""
        collection_table = props.get("CollectionTable", "")
        workflow_config_key = props.get("WorkflowConfigKey", "")
        if collection_table and workflow_config_key:
            cfg_obj = s3.get_object(Bucket=dest_bucket, Key=workflow_config_key)
            workflow_cfg = json.loads(cfg_obj["Body"].read())
            seed_result = _seed_default_collection(collection_table, workflow_cfg)
            print(f"default collection: {seed_result}")

        print(f"uploaded={total_uploaded} skipped={total_skipped}")
        send(event["ResponseURL"],
             {**resp, "Status": "SUCCESS",
              "Data": {"Uploaded": total_uploaded, "Skipped": total_skipped,
                       "DefaultCollection": seed_result}})
    except Exception as exc:
        print(f"ERROR: {exc}")
        send(event["ResponseURL"],
             {**resp, "Status": "FAILED", "Reason": str(exc), "Data": {}})
