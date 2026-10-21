import json
import boto3
import urllib.request
import urllib.error

cognito = boto3.client("cognito-idp")
sm = boto3.client("secretsmanager")

def handler(event, context):
    # keys only — a CFN custom-resource event carries deployment
    # values in ResourceProperties that do not belong in a log group
    print("Event keys:", sorted(event))
    request_type = event.get("RequestType")
    props = event.get("ResourceProperties", {})

    # Always signal success to CFN — failures would roll back the stack
    try:
        if request_type in ("Create", "Update"):
            _populate(props)
        _send(event, context, "SUCCESS", {})
    except Exception as exc:
        print("ERROR:", exc)
        _send(event, context, "FAILED", {}, reason=str(exc))

def _populate(props):
    user_pool_id   = props["UserPoolId"]
    client_id      = props["ClientId"]
    secret_arn     = props["SecretArn"]
    gateway_url    = props["GatewayUrl"]
    token_endpoint = props["TokenEndpoint"]
    scope          = props["Scope"]

    # Fetch the client secret — not available as a CFN GetAtt
    resp = cognito.describe_user_pool_client(
        UserPoolId=user_pool_id,
        ClientId=client_id,
    )
    client_secret = resp["UserPoolClient"]["ClientSecret"]

    bundle = {
        "endpointUrl":   gateway_url,
        "tokenEndpoint": token_endpoint,
        "clientId":      client_id,
        "clientSecret":  client_secret,
        "scope":         scope,
    }

    sm.put_secret_value(
        SecretId=secret_arn,
        SecretString=json.dumps(bundle),
    )
    print("Secret populated successfully")

def _send(event, context, status, data, reason=""):
    """Send response to the CFN pre-signed S3 URL."""
    props = event.get("ResourceProperties", {})
    body = json.dumps({
        "Status":             status,
        "Reason":             reason or status,
        # Use SecretArn as stable PhysicalResourceId so updates are
        # treated as in-place (not replacement) by CloudFormation.
        "PhysicalResourceId": props.get("SecretArn", context.log_stream_name),
        "StackId":            event["StackId"],
        "RequestId":          event["RequestId"],
        "LogicalResourceId":  event["LogicalResourceId"],
        "Data":               data,
    }).encode()

    req = urllib.request.Request(
        url=event["ResponseURL"],
        data=body,
        method="PUT",
        headers={"Content-Type": "", "Content-Length": str(len(body))},
    )
    with urllib.request.urlopen(req) as r:  # nosec B310
        print("CFN response status:", r.status)
