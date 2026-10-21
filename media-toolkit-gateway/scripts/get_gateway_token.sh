#!/bin/bash
# Headers helper for the media-toolkit MCP endpoint (Claude Code
# headersHelper / Codex http_headers_helper contract): fetches Cognito
# M2M credentials from Secrets Manager, exchanges them for an access
# token via client_credentials, and prints {"Authorization": "Bearer ..."}
# on stdout.
#
# Caching: keyed PER DEPLOYMENT (secret name + region), so two
# configured gateways on one machine never serve each other's token,
# and the TTL comes from the token response's own expires_in (minus a
# 10-minute safety margin) — never a hardcoded assumption about the
# deployment's configured validity. GATEWAY_TOKEN_NO_CACHE=1 forces a
# fresh mint (configure.sh --refresh-token uses it).
set -euo pipefail

# default precedence: env var (the configure.sh wrapper always exports
# both) > kit.env (the deployment identity the KitPublisher wrote into
# the published kit) > the legacy well-known name
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_HERE/../kit.env" ] && . "$_HERE/../kit.env"
SECRET_NAME="${GATEWAY_SECRET_NAME:-${KIT_SECRET_NAME:-/media-toolkit-gateway/dev/gateway-credentials}}"
REGION="${AWS_REGION:-${KIT_REGION:-eu-west-1}}"

KEY=$(printf '%s|%s' "$SECRET_NAME" "$REGION" | shasum -a 256 2>/dev/null | cut -c1-16 \
      || printf '%s|%s' "$SECRET_NAME" "$REGION" | sha256sum | cut -c1-16)
CACHE_FILE="/tmp/.media-toolkit-token.$KEY.json"
EXPIRY_FILE="/tmp/.media-toolkit-token.$KEY.expiry"

if [ "${GATEWAY_TOKEN_NO_CACHE:-0}" != "1" ] \
   && [ -f "$CACHE_FILE" ] && [ -f "$EXPIRY_FILE" ]; then
  NOW=$(date +%s)
  DEADLINE=$(cat "$EXPIRY_FILE" 2>/dev/null || echo 0)
  if [ "$NOW" -lt "$DEADLINE" ]; then
    cat "$CACHE_FILE"
    exit 0
  fi
fi

CREDS=$(aws secretsmanager get-secret-value \
  --secret-id "$SECRET_NAME" \
  --region "$REGION" \
  --query SecretString --output text)

TOKEN_ENDPOINT=$(echo "$CREDS" | jq -r .tokenEndpoint)
CLIENT_ID=$(echo "$CREDS" | jq -r .clientId)
CLIENT_SECRET=$(echo "$CREDS" | jq -r .clientSecret)
SCOPE=$(echo "$CREDS" | jq -r .scope)

RESP=$(curl -s -X POST "$TOKEN_ENDPOINT" \
  -u "${CLIENT_ID}:${CLIENT_SECRET}" \
  -d "grant_type=client_credentials" \
  --data-urlencode "scope=${SCOPE}")
TOKEN=$(echo "$RESP" | jq -r .access_token)
EXPIRES_IN=$(echo "$RESP" | jq -r '.expires_in // 3600')

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
  echo "get_gateway_token.sh: failed to obtain access token" >&2
  exit 1
fi

# refresh 10 minutes before the token's own expiry (floor: 60 s)
MARGIN=600
[ "$EXPIRES_IN" -le "$MARGIN" ] && MARGIN=$(( EXPIRES_IN > 120 ? 60 : 0 ))
echo $(( $(date +%s) + EXPIRES_IN - MARGIN )) > "$EXPIRY_FILE"

umask 077
jq -n --arg t "$TOKEN" '{"Authorization": ("Bearer " + $t)}' | tee "$CACHE_FILE"
