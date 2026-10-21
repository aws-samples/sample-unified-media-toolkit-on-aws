# get_gateway_token.ps1 — PowerShell twin of get_gateway_token.sh.
# Headers helper for the media-toolkit MCP endpoint (Claude Code
# headersHelper / Codex http_headers_helper contract): fetches Cognito
# M2M credentials from Secrets Manager via the AWS CLI, exchanges them
# for an access token via client_credentials, and prints
# {"Authorization":"Bearer ..."} on stdout.
#
# Behavior-identical to the bash version (keep the two in sync):
#   - default precedence: env var > kit.env > the legacy well-known name
#   - cache keyed PER DEPLOYMENT (secret name + region), TTL from the
#     token response's own expires_in minus a 10-minute margin
#   - GATEWAY_TOKEN_NO_CACHE=1 forces a fresh mint
# Targets Windows PowerShell 5.1 (preinstalled on Windows 10/11); also
# runs on PowerShell 7+. Requires AWS CLI v2 on PATH.

$ErrorActionPreference = "Stop"
# Windows PowerShell 5.1 may default to TLS 1.0 — the token endpoint
# requires 1.2+
[Net.ServicePointManager]::SecurityProtocol = `
    [Net.ServicePointManager]::SecurityProtocol -bor `
    [Net.SecurityProtocolType]::Tls12

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

# kit.env is bash syntax (KEY="value"); parse the two keys we honor
$KitSecret = $null; $KitRegion = $null
$KitEnv = Join-Path (Split-Path -Parent $Here) "kit.env"
if (Test-Path $KitEnv) {
    foreach ($line in Get-Content $KitEnv) {
        if ($line -match '^\s*KIT_SECRET_NAME\s*=\s*"?([^"#]+)"?') {
            $KitSecret = $Matches[1].Trim()
        }
        if ($line -match '^\s*KIT_REGION\s*=\s*"?([^"#]+)"?') {
            $KitRegion = $Matches[1].Trim()
        }
    }
}
$SecretName = if ($env:GATEWAY_SECRET_NAME) { $env:GATEWAY_SECRET_NAME }
              elseif ($KitSecret) { $KitSecret }
              else { "/media-toolkit-gateway/dev/gateway-credentials" }
$Region = if ($env:AWS_REGION) { $env:AWS_REGION }
          elseif ($KitRegion) { $KitRegion }
          else { "eu-west-1" }

# per-deployment cache key: first 16 hex of SHA256("secret|region")
$sha = [System.Security.Cryptography.SHA256]::Create()
$bytes = [Text.Encoding]::UTF8.GetBytes("$SecretName|$Region")
$Key = ([BitConverter]::ToString($sha.ComputeHash($bytes)) `
        -replace "-", "").ToLower().Substring(0, 16)
$Tmp = [IO.Path]::GetTempPath()
$CacheFile = Join-Path $Tmp ".media-toolkit-token.$Key.json"
$ExpiryFile = Join-Path $Tmp ".media-toolkit-token.$Key.expiry"

$NowEpoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

if ($env:GATEWAY_TOKEN_NO_CACHE -ne "1" -and
    (Test-Path $CacheFile) -and (Test-Path $ExpiryFile)) {
    $Deadline = 0
    [void][int]::TryParse((Get-Content $ExpiryFile -Raw).Trim(), [ref]$Deadline)
    if ($NowEpoch -lt $Deadline) {
        Get-Content $CacheFile -Raw | Write-Output
        exit 0
    }
}

$CredsJson = aws secretsmanager get-secret-value `
    --secret-id $SecretName --region $Region `
    --query SecretString --output text
if ($LASTEXITCODE -ne 0 -or -not $CredsJson) {
    Write-Error ("get_gateway_token.ps1: cannot read the credentials " +
        "secret '$SecretName' in $Region")
    exit 1
}
$Creds = $CredsJson | ConvertFrom-Json

$Basic = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(
    "$($Creds.clientId):$($Creds.clientSecret)"))
try {
    $Resp = Invoke-RestMethod -Method Post -Uri $Creds.tokenEndpoint `
        -Headers @{ Authorization = "Basic $Basic" } `
        -ContentType "application/x-www-form-urlencoded" `
        -Body @{ grant_type = "client_credentials"; scope = $Creds.scope }
} catch {
    Write-Error "get_gateway_token.ps1: token endpoint request failed: $_"
    exit 1
}
if (-not $Resp.access_token) {
    Write-Error "get_gateway_token.ps1: failed to obtain access token"
    exit 1
}
$ExpiresIn = if ($Resp.expires_in) { [int]$Resp.expires_in } else { 3600 }

# refresh 10 minutes before the token's own expiry (floor: 60 s)
$Margin = 600
if ($ExpiresIn -le $Margin) {
    $Margin = if ($ExpiresIn -gt 120) { 60 } else { 0 }
}
Set-Content -Path $ExpiryFile -Value ($NowEpoch + $ExpiresIn - $Margin)

$HeadersJson = @{ Authorization = "Bearer $($Resp.access_token)" } |
    ConvertTo-Json -Compress
Set-Content -Path $CacheFile -Value $HeadersJson
Write-Output $HeadersJson
