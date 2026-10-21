# Media Toolkit Gateway

The AgentCore Gateway stack of the Unified Media Toolkit: it exposes every
media micro-workflow (media processing, speech diarization, shot
segmentation, face recognition, cast resolution, artwork selection, audio
description, hybrid search, and more) as **one MCP endpoint** — an Amazon
Bedrock AgentCore Gateway. Any MCP-capable agent (Claude Code, Codex CLI,
Kiro, or your own) connects to it with an OAuth bearer token and gets every
workflow as callable tools.

There is no Docker image and no application compute here — all workflow
compute lives in the micro-workflow stacks this gateway wraps. The stack's
only build artifact is [gateway_tool/](gateway_tool/), a flat Python zip
holding its five Lambda handlers plus the nested client kit
(`client_kit.zip` — skills + configure/token scripts + this README,
assembled at build time and published to the proxy bucket's `_toolkit/`
prefix by the KitPublisher custom resource on every deploy).

Every workflow target follows the same three-tool contract:

| tool | what it does |
|---|---|
| `<target>___<start_tool>` | starts a run, returns a **handle** (`job_id`, `workflow_id`, `output_s3_prefix`) — never results |
| `<target>___get_job_status` | polls the handle; on success lists the outputs that exist |
| `<target>___describe_tool` | version, backend description, and the **output contract** (field-level meaning of every artifact) |

Runs are idempotent: resubmitting identical parameters is absorbed into the
existing run. Results land on S3 under
`micro-workflow/<target>/<workflow_id>/`; presign them for download via the
`media-outputs` target.

Document map: [connect an agent](#0-get-the-client-kit) (§0–7) ·
[architecture](#architecture) · [design decisions](#design-decisions) ·
[build & deploy](#build--deploy) · [testing](#testing) ·
[gotchas](#gotchas).

---

## 0. Get the client kit

Every deployment publishes a **client kit** to its proxy bucket — the
skills, the connection scripts, and this guide — so you need no access to
any source repository. With your own AWS credentials:

```bash
aws s3 cp s3://<proxy-bucket>/_toolkit/client-kit.zip . --region <region>
unzip client-kit.zip -d media-toolkit-kit && cd media-toolkit-kit
```

Then the quick path is one command (details in sections 1–5; the script
does nothing you couldn't do by hand):

```bash
./scripts/configure.sh --agent all
```

On **Windows**, the same one command in PowerShell — Windows PowerShell
5.1, preinstalled on Windows 10/11, works as-is (PowerShell 7 works too).
The only prerequisite is the
[AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html):

```powershell
Expand-Archive client-kit.zip -DestinationPath media-toolkit-kit
cd media-toolkit-kit
.\scripts\configure.ps1 -Agent all
```

`configure.ps1` is the behavior-identical PowerShell twin of
`configure.sh` (pass `-SecretName`/`-Region` the same way; Kiro's re-mint
is `-RefreshToken`), with two Windows substitutions: the auth helper is
registered as a full `powershell.exe -NoProfile -ExecutionPolicy Bypass
-File ...` invocation, and skills are COPIED rather than symlinked
(symlinks need Developer Mode) — re-run the script after a kit update to
refresh them.

A **published kit** (downloaded from `_toolkit/client-kit.zip`) already
knows its own deployment: the KitPublisher writes `kit.env` — the
deployment's secret path and region — into the zip at publish, and the
scripts default to it (this is what makes a derived GatewayName workable).
Pass `--secret-name`/`--region` only when targeting a *different* gateway,
or when running from a repo checkout (which has no `kit.env`):

```bash
./scripts/configure.sh --agent all \
  --secret-name "$(aws cloudformation describe-stacks --stack-name <bundle> \
      --query "Stacks[0].Outputs[?OutputKey=='GatewayCredentialsSecretName'].OutputValue" \
      --output text)" \
  --region <region>
```

`configure.sh` reads the credentials secret to discover the gateway
endpoint, connects Claude Code (`.mcp.json`, auto-refreshing auth), Codex
CLI (`~/.codex/config.toml`, auto-refreshing auth), and/or Kiro
(`~/.kiro/settings/mcp.json`, static 12 h token —
`./scripts/configure.sh --refresh-token` re-mints it), and symlinks the
skills into each agent's skills directory. Merges touch only the
`media-toolkit` server entry, back up the file first, and never write the
client secret anywhere.

Install the `skills/` folder **whole** — the skills cross-reference each
other (the per-workflow references live once, under
`skills/video-processing/references/`, and the other skills link to them
by relative path), so cherry-picking a single skill directory leaves its
reference links dangling.

One naming note: **`media-toolkit`** is the client-side MCP server label
used everywhere below; the deployment-side names (the secret path, the
Cognito domain) are infrastructure identity — don't rename those to match.

## 1. Get the connection credentials from Secrets Manager

The deployment writes everything a client needs into **one Secrets Manager
secret** (populated automatically at stack create):

```
/<GatewayName>/<environment>/gateway-credentials
```

`<GatewayName>` is whatever the deployment pinned — or, when the bundle
was deployed with the parameter left empty, the derived
`{prefix}-{stack-id}` form. Read the exact name from the stack output
`GatewayCredentialsSecretName` (a published kit already carries it in
`kit.env`). The examples below use the default gateway name.

Download it with AWS credentials that are allowed
`secretsmanager:GetSecretValue` on that secret:

```bash
aws secretsmanager get-secret-value \
  --secret-id /media-toolkit-gateway/dev/gateway-credentials \
  --region <region> \
  --query SecretString --output text > ~/.media-gateway-credentials.json
chmod 600 ~/.media-gateway-credentials.json
```

The secret is a JSON bundle with five fields:

```json
{
  "endpointUrl":   "https://<gateway-id>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp",
  "tokenEndpoint": "https://<domain>.auth.<region>.amazoncognito.com/oauth2/token",
  "clientId":      "<cognito app client id>",
  "clientSecret":  "<cognito app client secret>",
  "scope":         "<resource-server>/invoke"
}
```

**Handle it like a password.** Keep the file outside any git repository
with `600` permissions, never paste the `clientSecret` into configs, chat
sessions, or logs. If it leaks, rotate: regenerate the Cognito app client
secret and re-run the deployment's SecretsPopulator.

## 2. Get an access token

Exchange the client credentials for a bearer token (`client_credentials`
grant). Tokens are valid for **12 hours**.

```bash
CREDS=$(cat ~/.media-gateway-credentials.json)
curl -s -X POST "$(echo "$CREDS" | jq -r .tokenEndpoint)" \
  -u "$(echo "$CREDS" | jq -r .clientId):$(echo "$CREDS" | jq -r .clientSecret)" \
  -d "grant_type=client_credentials" \
  --data-urlencode "scope=$(echo "$CREDS" | jq -r .scope)" \
  | jq -r .access_token
```

You rarely need to do this by hand:
[`scripts/get_gateway_token.sh`](scripts/get_gateway_token.sh) does the
whole dance — reads the secret, exchanges it, caches the token in `/tmp`,
and prints `{"Authorization": "Bearer ..."}` in exactly the shape Claude
Code's MCP client expects. It honors two environment variables:
`GATEWAY_SECRET_NAME` (defaults to the dev secret name above) and
`AWS_REGION`.

On Windows,
[`scripts/get_gateway_token.ps1`](scripts/get_gateway_token.ps1) is the
identical PowerShell twin (same env variables, same per-deployment
`%TEMP%` cache, same `{"Authorization": "Bearer ..."}` output).

## 3. Connect Claude Code

Add the gateway as an HTTP MCP server with the token helper. In your
project's `.mcp.json` (project scope — safe to commit, it contains no
secrets, only the endpoint and a helper path):

```json
{
  "mcpServers": {
    "media-toolkit": {
      "type": "http",
      "url": "https://<gateway-id>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp",
      "headersHelper": "/path/to/media-toolkit-gateway/scripts/get_gateway_token.sh"
    }
  }
}
```

The `url` is the secret's `endpointUrl`. `headersHelper` names a script
Claude Code runs fresh on every (re)connect — it must print a JSON object
of header key/values on stdout within 10 seconds.
[`scripts/get_gateway_token.sh`](scripts/get_gateway_token.sh) satisfies
that contract and caches the token locally, so reconnects are instant and
expiry is handled for you.

The first time Claude Code sees a project-scoped server it asks you to
approve it (one-time, per project). Alternatively, register it for
yourself only with a static token (no helper flag exists on the CLI):

```bash
claude mcp add --scope user --transport http media-toolkit \
  "https://<gateway-id>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp" \
  --header "Authorization: Bearer $(bash scripts/get_gateway_token.sh | jq -r .Authorization | cut -d' ' -f2)"
```

(Static headers freeze the token — after 12 hours re-add or switch to the
`headersHelper` form, which is the recommended one.)

Verify: `claude mcp list` should show the server connected, and `/mcp`
inside a session lists tools like `speech-diarization___diarize_speech`,
`artwork-selection___select_artwork`,
`media-outputs___get_download_urls`. A quick smoke prompt:

> Using the media-toolkit tools, call
> `artwork-selection___describe_tool` and summarize its output contract.

**Windows:** `.\scripts\configure.ps1 -Agent claude` writes the same
`.mcp.json`, with `headersHelper` set to the full PowerShell invocation —
a bare `.ps1` path would not execute from the shell Claude Code spawns.
Approve the project-scoped server once on first launch.

## 4. Connect Codex CLI

Codex supports remote MCP servers natively, with the same dynamic-auth
quality as Claude Code: `http_headers_helper` names a local command
returning JSON headers, re-invoked on 401/403. In `~/.codex/config.toml`:

```toml
[mcp_servers.media-toolkit]
url = "https://<gateway-id>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp"
http_headers_helper = "/path/to/kit/scripts/gateway_headers.sh"
```

(`gateway_headers.sh` is the wrapper `configure.sh` generates with your
deployment's secret name baked in; pointing at `get_gateway_token.sh`
directly also works when the default secret name is right.)

Codex also reads the same skills natively — it implements the open Agent
Skills standard and follows symlinks — from `.agents/skills/` (repo scope)
or `~/.agents/skills/` (user scope).

**Windows:** `.\scripts\configure.ps1 -Agent codex` writes the same
managed block into `%USERPROFILE%\.codex\config.toml`, with
`http_headers_helper` carrying the full PowerShell invocation.

## 5. Connect Kiro

Kiro supports remote MCP servers natively. In `.kiro/settings/mcp.json`
(workspace) or `~/.kiro/settings/mcp.json` (user; workspace wins when
both define the same server):

```json
{
  "mcpServers": {
    "media-toolkit": {
      "url": "https://<gateway-id>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp",
      "headers": {
        "Authorization": "Bearer <token from step 2>"
      },
      "disabled": false,
      "autoApprove": [
        "artwork-selection___get_job_status",
        "artwork-selection___describe_tool"
      ]
    }
  }
}
```

`./scripts/configure.sh --agent kiro` (run from the project you use Kiro
in) writes the **workspace** file — workspace merges over user level, so
the entry can never be shadowed, and it sits beside the skills the same
script installs. Kiro hot-reloads the file on save; no restart needed.

Three cautions:

- The `headers` value is a **static token** (Kiro has no helper-script
  hook), so it expires after 12 hours — re-mint with
  `./scripts/configure.sh --refresh-token` from the same project (or
  `bash scripts/get_gateway_token.sh | jq -r .Authorization` and paste).
- The workspace file now **contains a bearer token — it must be
  gitignored** (`.kiro/`). The script warns when it is not. If your
  project must commit a token-free `.kiro/settings/mcp.json`, put the
  tokened entry in the user-level file instead and keep the committed
  copy's entry disabled.
- `autoApprove` is optional and defaults to `[]` (confirm each call).
  List read-only families (`get_job_status` / `describe_tool`) to skip
  prompts, or `"*"` to approve everything — but never `"*"` against a
  gateway with the admin targets attached.

**Windows:** `.\scripts\configure.ps1 -Agent kiro` writes the same
workspace `.kiro\settings\mcp.json` with the static 12 h token; re-mint
with `.\scripts\configure.ps1 -RefreshToken` from the project.

## 6. Skills and prompts — the three layers

Agents work best when the workflow recipes live in files the agent loads
on demand, instead of being re-explained per session. This repo ships
them in three layers — one source of truth per layer; downstream
harnesses get *packaging*, never content forks:

| Layer | Lives at | Owns | 1:1 with |
|---|---|---|---|
| **Capability** | [prompts/](prompts/) | What each tool does, parameters, output labels, idempotency — claims pinned to code | gateway target |
| **Routing** | [skills/video-processing/](skills/video-processing/) | Intent→tool table, cross-tool rules, default async lifecycle | the whole gateway |
| **Playbook** | [skills/](skills/) (analyze-video, create-audio-description, …) | Multi-workflow composition: pipeline graphs, joins, state files | a use case |

Rules that keep it one source of truth:

- `skills/video-processing/references` is a **symlink to `../../prompts`**
  — never a copy. The client kit zip dereferences it at build time
  (guarded in `build_client_kit_zip`), so downstream harnesses receive
  real files synchronized by deploy, not by hand.
- Every gateway target must appear in the routing skill — a bundle test
  (`test_routing_skill_covers_every_gateway_target`) pins it, so
  registering a new micro-workflow fails the suite until its router row
  exists.
- Purpose-built harnesses override ONLY the router's fenced "Harness
  lifecycle (default)" section (and may exclude targets their catalog
  lacks); the intent table, references, and cross-tool rules are
  canonical and must not be edited downstream.
- Playbook skills state composition knowledge only; per-tool facts belong
  in the capability prompt the playbook references. When editing a
  playbook that restates tool facts, move them down a layer rather than
  updating the copy.
- All prompts/ and skills/ files carry agentskills.io-style frontmatter
  (`name`, `description`, `metadata.version`); prompt files additionally
  mark `metadata.layer: workflow-reference`.
- Known gap: `artwork-selection` has no capability prompt yet — the
  router row points at the `select-artworks` playbook until one is
  written.

The playbook skills:

| skill | what it does |
|---|---|
| `list-ingest-files` | browse the ingest bucket as a table |
| `inspect-video` | probe + summarize one source file |
| `analyze-video` | run the full workflow chain on an asset, persist a state file the other skills resolve |
| `visualize-metadata` | build the interactive review page (episodic or news layout) from a frozen template |
| `select-artworks` | ranked artwork candidate slates per archetype, editorial brief honored |
| `create-audio-description` | AD script + narration + mix + the AD review bench page |
| `export-package` | FCPXML editorial timeline package for NLEs |
| `review-compliance` | content-compliance evidence page: visual/spoken/word-list findings on a synced timeline — evidence, never a verdict |

### Install

From your project root (pick copy or symlink; symlinks track updates to a
checked-out toolkit repo):

```bash
# Claude Code
mkdir -p .claude/skills
ln -s /path/to/media-toolkit-gateway/skills/* .claude/skills/

# Kiro
mkdir -p .kiro/skills
ln -s /path/to/media-toolkit-gateway/skills/* .kiro/skills/
```

Claude Code invokes them as `/analyze-video <asset>`; Kiro picks them up
from its skills directory. Skills chain through a shared convention:
`analyze-video` writes `outputs/analyze-video/{ASSET_ID}.state.json`, and
the downstream skills resolve their run prefixes from it.

### Conventions the shipped skills follow (and yours should too)

- **Always poll with the returned `job_id`** — start tools return
  handles, never results.
- **Read `describe_tool` once per target** when writing a new skill: the
  output contract documents every artifact field.
- **Reuse `idempotency_token`** when a skill may be re-run on the same
  inputs: the same token + parameters returns the existing run instead of
  paying twice.
- **Review pages are template-first**: the HTML-producing skills ship
  frozen page templates under `references/` and only inject data — if a
  page needs to change, edit the template deliberately, never restyle per
  run.
- **Never embed credentials in a skill file** — connection setup
  (sections 1–5) is per-machine; skill files get committed.

### Prompt files

[prompts/](prompts/) holds one capability prompt per workflow target
(`audio_stem_agent.md`, `speech_diarization_agent.md`, …
`media_uploads_agent.md`). These belong in the agent's reasoning context,
not in the tool schema. The micro-workflow prompts cover the shared
response contract (`workflow_id`, idempotent resubmit, `outputs` list,
the `media-outputs` download-URL tools) plus per-tool parameter
heuristics and output-label tables.

**Three prompts deliberately do NOT describe the shared `outputs[]`
contract, because their targets don't implement it.**
`prompt_registry_agent.md` goes further — its target has no JOBS at all,
no workflow_id/poll/outputs anywhere, and its prompt opens by saying so.
`hybrid_search_agent.md` — hybrid-search jobs return
`prepared`/`skipped`/`indexed` counts, never an `outputs[]` list, and its
results are consumed inline from `search_content`. `tts_agent.md` —
text-to-speech returns a per-transcript `outputs[]` keyed by
`transcript_index` with `qc_passed`/`qc_similarity` and no
`{uri, mime, label}` triples, *or* a bare `voice` record for
`register_voice`. Both open by saying so explicitly: an agent that
carries the shared contract as an assumption will hunt for a `label` that
does not exist. `tts_agent.md` states the shape difference before
anything else for that reason, and it is the one thing to preserve if the
file is ever rewritten. (Why tts is not conformed: see
[design decisions](#why-text-to-speechs-outputs-is-not-conformed).)

`tts_agent.md` carries three further facts the tool schema cannot
express:

- **Batch size is not the schema's hard cap.** The prompt carries a
  payload table with two columns, per-batch and whole-request, because
  the two bounds are different in kind: the per-batch one applies to each
  call on its own, while the Map-aggregate one is the sum over every
  transcript in the request and so splitting cannot relieve it. The hard
  cap of 100 binds first only for very short texts; real dialogue
  batches hit the payload bounds well before or after it depending on
  line length.
- **`target_durations_s` must not be derived from a source speaker's
  timings** just because a diarized transcript has them — the ±10 %
  stretch bound converts good clips into failed entries. The prompt says
  this at the parameter *and* again in the dub-one-speaker chain, since
  that chain is exactly where the timings are lying around.
- **QC failure is a review pointer, not an error**, and the correct
  response to `all_passed: false` is *not* to re-synthesize the batch.
  The QC transcriber is non-deterministic on identical audio, so an
  auto-retry pays for everything again to move a noisy score.

---

## Architecture

### What the stack creates

```
Cognito User Pool (M2M only, client_credentials flow)
├─ Resource server       → defines the <OAuthScope>/invoke scope
├─ App client            → issues access tokens (client secret stored in Secrets Manager)
└─ User pool domain      → exposes the /oauth2/token endpoint

Secrets Manager secret   → credentials bundle { endpointUrl, tokenEndpoint,
                           clientId, clientSecret, scope }
  └─ SecretsPopulator    → custom resource Lambda that writes the bundle on Create/Update

Client kit               → _toolkit/client-kit[-{version}].zip on the proxy bucket
  └─ KitPublisher        → custom resource Lambda (same zip) that publishes the kit
                           baked into its own package on Create/Update

AgentCore Gateway        → CUSTOM_JWT auth, MCP protocol
├─ GatewayExecutionRole  → IAM role used by the gateway to invoke tool Lambdas
├─ one GatewayTarget per workflow (most optional — empty ARN parameter = omitted)
│    audio-stem, speech-enhancement, speech-diarization, media-processing,
│    shot-segmentation, face-recognition, hybrid-search, hybrid-search-admin,
│    text-to-speech, bedrock-inference, prompt-registry, prompt-registry-admin,
│    conversation-analysis, shot-metadata-enrichment, program-structure-analysis,
│    audio-description, cast-resolution, artwork-selection
├─ MediaOutputsTarget       → target: media-outputs (get_download_url[s])
│    └─ DownloadUrlFunction → Lambda created in this stack (gateway_tool zip);
│                             presigns GET URLs for keys under micro-workflow/
├─ MediaIngestTarget        → target: media-ingest (list_files; optional)
│    └─ IngestListFunction  → Lambda created in this stack (gateway_tool zip);
│                             lists source media in the ingest bucket
└─ MediaUploadsTarget       → target: media-uploads (start / part-urls /
     │                        complete / abort; optional — same IngestBucket
     │                        gate as media-ingest)
     └─ UploadFunction      → Lambda created in this stack (gateway_tool zip);
                              presigns single/multipart uploads INTO the ingest
                              bucket (control plane only — bytes go client-to-S3)
```

### Auth flow

The gateway uses `CUSTOM_JWT` authorisation backed by the Cognito User
Pool above. Callers obtain a token via the OAuth 2.0 `client_credentials`
flow:

```
POST <tokenEndpoint>
Authorization: Basic base64(<clientId>:<clientSecret>)
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&scope=<OAuthScope>/invoke
```

The resulting bearer token is passed as `Authorization: Bearer <token>` on
every MCP request. Cognito M2M access tokens carry a `scope` claim but no
`aud` claim — the gateway is configured with `AllowedScopes` (not
`AllowedAudience`) for this reason.

The credentials bundle is written to Secrets Manager by
`SecretsPopulatorFunction` after the gateway and Cognito client are both
ready (see [§1](#1-get-the-connection-credentials-from-secrets-manager)).

### Gateway targets and tools

Tool names over MCP use the `{target}___{tool}` convention. Most workflow
targets have exactly the three contract tools (one submit,
`get_job_status`, `describe_tool`):

| target | submit tool | notes |
|---|---|---|
| audio-stem | `separate_audio_stems` | Bandit v2 stem separation |
| speech-enhancement | `enhance_speech` | DeepFilterNet3 in-place dialogue noise suppression — a listening deliverable. The current model measured worse than the mix as a transcription input, so the mix stays the validated ASR/diarization default (enhanced not recommended there today) |
| speech-diarization | `diarize_speech` | Whisper + speaker clustering |
| shot-segmentation | `segment_shots` | DINOv2 shot segmentation |
| conversation-analysis | `analyze_conversation` | transcript → topic segments; consumes prompt-registry + bedrock-inference internally |
| shot-metadata-enrichment | `enrich_shot_metadata` | shot run → per-shot metadata + scenes; batch by default |
| program-structure-analysis | `analyze_program_structure` | program structure over sibling runs |
| audio-description | `generate_ad` | AD script generation from completed sibling runs (4 required + a people source: face-recognition or cast-resolution) + a tts voice_id. Renders no audio — the render-manifest output chains into media-processing `ad_mix` |
| cast-resolution | `resolve_cast` | diarization SPEAKER_IDs + face ids + on-screen text → named person entities with evidence; its boolean flags default true and exist to REMOVE cost |
| artwork-selection | `select_artwork` | ranked artwork candidate frames, optional editorial brief |

The irregular targets:

- **media-processing** (5 tools) — three submit shapes:
  `media-processing___process_media` (ffmpeg transcode/extract over ONE
  asset), `media-processing___quick_edit` (compose one output video from
  an inline cut list over 1+ sources), and
  `media-processing___capture_frames` (one still per ms-marker, snapped
  to each source's frame grid). Three input shapes, three tools, because
  `SchemaDefinition` has no `oneOf` and the family answer to a second
  shape is a second tool.
- **face-recognition** (11 tools) — `recognize_faces` + poll/describe,
  plus collection management: `create_collection`, `list_collections`,
  `list_faces`, `get_face` (the late-binding live-name resolver,
  tombstone-aware), `batch_update_faces` (synchronous, DDB-only),
  `batch_index_faces` (async, ETag-idempotent), `batch_delete_faces`
  (async, tombstoned), `delete_collection` (default collection requires
  `force=true`).
- **hybrid-search** (6 tools) — `index_content` (async contract-driven
  indexing), `search_content` (sync hybrid search: SLM decompose →
  parallel ANN+FTS → weighted RRF; `@Title`/quoted/`says:`/scope
  operators; image search), `create_collection` (create-if-absent, never
  modifies existing), `list_collections`, poll, describe.
- **hybrid-search-admin** (3 tools, separate Lambda + IAM role — attach
  only for trusted admin agents) — `configure_search_backend` (edit
  retrieval weights/scopes/schema map), `delete_collection` (full sweep;
  'default' requires force), describe.
- **text-to-speech** (6 tools) — `synthesize_speech` (CosyVoice3
  zero-shot-cloning batch; optional pace/emotion style hint — English
  voices only), `register_voice` (ASR-validated reference WAV),
  `list_voice_collection`, `get_voice`, poll, describe.
- **bedrock-inference** (5 tools) — `run_inference` (N already-rendered
  prompt records through a model chain; sync under a lease, or a
  50%-cost batch job), poll (partial success is success — read
  `records_failed`), plus the registry introspection pair
  `list_models`/`get_model` (per-geo — call before the first submit),
  describe.
- **prompt-registry** (7 tools, all synchronous CRUD — no state machine,
  no jobs) — `put_prompt` (immutable versions; stage dev/test/prod),
  `get_prompt` (named-missing never falls back), `list_prompts`,
  `promote_prompt` (one-way ratchet dev→test→prod; idempotent),
  `delete_prompt` (prod refused), `deprecate_prompt` ('default'
  protected), `describe_tool` (version, inline output contract, and the
  WHOLE variable vocabulary; deliberately no state_machine_arn).
- **prompt-registry-admin** (7 tools — SUPERUSER) — the same seven tools
  backed by a separate Lambda running `SERVE_INTERNAL=true`: categories
  whose default version declares `Audience: workflow-internal` (matched
  prompt+parser pairs) are served HERE and invisible on the normal
  target — `list_prompts` omits them, every category-taking tool refuses
  by name. Workflow children also consume this Lambda's ARN directly
  (internal callers are trusted). Same posture as hybrid-search-admin:
  leave the parameter empty for gateways serving untrusted agents. A
  bundle test pins the two targets' InlinePayloads identical.
- **media-outputs / media-ingest / media-uploads** — the three targets
  whose Lambdas live in THIS stack; see
  [design decisions](#media-outputs-a-singularplural-pair-on-purpose).

| MCP tool name | Description |
|---------------|-------------|
| `media-outputs___get_download_url` | Presign one output object — `s3uri_or_key` is a single s3:// uri or bare key under `micro-workflow/` (or a tts voice preview under `_model_artefacts/voice_catalog/`); expiry clamped 60–3600 s. Returns a flat `{url, expires_at}` |
| `media-outputs___get_download_urls` | Presign 1–50 output objects in one call; expiry shared by the batch. Returns `{urls, count, expires_at}` |
| `media-ingest___list_files` | List source media in the ingest bucket — optional `prefix` (server-side) and `extension` (client-side, case-insensitive) filters; `max_results` default 20 clamped 1–200; `next_token` pagination. The returned `s3uri` feeds workflow submit tools directly |
| `media-uploads___start_upload` | Begin uploading ONE local file (≤ 5 GiB) to the ingest bucket. ≤ 64 MiB → `mode: "single"` with one presigned PUT url (Content-Type folded into the signature; echo it). Larger → `mode: "multipart"`: `upload_id` + the COMPLETE presigned part-URL set. MIME auto-derived from the key extension; existing keys refused unless `overwrite: true` |
| `media-uploads___get_upload_part_urls` | Re-presign named parts (≤ 50 per call) — the HTTP-403 refresh tool, never pagination |
| `media-uploads___complete_upload` | `CompleteMultipartUpload` from the client's collected `{part_number, etag}` list; the object exists only after this; response carries the `s3uri` plus a ListBucket-verified size |
| `media-uploads___abort_upload` | Drop an abandoned multipart session (incomplete parts bill until removed) |

### The shared response contract

All workflow targets' submit and `get_job_status` responses include
`workflow_id` and `output_s3_prefix`. On SUCCEEDED, the media-asset
workflows' `get_job_status` returns an `outputs` list of
`{uri, mime, label, desc}` entries — plain `s3://` URIs; entries whose
uri ends with `/` are prefixes and carry no mime; `desc` is a one-line
role description joined from the stack's
[gateway_tool/output_contract.json](gateway_tool/output_contract.json)
(omitted, never fabricated, when the contract lacks the label).

Labels and output basenames are **stable API** (kebab-case label grammar,
pinned by the bundle). Any rename must ship WITH a poll-time alias in the
workflow's gateway so pre-rename runs and their idempotent resubmits
still answer in the current vocabulary — alias-on-read is the price of a
rename, never optional.

The `media-outputs` tools are the only way to obtain HTTP access —
presigned URLs are never inlined in status responses (~1.3 KB ≈ 350
context tokens each).

`describe_tool` returns
`{version, state_machine_arn, description, output_contract}`
(text-to-speech's returns two state machine ARNs instead of one) —
introspection only, deliberately without the full state machine
definition (token cost, internals leakage). `output_contract` is the
field-level contract for the workflow's outputs, keyed by label —
basename, kind, mime, the one-line description that doubles as `desc`,
and a compact example of each file's field names/nesting. It ships in
each stack's gateway zip, so it is static per `TOOL_VERSION` and CI can
pin it without running a job; anything not listed in it (ephemeral
intermediates, per-chunk scratch) is internal and may change without
notice. text-to-speech and hybrid-search carry `kind: "inline"`
contracts documenting their inline result shapes instead of files.

Two targets deliberately diverge from the shared `outputs[]` shape (the
test harness exercises them in a reduced form for the same reason):

- **text-to-speech** — `get_job_status` returns either a `voice` record
  or a per-transcript `outputs` list depending on which tool started the
  job, and only the synthesis shape carries `output_s3_prefix` — a
  `voice` record is a catalog entry, not a run with an output location.
- **hybrid-search** — `index_content` jobs return
  `prepared`/`skipped`/`indexed` counts, never an `outputs[]` list.

### The three ids

Each layer has exactly one id and it means exactly one thing. (An earlier
spelling used `uuid` for all three, which is what made every echo bug
possible.)

| Layer | Key | Meaning | Lifetime |
|---|---|---|---|
| Gateway **input** | `idempotency_token` | The caller's assertion that this request is the same request as a previous one. Optional; 1-80 chars of `A-Z a-z 0-9 _ . -`. | Folded into the hash at the gateway and **discarded** — never forwarded to a state machine, never stored, never echoed in any response. |
| Gateway **output** | `workflow_id` | The service's name for the run. | The caller's handle for polling and for locating outputs at `micro-workflow/{target}/{workflow_id}/`. |
| SFN **input** / state `event` | `uuid` | *This execution's local identifier.* | Travels through every state; equals the execution name when the gateway started it. |

The event key deliberately keeps the name `uuid` and must not be renamed:
every state derives its output prefix from it and several reconstruct the
execution ARN from it. A caller never sees `uuid` in a response, and no
tool accepts it as input: the start tools reject it
(`INVALID_PARAMETER`), and every other tool accepts it with a logged
warning so a hallucinated parameter cannot lose a poll mid-job.

Tool schemas (InputSchema / OutputSchema) are defined as native YAML
`InlinePayload` lists in [cfnstack.yaml](cfnstack.yaml).
`SchemaDefinition` only supports `Type`, `Description`, `Properties`,
`Required`, and `Items` — no `enum`, `minimum`, `maximum`, etc.

### Parameters

| Parameter | Default | Required | Purpose |
|-----------|---------|----------|---------|
| `AudioStemToolLambdaArn` | — | yes | `GatewayToolLambdaArn` output from the audiostem-separation-on-aws stack |
| `SpeechDiarizationToolLambdaArn` | — | yes | `GatewayToolLambdaArn` output from the speech-diarization-on-aws stack |
| `MediaProcessingToolLambdaArn` | — | yes | `GatewayToolLambdaArn` output from the media-processing stack |
| `ShotSegmentationToolLambdaArn` | `""` | no | shot-segmentation-on-aws; empty omits the target |
| `FaceRecognitionToolLambdaArn` | `""` | no | face-recognition-on-aws; empty omits the target |
| `HybridSearchToolLambdaArn` | `""` | no | hybrid-search; empty omits the target |
| `HybridSearchAdminToolLambdaArn` | `""` | no | hybrid-search `AdminToolLambdaArn`; empty omits the admin target (recommended for gateways serving untrusted agents) |
| `TextToSpeechToolLambdaArn` | `""` | no | tts-on-aws; empty omits the target |
| `BedrockInferenceToolLambdaArn` | `""` | no | bedrock-inference-on-aws; empty omits the target |
| `PromptRegistryToolLambdaArn` | `""` | no | prompt-registry-on-aws; empty omits the target |
| `PromptRegistryAdminToolLambdaArn` | `""` | no | prompt-registry-on-aws `AdminToolLambdaArn` — the SUPERUSER surface; empty omits it (recommended for gateways serving untrusted agents) |
| `ConversationAnalysisToolLambdaArn` | `""` | no | conversation-analysis-on-aws; empty omits the target |
| `ShotMetadataEnrichmentToolLambdaArn` | `""` | no | shot-metadata-enrichment-on-aws; empty omits the target |
| `ProgramStructureAnalysisToolLambdaArn` | `""` | no | program-structure-analysis-on-aws; empty omits the target |
| `AudioDescriptionToolLambdaArn` | `""` | no | audio-description-on-aws; empty omits the target |
| `CastResolutionToolLambdaArn` | `""` | no | cast-resolution-on-aws; empty omits the target |
| `ArtworkSelectionToolLambdaArn` | `""` | no | artwork-selection-on-aws; empty omits the target |
| `SpeechEnhancementToolLambdaArn` | `""` | no | speech-enhancement-on-aws; empty omits the target |
| `ProxyBucket` | — | yes | S3 proxy bucket the micro-workflows write outputs to (same value passed to each micro-workflow stack); scopes the download-URL IAM policy to `micro-workflow/*` + `_model_artefacts/voice_catalog/*` (voice previews) |
| `IngestBucket` | `""` | no | S3 ingest bucket holding source media; `ListBucket` for `media-ingest___list_files` and write access for the media-uploads tools. Empty omits BOTH targets |
| `UploadPrefix` | `""` | no | Optional key prefix the media-uploads tools may write under (e.g. `uploads/`); empty allows the whole ingest bucket; must not start with `/`. Ignored when `IngestBucket` is empty |
| `GatewayName` | `media-toolkit-gateway` | no | Gateway name; also used as the Cognito domain prefix and secret path component |
| `Environment` | `dev` | no | `dev` \| `staging` \| `prod` |
| `OAuthScope` | `media-workflows` | no | Cognito resource server Identifier; full scope string is `<OAuthScope>/invoke` |
| `SecretsKmsKeyId` | `alias/aws/secretsmanager` | no | KMS key for the credentials secret; provide a CMK ARN for stricter control |

### Stack outputs

No outputs are **exported** — cross-stack values are passed explicitly as
parameters (`GatewayToolLambdaArn` outputs from the micro-workflow stacks
→ `*ToolLambdaArn` parameters here), never via `Fn::ImportValue`. Exports
hard-lock the producing stack: an exported value cannot change and the
stack cannot be deleted while any importer exists.

The outputs you will actually use: `GatewayUrl` (the MCP endpoint to pass
to agents), `GatewayCredentialsSecretName`/`...Arn` (the credentials
bundle), `GatewayIdentifier`/`GatewayArn`, `UserPoolId`,
`GatewayExecutionRoleArn`, one `*TargetId` per attached target
(conditional targets emit theirs only when attached), and the ARNs of the
three Lambdas created in this stack (`DownloadUrlFunctionArn`,
`IngestListFunctionArn`, `UploadFunctionArn`).

---

## Design decisions

### media-outputs: a singular/plural pair, on purpose

Both `get_download_url` and `get_download_urls` are served by the one
`DownloadUrlFunction` Lambda
([gateway_tool/download_url.py](gateway_tool/download_url.py)),
dispatched on the tool name; each takes exactly one input parameter and
ignores the other's.

| | `get_download_url` | `get_download_urls` |
|---|---|---|
| input | `s3uri_or_key` (required) | `uris` (required, 1–50) |
| response | `{url, expires_at}` | `{urls: [{uri, url}], count, expires_at}` |
| bad uri | whole request fails | fails only its own entry |

Batching was **added as a second tool rather than by widening the
first**, and that is the whole point of the pair. The obvious cheaper
move — teach `get_download_url` to accept `uris` and always return
`{urls: […]}` — was deployed briefly and reverted: a compatible *request*
parameter cannot rescue a caller that reads `result["url"]`, because the
break is in the *response*. Anything already written against the flat
shape breaks silently on a field that is simply gone. So
`get_download_url`'s InputSchema and OutputSchema are frozen at their
original bytes, and the plural tool carries the nested shape.

The costs of the pair are real and were accepted: a second ~2.4 KB schema
block in every `tools/list` for every agent, and a routing decision an
agent can get wrong. The prompts mitigate the latter by stating the rule
as "never call the singular tool in a loop."

In the plural tool an invalid uri fails **only its own entry**
(`error`/`message`), so one bad uri never discards the good presigns
beside it; `count` is the number that succeeded. Whole-request failures
in either tool (missing param, `> 50`, non-list `uris`, unparseable
`expiry_seconds`) return the flat `{error, errorCode, message}`.

The 50-uri cap is a **context budget, not a compute one** —
`generate_presigned_url` is local crypto with no API call, but a URL on
this bucket measures ~1.3 KB (the assumed-role session token in the query
string dominates), so a full batch is ~16 K tokens. That is why the agent
prompts tell agents to batch *and* to presign only what they need; the
two are not in tension.

### media-ingest: the inverse of media-outputs, deliberately separate

It reads the INGEST bucket (the caller's source media) where
media-outputs presigns the PROXY bucket (workflow outputs), and its
`IngestListFunction` Lambda
([gateway_tool/ingest_list.py](gateway_tool/ingest_list.py)) carries its
own role scoped to `s3:ListBucket` on the ingest bucket alone — no
`GetObject`, no presigning: workflows read ingest objects server-side by
the returned uri. The whole target is conditional on the `IngestBucket`
parameter (empty = omitted), like the optional workflow targets.

Two implementation choices worth knowing. **The default page is 20, and
that is a context budget** (the same reasoning as the 50-uri presign
cap): every row lands in the calling agent's context, and
prefix+extension filtering answers the typical "find my file" question
inside one page — `next_token` covers the rest. **`next_token` is the
last key scanned, resumed via `StartAfter`, not S3's ContinuationToken**
— extension filtering is client-side (S3 cannot match suffixes), so a
page can be cut mid-way by `max_results`, which a ContinuationToken
(always pointing past the whole page) cannot express without skipping
matches. A sparse filter over a huge bucket is bounded by a
10,000-keys-scanned cap per call; the token continues the scan.

### media-uploads: the write-direction sibling, control plane only

The bytes NEVER pass through the gateway: the client PUTs them straight
to S3 via the presigned URLs — which is what makes multipart the speed
mechanism (parallel part PUTs), not just a size mechanism.
`UploadFunction` ([gateway_tool/upload_tool.py](gateway_tool/upload_tool.py))
carries the third separate role: `s3:PutObject` (covers single PUT,
UploadPart, Create/CompleteMultipartUpload) + `s3:AbortMultipartUpload`
on `${IngestBucket}/${UploadPrefix}*` plus `ListBucket` for the
overwrite-existence check — still **no GetObject**. Gated by the same
`IngestBucket` condition as media-ingest; `UploadPrefix` optionally
confines writes to a key prefix.

Design points worth knowing:

- **Synchronous tools, no jobs**: no `workflow_id`, no idempotency token
  (ignored with a log line) — the multipart `upload_id` is the handle,
  and S3 happily holds several concurrent sessions for one key (the last
  `complete` wins).
- **The 5 GiB cap bounds every multipart plan to ≤ 50 parts**, so
  `start_upload` always returns the COMPLETE part-URL set in one
  response; `get_upload_part_urls` is the 403-refresh tool, never
  pagination.
- **A presigned URL can die before its nominal expiry** — it is signed
  with the Lambda's session credentials — so the uploader contract is
  "refresh on 403 via `get_upload_part_urls`", mirroring media-outputs'
  "re-request on 403".
- **Incomplete multipart uploads bill invisibly until aborted**: the
  ingest bucket (customer-owned, not in these templates) should carry an
  `AbortIncompleteMultipartUpload` lifecycle rule (~7 days) as the
  backstop behind `abort_upload` — a documented operator requirement, not
  something the stack can declare.
- **Malformed parameter TYPES are refused by the gateway's own schema
  layer, before the Lambda runs** (measured live) — a boolean
  `size_bytes`, or an omitted `key`, comes back as a
  `ValidationException` naming the field rather than the handler's
  `{error, errorCode, message}`; `upload_tool.py`'s own coercion is
  defence behind that, for any path that bypasses the schema. A probe
  asserting only on `errorCode` reads that correct refusal as a
  pass-through — accept either shape.
- **`abort_upload`'s `NO_SUCH_UPLOAD` means the session is GONE, not
  that the id was never valid** — a completed session, a lifecycle-swept
  one, and a phantom id all return it; on an abort retry read it as
  "already landed". S3 documents no abort idempotency, so nothing may
  rely on a re-abort succeeding.
- The `upload-media` skill carries the client-side composition (parallel
  workers, ETag collection, retry) with a stdlib-only helper. Per-file
  calls are deliberate — a batched `start_uploads` would multiply whole
  upload *sessions* (each up to 50 URLs) past the context budget, and
  the media-outputs singular/plural history says batching arrives later
  as a second tool if ever needed.

### Idempotency: service-derived output prefixes

Callers never pass `output_s3_prefix`; every workflow writes to
`s3://{ProxyBucket}/micro-workflow/{target-name}/{workflow_id}/`, where
`workflow_id = uuid5(canonical input + tool version + optional
idempotency_token)`. The token never names the execution — it folds into
the hash and is then discarded — so identical requests are idempotent
(resubmit returns the existing run's handle via ExecutionAlreadyExists)
while any parameter, token, or version change re-runs to a fresh prefix.
Each micro-workflow stack sets `TARGET_NAME` on its gateway Lambda — the
value must equal the GatewayTarget `Name` here.

**Omitting the token means two different things.** For the asset-shaped
targets the parameters name the asset (which video), so they fully
determine the output and omitting the token still dedups on the parameter
hash. For `face-recognition` and `text-to-speech` neither is tied to one
asset — face-recognition reads a persistent mutable face collection, tts
renders against a mutable voice catalog — so identical parameters can
legitimately need to run again, and **an absent token means "do not
deduplicate me"**: the gateway generates one.

**A submit is a handle, never results.** An absorbed resubmit used to
*describe* the existing run and hand back its full results, so a submit's
shape depended on **when** it was called. Two consequences, and the
second is why it changed: (1) an agent had to branch on `status` to know
which shape it held; (2) the declared `OutputSchema` could never match
the reachable field set, so "declared == reachable" was aspirational
rather than checkable. Now every submit returns the same fields either
way and only `status` differs; results live in `get_job_status` alone.

**`message` is the only signal a submit was absorbed, and comparing
`job_id`s cannot replace it.** The execution name is a hash of the same
input either way, so a fresh start of a run that has since aged out
returns a *byte-identical* ARN. A missing `message` means a second
execution was launched — the failure the whole contract exists to
prevent.

**`current_state` is the one deliberate asymmetry.** Present on a RUNNING
absorb, because "which stage is it in" is the caller's actual question
and worth a `GetExecutionHistory` paging pass; absent once the run has
stopped, because a stopped execution is in no state and returning
`"unknown"` would buy a non-answer with a paging pass. tts has it on no
path (two state machines, no helper). Pinned in both directions by a
bundle unit test.

**A terminal failure is sticky for ~90 days, and that number matters.**
Step Functions reserves an execution name for 90 days after the run
stops, so an identical resubmit keeps returning that same
FAILED/ABORTED/TIMED_OUT verdict for that whole window — long enough to
be permanent from any caller's point of view, but *not* forever, which is
why the schemas and prompts say 90 days rather than "forever": an agent
told a state is permanent has no reason to ever look again. Only a
changed parameter, a different token, or a new tool version re-runs it.
**A token is not a retry button** — it works only because it changes the
request's identity, so a plain retry with the same token is
indistinguishable from the original call and gets the original verdict.

**hybrid-search's `index_content` is deliberately never idempotent at the
execution layer** (its executions are named uuid4) — and here it *should*
be, because `index_content` is an accumulating write to a shared
collection. Two calls with different manifests must both run; collapsing
them on an input hash would silently drop the second. Its real guarantee
sits at the layer it belongs to, per-*document*:
`doc_id = md5(collection:asset:doc_type:start:end:chunk)` filters
duplicates before any model spend — a full resubmit of an already-indexed
manifest reports `prepared=0` with everything skipped.

**Retired contracts, kept retired.** A duplicate tts submit once answered
`{"status": "ALREADY_RUNNING"}` and nothing else — wrong twice over: the
caller got no results for work already paid for, and the status word was
a claim about the *run* that nothing had checked (a run that succeeded
hours earlier and one that FAILED both answered it). It described the
gateway's own bookkeeping (the uuid5 name is taken), not the job. The
shared `_start_execution` now `DescribeExecution`s the existing run and
returns the sibling handle shape; the one fact the status could no longer
carry — *this call did not resubmit* — moved to the `message` field.
Similarly, responses once echoed the caller-visible `correlation_uuid`;
the echo was removed entirely across all gateways rather than repaired
after a poll-path echo bug — `workflow_id` is now the only identity any
response carries, and a caller who supplied a token already has it.

### Why text-to-speech's `outputs[]` is not conformed

The question "can we make it conform" was asked and answered NO for the
shape, YES for everything else. The siblings' `label` is a **role**
vocabulary — one entry per distinct KIND of file (`transcript`,
`caption`, `speaker-manifest`, `usage`) — which is precisely what makes
"select by label, never parse filenames" a meaningful rule. tts's outputs
are N instances of ONE role (clip 0..N-1), so conforming would force
either a non-discriminating `label: "clip"` repeated N times (the rule
becomes vacuous) or dropping the per-clip `duration_s` / `stretched` /
`qc_passed` / `qc_similarity` / `error` that a triple has nowhere to
hold. `transcript_index` has no sibling analogue because **no sibling
takes an input array** — it is the join key back to the caller's
`transcripts[]`. Decisively, **speech-diarization already hit this and
answered the same way**: its per-speaker WAVs are deliberately not
enumerated as labeled entries (speaker ids can be non-contiguous) — it
emits one `speakers/` prefix entry plus a manifest, so the house answer
for N-of-a-kind files is prefix-plus-manifest, and tts's per-clip
`outputs[]` IS that manifest inlined.

What was NOT defensible, and was fixed: no `workflow_id` on the poll
(every sibling has it; an agent resuming with only a `job_id` could not
name the run), no `output_s3_prefix`, and an unreachable usage document —
now exposed as `usage_uri` since there is no label to hang it on. And
`register_voice`'s poll resolves `preview_uri` instead of returning the
raw S3 key that `get_voice` already resolved — one target was returning
one record in two shapes.

### Template and infrastructure notes

- **MCP tool names must stay ≤ 64 chars** — Bedrock rejects the entire
  ConverseStream on an oversize `{target}___{tool}` name. With the
  longest verb `separate_audio_stems` (20 chars) + `___` (3), **target
  names must be ≤ 41 chars**. `build-deploy-s3-dist.sh` enforces this at
  build time (`validate_tool_name_lengths`).
- **`CUSTOM_JWT` + `AllowedScopes`** — Cognito M2M tokens have no `aud`
  claim; `AllowedAudience` would always reject them. `AllowedScopes`
  validates the `scope` claim instead.
- **`CredentialProviderConfigurations: GATEWAY_IAM_ROLE`** — required for
  Lambda targets; tells the gateway to use its own IAM execution role
  when invoking the tool Lambdas.
- **Native YAML `InlinePayload`** — `ToolDefinition` lists must be native
  YAML structs, not a `!Sub |` JSON string. PascalCase keys (`Name`,
  `Description`, `InputSchema`, `OutputSchema`, `Properties`, `Required`,
  `Items`, `Type`).
- **`output_keys` is a nested object** — media-processing group names are
  runtime-dynamic (determined by the caller's `output_groups`), so they
  cannot be enumerated in `SchemaDefinition`; `output_keys` is typed as
  `object` with a prose description.
- **SecretsPopulator custom resource** — runs on Create and Update;
  no-ops on Delete. Uses the secret ARN as `PhysicalResourceId` so stack
  updates are in-place rather than replacement.
- **Stack has no Docker image, but it does have a Lambda zip** — its five
  handlers live in [gateway_tool/](gateway_tool/) (`download_url.py`,
  `ingest_list.py`, `upload_tool.py`, `secrets_populator.py`,
  `kit_publisher.py`) and ship as one flat zip referenced by
  `Code.S3Bucket`/`S3Key`. What forced the move off inline `ZipFile` is
  **naming, not size**: CloudFormation writes inline code to a file
  called `index`, so every inline handler must be `index.handler` —
  multiple handlers cannot share one inline block. (The inline size
  ceiling is 4 MB, nowhere near either file; do not "fix" a nonexistent
  few-KB limit.) Consequences: the zip must be flat (`zip -j`) for the
  handler paths to resolve, and the stack depends on its build artifact
  being in S3 — `create_update_stack.sh` cannot deploy a hand-edited
  `cfnstack.yaml` without running the build first.
- **Both log groups carry a `-log` SUFFIX and are reached via
  `LoggingConfig`, not by name.** They are *not*
  `/aws/lambda/{FunctionName}`, and that is deliberate: a deployment can
  encounter pre-existing Lambda-auto-created canonical groups (verified
  live — a stack create failed pre-flight with
  `Validation failed with 2 error(s)` and **no per-resource event**). CFN
  cannot adopt an existing group, and `Conditions` cannot branch on a
  runtime lookup. Renaming later was rejected as a fix because
  `LogGroupName` is a `createOnlyProperty` — a rename is a replacement
  that *deletes* the old group's contents. So the group names carry
  `-log` and each function points at its group through
  `LoggingConfig.LogGroup`. Any orphaned canonical groups left behind by
  earlier auto-creation are optional cleanup, not a deploy prerequisite.
- **Neither Lambda role may carry `AWSLambdaBasicExecutionRole`** — it
  grants `logs:CreateLogGroup` on every group, which lets a function that
  cannot reach its declared group silently create an unmanaged one
  instead of failing visibly (exactly how orphaned groups come to
  exist). Both roles use an inline
  `logs:CreateLogStream`/`PutLogEvents` scoped to the declared group's
  ARN. A side effect worth knowing: with no managed policy ARNs left,
  `validate-template` reports only `CAPABILITY_IAM` —
  `create_update_stack.sh` passes exactly that. A bundle test pins the
  absence of the managed policy; a Statement-scanning test cannot see a
  managed policy, which is how it once shipped.

---

## Build & deploy

This stack must be deployed after the three required micro-workflow
stacks (audio-stem, speech-diarization, media-processing). Every other
`*ToolLambdaArn` is optional — omit it or pass an empty string to deploy
without that target.

### Build

The build script packages [gateway_tool/](gateway_tool/) into
`media-toolkit-gateway-gateway-tool-{VERSION}.zip`, then copies
`cfnstack.yaml` to
`dist/global-s3-assets/media-toolkit-gateway-stack.template` with the
`%%BUCKET_NAME%%` / `%%KEY_PREFIX%%` / `%%VERSION%%` /
`%%PKG_GATEWAY_TOOL%%` tokens substituted, and uploads both to S3. It
fails the build if any token survives substitution — a literal
`%%PKG_GATEWAY_TOOL%%` as an `S3Key` would otherwise fail the stack
minutes into a deploy.

```bash
bash ./build-deploy-s3-dist.sh \
    --bucket MY_BUCKET \
    [--solution unified-media-toolkit] \
    [--version VERSION] \
    [--single-region]
```

The script prints the template S3 URL on completion — copy it for the
deploy step.

### Deploy

```bash
bash ./create_update_stack.sh \
    --stack-name my-media-toolkit-gateway \
    --audio-stem-lambda-arn <AudioStemToolLambdaArn> \
    --speech-diarization-lambda-arn <SpeechDiarizationToolLambdaArn> \
    --media-processing-lambda-arn <MediaProcessingToolLambdaArn> \
    --shot-segmentation-lambda-arn <ShotSegmentationToolLambdaArn> \
    --face-recognition-lambda-arn <FaceRecognitionToolLambdaArn> \
    --text-to-speech-lambda-arn <TextToSpeechToolLambdaArn> \
    --proxy-bucket <ProxyBucket> \
    [--template-url <URL printed by build-deploy-s3-dist.sh>] \
    [--gateway-name media-toolkit-gateway] \
    [--environment dev] \
    [--oauth-scope media-workflows] \
    [--region <region>] \
    [--profile my-profile]
```

The template is always fetched from S3 — it exceeds the 51,200-byte
`--template-body` limit. `--template-url` is optional: the script derives
the URL from the `Bucket`/`KeyPrefix` mappings inside the **built**
template under `dist/global-s3-assets/`. It reads the built copy rather
than `cfnstack.yaml` so an un-built template fails loudly on the
`%%BUCKET_NAME%%` placeholder instead of resolving to a wrong bucket.

---

## Testing

Unit tests (no AWS access needed):

```bash
cd media-toolkit-gateway && python3 -m unittest discover -s test -p 'test_*.py'
```

These pin, among other things: every `{target}___{tool}` name referenced
by a shipped skill or this README exists in the template; the skills
carry no deployment values (account ids, real buckets, personal
identifiers); and the KitPublisher's Create/Update/Delete behavior.

### Live gateway harness

[scripts/test_gateway.py](scripts/test_gateway.py) tests the gateway
directly without any agent:

```bash
# List tools only
python scripts/test_gateway.py \
    --secret-name /media-toolkit-gateway/dev/gateway-credentials

# Submit a media-processing job and poll to completion
python scripts/test_gateway.py \
    --secret-name /media-toolkit-gateway/dev/gateway-credentials \
    --run media-processing \
    --asset-s3-uri s3://my-ingest-bucket/test/sample.mp4

# Submit every covered workflow target under one idempotency token and poll
python scripts/test_gateway.py \
    --secret-name /media-toolkit-gateway/dev/gateway-credentials \
    --run all --idempotency-token my-run-001 \
    --asset-s3-uri s3://my-ingest-bucket/test/sample.mp4 \
    --tts-voice-id <voice_id> \
    --documents-s3-uri s3://<proxy>/_indexing/<collection>/contracts.jsonl

# Re-run the same arguments and assert each target's resubmit contract
python scripts/test_gateway.py ... --run all --idempotency
```

The script: fetches the credentials bundle from Secrets Manager, obtains
a Cognito token, calls `tools/list` to verify the deployed tools are
advertised, then optionally submits jobs and polls until a terminal state
or timeout. On SUCCEEDED it validates the `outputs[]` contract,
batch-presigns the first up-to-3 file entries via
`media-outputs___get_download_urls` together with one deliberately
invalid uri — asserting request-order echo, `count`, and that the bad
entry fails alone — then presigns one entry via the singular
`media-outputs___get_download_url` to assert its flat `{url}` shape, and
fetches a URL over HTTP. There is no output-prefix flag — outputs always
land at the service-derived `micro-workflow/{target}/{workflow_id}/`
prefix.

`--run` does not cover every target with a start tool (the LLM-consuming
targets have their live checks in their own repos or are chained the way
`--tts-transcript-s3-uri` chains diarization). Not covered by `--run` is
not the same as not covered at all: **every target sits in the harness's
`TOOLS` map**, so `tools/list` verifies every advertised tool. That was
once a real gap — targets absent from the map fell into the
`EXTRA (unexpected, not a failure)` branch, `missing` could never contain
them, and a target that silently failed to create still printed PASS. The
fix: `expected` is built from the full `TOOLS` map, `OPTIONAL_TARGETS` is
a module constant naming exactly the targets whose `GatewayTarget`
carries a `Condition`, and a bundle test pins both structures against
`cfnstack.yaml` in **both** directions — the drift happened because
nothing compared them.

**Three response shapes, three result checks** — a single `outputs[]`
check cannot cover them, so the harness dispatches on target:

| Target(s) | SUCCEEDED shape | What the harness asserts |
|---|---|---|
| audio-stem, speech-diarization, media-processing, shot-segmentation, face-recognition | `workflow_id` + `outputs[]` of `{uri, mime, label}` | the shared contract, then the presign round trip |
| text-to-speech | `workflow_id` + `output_s3_prefix` + `usage_uri` as usual, but `outputs[]` is per-transcript, keyed by `transcript_index` (no label/mime) + `batch_qc_summary` | dense in-order indices, no per-clip `error`, a self-consistent QC rollup — a **qc failure is reported, not failed on**, because the QC transcriber is not deterministic across invocations — plus the three parity fields and that every clip uri sits under the advertised prefix |
| hybrid-search | `prepared`/`skipped`/`indexed` counts, never `outputs[]` | counts are ints, `indexed == prepared`, and that the manifest resolved to a non-zero document count |

### `--tts-transcript-s3-uri` — synthesizing one real speaker

`--run text-to-speech` synthesizes three canned lines by default. Pass
`--tts-transcript-s3-uri` a completed speech-diarization run's
`transcripts/final_transcript.json` and it instead synthesizes **every
sentence carrying one `speaker_id`**, chaining the two workflows over one
asset the way an agent would:

```bash
python scripts/test_gateway.py \
    --secret-name /media-toolkit-gateway/dev/gateway-credentials \
    --run text-to-speech --tts-voice-id <voice_id> \
    --tts-transcript-s3-uri s3://<proxy>/micro-workflow/speech-diarization/<workflow_id>/transcripts/final_transcript.json \
    [--tts-speaker-id SPEAKER_01] --idempotency-token <tok> --idempotency
```

The default speaker is the one with the **most sentences**, not
`SPEAKER_00` — the point of the flag is a real dialogue load. Ties break
on `speaker_id`, so the pick is deterministic; a resubmit that chose
differently would silently change the request.

Three things this deliberately does:

- **Resolved once, in `main()`, and stashed on the namespace.**
  `build_args()` must stay a pure function of the namespace or the
  resubmit derives a second `workflow_id` and `--idempotency` reports a
  contract failure that is really a harness bug. An S3 read inside
  `build_args` would put a network fetch in the hash preimage.
- **Checks BOTH of tts's payload bounds, because neither dominates.** The
  per-batch bound and the Map-aggregate bound cross at a text length well
  below tts's documented per-text limits — a real speaker's turns are
  full of short lines, and a hundred of those are bounded by the
  *aggregate*, which splitting into batches cannot relieve. Checking only
  the documented-tighter bound would let exactly the short-turn case
  through. The local estimator copy is verified byte-exact against the
  service's `estimate_batch_payload_bytes` and **deliberately
  over-charges** on the aggregate (fixed envelope count where the real
  gate passes the true iterator count) — over-charging is the safe
  direction for a client-side pre-check. If either ever diverges further,
  re-derive it rather than nudging a constant. Both are checked in
  `resolve_speaker_transcripts` so the error names the speaker and points
  at `--tts-speaker-id`, instead of arriving as an `INVALID_PARAMETER`
  from a submit.
- **Does NOT derive `target_durations_s`,** although each sentence's real
  `end - start` is right there and tts accepts a 1:1 list. The service
  bounds stretching at ±10 % and *raises* past it, and a cloned voice
  reading someone else's line has no reason to land within 10 % of the
  source speaker's pace. A raised clip records `{"error": ...}` and fails
  `verify_tts_result`, so wiring the durations in would make the test's
  verdict depend on how fast the person in the source video talks.
  Duration control needs its own test with its own tolerance.

A failure to resolve is **fatal, not a fallback to the canned lines** — a
silent fallback would report PASS while claiming to have tested a real
speaker. `verify_tts_result` takes its expected clip count from
`build_args`, so there is one source of truth for what was submitted.

### `--idempotency-token` and `--idempotency`

`--idempotency-token` supplies one token to every start tool that
declares one (`TOKEN_ACCEPTING_TARGETS` — the shared-contract targets
plus `text-to-speech`; **not** the same set as
`SHARED_CONTRACT_TARGETS`, since tts accepts a token but has its own
response shape, and `hybrid-search`'s uuid4-named indexer takes none at
all). The token **names nothing** — it folds into the `workflow_id` hash
and is then discarded — so reusing it with different parameters correctly
starts a fresh run. The inverse matters for the test: a resubmit only
reaches the existing run when *every* argument matches byte-for-byte,
which is why `build_args()` is a pure function of the parsed CLI
namespace rather than re-derived per call.

`--idempotency` **requires `--idempotency-token`** when
`face-recognition` or `text-to-speech` is in the target set
(`TOKEN_REQUIRED_FOR_IDEMPOTENCY` — see
[the design section](#idempotency-service-derived-output-prefixes)), and
`main()` refuses with an explanatory message rather than letting the run
fail: without a token those targets *correctly* start a fresh run, and
the assertion would fail on right behavior — the worst kind of test
failure, since the obvious reading is that the target regressed. The two
sets are kept separate because they answer different questions — "is the
field accepted?" versus "does omitting it change the outcome?" — and only
the second gates the CLI. Note that **sending a token to a tool that does
not honour one is not an error**: every gateway accepts the field on
polls, reads and searches and logs a warning instead of rejecting, on the
grounds that a hallucinated parameter name must not fail a request over a
field that could not have changed the answer. The harness still sends it
only where it is honoured, so the resubmit hash preimage stays honest.

`--idempotency` re-submits and asserts the resubmit contract:

| Contract | Targets | Resubmit returns |
|---|---|---|
| `RETURNS_EXISTING` | audio-stem, speech-diarization, media-processing, shot-segmentation, face-recognition, text-to-speech | no second execution, and the same **handle** a fresh start returns — same `workflow_id` + `job_id`, the existing run's `status`, a `message` saying the call was absorbed, and `error` when that status is a failure. **No results.** |

There is exactly **one** contract, and `IDEMPOTENCY_CONTRACT` stays a
dict rather than collapsing to a set so a future target with a genuinely
different behavior has somewhere to declare it. The idempotency pass
**polls** where it used to read the submit — that round trip IS the
contract, not an inefficiency to remove. `--idempotency` fails on a
missing `message` (see the design section for why `job_id` comparison
cannot replace it), and `SUBMIT_HANDLE_FIELDS` pins the submit's field
set as an **upper bound over all paths**, never an equality, since
`current_state` is absent once a run stops and `message`/`error` appear
only on an absorbed one.

One target is **excluded by design** (`IDEMPOTENCY_EXCLUDED`), not
overlooked: `hybrid-search` — see the design section. It is still fully
exercised by `--run`.

`face-recognition` is the target the assertion protects most: a duplicate
run burns a semaphore slot, billed Rekognition calls, and can auto-index
faces into a persistent collection. `--run` passes `max_faces_to_index:
0` so a failed assertion cannot mutate a live collection.

The harness reports a sticky terminal failure loudly, with the `error`
cause (which is what decides between fixing a parameter and supplying a
new token), since it otherwise reads like a transient poll result.

---

## Gotchas

- **Editing a handler without bumping `.version` ships stale code** —
  `S3Key` is
  `{solution}/{version}/media-toolkit-gateway/media-toolkit-gateway-gateway-tool-{version}.zip`,
  and CloudFormation does not detect S3 content changes, only
  key/version changes. Rebuilding at the same `.version` overwrites the
  same key, so the stack update sees an unchanged `Code` block and
  leaves the old code running. Bump `.version` (or pass `--version`)
  whenever a handler changes — same convention as the sibling
  micro-workflow stacks. The client kit rides the same zip, so a
  skills/README/script change also deploys only with a `.version` bump.
- **`zip` exits 0 when only *some* members are missing** — it warns,
  packages the rest, and returns success (it fails only when nothing
  matched). `build_gateway_tool_zip` therefore checks each member with
  `[ -f ]` before zipping; trusting the exit code would ship a Lambda
  whose handler module simply isn't in the archive.
- **The `uuid` event key must not be renamed** and no tool accepts it as
  input — see [the three ids](#the-three-ids).
- **Windows: "running scripts is disabled on this system"** — the
  machine's PowerShell execution policy blocks unsigned scripts. The
  helper invocations written by `configure.ps1` already carry
  `-ExecutionPolicy Bypass`; to run `configure.ps1` itself use
  `powershell -ExecutionPolicy Bypass -File .\scripts\configure.ps1 ...`
  (process-scoped — no machine setting is changed).
- **Windows: `aws` is not recognized** — install the AWS CLI v2 (MSI or
  `winget install Amazon.AWSCLI`) and reopen the terminal.
- **401 / connection refused mid-session** — the bearer token expired
  (12 h). Claude Code's headersHelper refreshes automatically on
  reconnect; in Kiro, refresh the token in the config's `headers` and
  reconnect from the MCP panel.
- **`AccessDeniedException` fetching the secret** — your AWS identity
  lacks `secretsmanager:GetSecretValue` on the gateway-credentials
  secret; this is deliberate least-privilege, request access rather than
  sharing bundles.
- **A tool refuses with `INVALID_PARAMETER`** — the message names the
  condition; `describe_tool` documents the accepted parameters and
  bounds.
- **Status shows `SUCCEEDED` but an expected output is missing** — some
  outputs are conditional on their source being present;
  `get_job_status` lists only what exists, and the output contract
  explains each condition.
