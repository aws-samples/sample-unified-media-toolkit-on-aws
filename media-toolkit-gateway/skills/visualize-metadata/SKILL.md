---
name: visualize-metadata
description: Build the self-contained interactive HTML visualization page from a completed analyze-video run (fresh output JSON + presigned media via the media-toolkit), write it under outputs/, and open it in Chrome. Use when the user asks to visualize, view, or build the HTML page for an analyzed video, or to refresh an existing page's expired media links.
metadata:
  version: 1.0.0
compatibility: Requires the media-toolkit MCP server to be connected and a completed analyze-video run (its state file). Reads outputs from the proxy bucket; writes the HTML into the workspace and opens Chrome (macOS `open -a "Google Chrome"`).
---

# Visualize metadata (HTML page)

Turn a completed `analyze-video` run into ONE self-contained HTML page and open
it in Chrome. This skill runs NO workflows — it reads the analysis outputs
(fresh, via the gateway), presigns the media (via the gateway), builds the page,
and launches the browser.

> ⚠️ ALWAYS go through the media-toolkit tools. Presign media with
> `media-outputs___get_download_url(s)` and fetch every output JSON from those
> presigned URLs. Do NOT use `aws s3 cp` / `aws s3 presign` / `aws s3 ls` — the
> gateway is the contract. The AWS CLI is an emergency fallback ONLY when the
> gateway is unreachable, and any CLI-minted URLs MUST be replaced with
> gateway-issued ones once it is back. On a gateway auth error, ask the user to
> refresh the gateway token and reload rather than falling back silently.

> ⚠️ IDs FLOW, they are never retyped: copy every workflow_id / run prefix VERBATIM from the sibling run's response (`output_s3_prefix`) or the recorded state file — never from memory. A mistyped-but-EXISTING prefix binds against the wrong run and nothing will refuse it; only a nonexistent one fails validation.

## Input

`$ARGUMENTS` is an `ASSET_ID` or shorthand name (e.g. "Love and Peace").

Resolve the run from `outputs/analyze-video/{ASSET_ID}.state.json`:
- **State file found** → use its recorded s3:// output uris.
- **No state file** → tell the user to run `analyze-video {ASSET_ID}` first (this
  skill does not analyze). If they instead supply explicit run prefixes/uris,
  accept those.

## Steps

1. **Read the state file** to get the per-step output uris (proxy, stems,
   `stem-waveforms`, shots.json, transcript, conversation analysis, SME
   scenes/shots, PSA structure/rundown/manifest, cast-resolution
   persons/speaker-map/credits, face `facemetadata` + `rep_face_key` crops,
   artwork `artwork_candidates.json` + frames when that step ran) and the
   face `collection`.
2. **Fetch the analysis JSON FRESH** each time so the page reflects the latest
   run: presign each output object with `media-outputs___get_download_url(s)` and
   retrieve the JSON from those URLs. Downsample `stem_waveforms.json` to ~0.5 s
   bins when embedding (keeps the file small). Pull the full per-shot `shots.json`
   for exact shot timings.
3. **Presign ONLY media** with `media-outputs___get_download_urls`
   (batch ≤50, `expiry_seconds: 3600`): proxy mp4, the 3 stem WAVs, each
   person's `rep_face_key` crop, and each artwork candidate's frame jpg when
   artwork ran. Only presign objects, never prefix entries.
4. **Build the page from the FROZEN template** — read
   [references/visualization.md](references/visualization.md) BEFORE writing.
   There is ONE template, `references/template_unified.html`; the content mode
   (`news` vs `narrative`) comes from PSA's `manifest.json` `mode` field and
   rides in DATA — it is never inferred from prompt-specific fields. COPY the
   template verbatim, build the DATA object per the contract (including the
   rundown fusion the contract spells out), and substitute the tokens
   (`/*__DATA__*/`, `__ASSET__`). Never author or restyle page chrome: the
   template IS the approved format. Write the result to
   `outputs/visualize-metadata/{ASSET_ID}.html`.
5. **Validate**: the embedded DATA JSON parses, and no `__` token survives in
   the file.
6. **Open Chrome** when the page is ready:
   `open -a "Google Chrome" "outputs/visualize-metadata/{ASSET_ID}.html"`
   (macOS). Tell the user to hard-refresh (Cmd+Shift+R) if a prior version was
   cached, and that **media links expire in ~1 hour** — re-invoke this skill to
   refresh them.

## Mode choice + what the template already handles

- **Mode comes from PSA, presence decides the rest.** `news` vs `narrative`
  is PSA `manifest.json`'s `mode` field (PSA absent → `narrative`). Every
  optional module — PROGRAM STRUCTURE lanes, the Artwork tab, credits,
  stems, faces — renders only when its DATA key exists; a skipped run
  degrades by omission.
- Everything about the page itself — timeline groups, transport, tabs,
  waveform lanes, follow/zoom behavior, subtitles, cost table — lives in
  the template and is NOT yours to re-decide. If the page looks wrong, the
  fix is a deliberate template edit (a reviewed change to references/),
  never a per-run rewrite.

## Notes

- This skill is idempotent and cheap: re-running it just re-presigns media and
  rewrites the page. It is the intended way to refresh expired media links.
- If the state file records a skipped stage (e.g. face-rec skipped), the page
  degrades by omission — drop that tab/lane and note it in Summary.
