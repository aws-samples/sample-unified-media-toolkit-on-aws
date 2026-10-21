---
name: export-package
description: Export a completed analyze-video run as an editorial FCPXML timeline for DaVinci Resolve (and other NLEs), packaged in a folder with the downloaded proxy video, audio stems, and a sidecar SRT. Shots, scenes, conversation chapters, PSA break candidates, and faces become markers / keyword ranges; the transcript becomes a caption track. Use when the user asks to export an FCPXML / editorial timeline / Resolve package for an analyzed video.
metadata:
  version: 1.0.0
compatibility: Requires the media-toolkit MCP server to be connected and a completed analyze-video run (its state file). Downloads media via gateway-presigned URLs; writes an FCPXML package folder into the workspace.
---

# Export package (FCPXML editorial timeline)

Turn a completed `analyze-video` run into a portable editorial package: a
frame-accurate FCPXML timeline plus the downloaded proxy + audio stems + a
sidecar SRT, all in one folder that relinks by relative path. Runs NO workflows.

> ⚠️ ALWAYS go through the media-toolkit tools. Presign/download media
> and fetch output JSON via `media-outputs___get_download_url(s)`. Do NOT use
> `aws s3 cp` / `aws s3 presign` / `aws s3 ls` — the gateway is the contract; the
> AWS CLI is an emergency fallback ONLY when the gateway is unreachable. On a
> gateway auth error, ask the user to refresh the gateway token and reload.

> ⚠️ IDs FLOW, they are never retyped: copy every workflow_id / run prefix VERBATIM from the sibling run's response (`output_s3_prefix`) or the recorded state file — never from memory. A mistyped-but-EXISTING prefix binds against the wrong run and nothing will refuse it; only a nonexistent one fails validation.

## Input

`$ARGUMENTS` is an `ASSET_ID` or shorthand name (e.g. "Love and Peace").

Resolve the run from `outputs/analyze-video/{ASSET_ID}.state.json`:
- **State file found** → use its recorded s3:// output uris + fps/duration.
- **No state file** → tell the user to run `analyze-video {ASSET_ID}` first.

## Output

A self-contained folder `outputs/export-package/{ASSET_ID}/` — see
[references/fcpxml.md](references/fcpxml.md) for the exact FCPXML structure,
metadata→timeline mapping, and validity rules. READ IT before generating.

```
{ASSET_ID}/
├── {ASSET_ID}.fcpxml     # asset src = ABSOLUTE file:// into ./media/ (see Notes)
├── {ASSET_ID}.fcp7.xml   # FCP7 xmeml for Adobe Premiere (always emitted)
├── media/  proxy.mp4  speech.wav  music.wav  sfx.wav
├── captions.srt
└── metadata/  shots.json scenes.json transcript.json conversation.json structure.json faces.json
```

## Steps

1. **Read the state file**; collect the output uris (proxy, 3 stems, shots.json,
   transcript, conversation analysis, SME scenes, PSA structure, facemetadata)
   and the **frame rate + duration** (from shots.json / the proxy probe — never
   hardcode fps).
2. **Fetch the analysis JSON FRESH** via gateway presigns; save copies under
   `metadata/` for reference and to build the timeline from.
3. **Download the media** into `media/`: presign proxy + speech/music/sfx WAVs
   with `media-outputs___get_download_url` and fetch the bytes (the proxy is the
   editorial reference; stems are secondary audio-role clips). This is still the
   gateway path — gateway-issued URLs, we just download the object.
4. **Build `{ASSET_ID}.fcpxml`** per references/fcpxml.md, AND `{ASSET_ID}.fcp7.xml` (FCP7 xmeml for Premiere — see the FCP7 flavor section there; both are always emitted):
   - one `format` + `asset` (proxy, relative src), one sequence/spine, fps and
     resolution from the real values; all times snapped to the frame grid.
   - **Shots** → chapter/plain markers at each shot start.
   - **Scenes** → keyword ranges (`Scene {id}: {setting}`).
   - **Conversation chapters** → chapter markers at each segment start (topic).
   - **PSA break candidates** → to-do markers (`BREAK? score {n}`).
   - **Faces/people** → keyword ranges per appearance (name / `Unknown {id}`).
   - **Stems** → speech/music/sfx as secondary-lane audio-role clips.
   - **Transcript** → a `<caption>` per sentence (caption track).
5. **Write `captions.srt`** from the transcript sentences (always, as a fallback
   every NLE reads).
6. **Validate**: the FCPXML is well-formed XML, every time value is a rational on
   the frame grid, markers/keywords/captions are `start`-ascending within their
   parent, and names/text are XML-escaped. Report any construct you downgraded
   for Resolve compatibility.
7. **Report**: the package folder path, what each mapping produced (counts:
   N shot markers, N scenes, N chapters, N break markers, N face keyword ranges,
   N captions), the fps/resolution used, and how to import into Resolve
   (File → Import → Timeline → select the .fcpxml; media relinks from ./media).

## Notes

- Default to **one proxy clip + markers/keywords**, not one clip per shot — it
  relinks cleaner and keeps audio continuous. Offer per-shot clips only if the
  user asks.
- Stems are added muted/secondary (the proxy already carries the mix); they are
  there for the editor to solo, not to double the mix.
- Downloading the proxy can be large; report the size. If the download fails,
  still emit the FCPXML but point its asset at the presigned proxy URL and note
  that the package is not self-contained until the media is fetched.
- The FCPXML must reference the media as ABSOLUTE `file://` URLs into this
  package's `media/` folder — NOT a relative `./media/...` path (Resolve's
  importer fails to resolve relative media-rep src: "N of N clips were not yet
  found") and never an s3:// uri. The media still sits beside the FCPXML so the
  folder stays self-contained; the paths are just absolute.
