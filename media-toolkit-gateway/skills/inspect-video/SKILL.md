---
name: inspect-video
description: Extract technical metadata for a video/audio asset via the media-toolkit process_media tool (metadata-only, no transcode) and display it as markdown tables. Use when the user asks to inspect, probe, or get the metadata/codec/resolution/duration of a media file in S3.
metadata:
  version: 1.0.0
compatibility: Requires the media-toolkit MCP server to be connected. Reads the source asset from S3 and writes metadata.json to the proxy bucket.
---

# Inspect video (technical metadata)

Extract technical metadata for a media asset using the media-toolkit
`media-processing___process_media` tool with a single `metadata` output group
(no transcode — the cheapest, fastest path), then render the result as markdown
tables.

## Input

The user supplies the source asset as an `s3://` URI (or names a file we can
resolve to one, e.g. via the list-ingest-files skill). `$ARGUMENTS` is treated
as the S3 URI when present. If no asset can be determined, ask for the `s3://`
URI before running.

## How to run

1. Submit a metadata-only job with `media-processing___process_media`:
   - `asset_s3uri`: the source `s3://` URI.
   - `output_groups`: exactly one metadata group requesting full detail:
     ```json
     [{ "format": "metadata", "group_name": "metadata", "detail": "full" }]
     ```
     `detail: "full"` adds per-stream disposition and tags (language, track
     title, default/forced flags). A metadata group runs NO transcode, so this
     is cheap regardless of source size or format (MXF, MP4, MOV, ProRes, etc.).
   - The tool returns a `job_id` (Step Functions execution ARN) and `RUNNING`.

2. Poll `media-processing___get_job_status` with that `job_id` until
   `status` is `SUCCEEDED` (metadata jobs finish in seconds). The success
   payload includes top-level facts (`duration_sec`, `has_video`, `has_audio`,
   `is_all_intra`) and an `outputs` list with the `metadata` object's `uri`.

3. Fetch the metadata JSON:
   - Presign the metadata `uri` with `media-outputs___get_download_url`.
   - Retrieve the JSON from the presigned URL (it lives on the proxy bucket).

4. Render the metadata as the tables below. Only include rows that are present
   in the JSON; omit fields the probe didn't report rather than inventing them.

## Output format

Prefer NARROW, vertical tables (Field | Value) over wide many-column tables —
they stay readable in a chat pane and never wrap awkwardly. One table per
stream, not one giant row-per-stream grid. Lead with a one-line summary, use
section headings with an icon, humanize every raw value, and keep the raw value
in parentheses only where it adds signal.

Start with a single summary line, e.g.:

> **Love_and_Peace_UHD4k5994_HDR_P3PQ_720p_4500kbps.mp4** — 11:59 · H.264 1280×720 23.976p SDR · AAC stereo 48 kHz · 399.8 MB

### 📦 Container

| Field | Value |
|-------|-------|
| Format | QuickTime / MOV |
| File size | 399.8 MB (419,183,874 bytes) |
| Duration | 00:11:59 (718.968 s) |
| Overall bitrate | 4.66 Mb/s |
| Streams | 2 — 1 video, 1 audio (0 programs) |
| All-intra | No |
| Created | 2019-03-12 10:48:23 UTC |

Formatting rules:
- Duration as `HH:MM:SS` first, seconds in parentheses.
- Sizes humanized (MB/GB, 1 decimal), raw bytes in parentheses.
- Bitrates as Mb/s (video/overall) or kb/s (audio), 1–2 decimals.
- Booleans as **Yes**/**No**, never `true`/`false`.
- Timestamps as `YYYY-MM-DD HH:MM:SS UTC`.

### 🎬 Video — stream N (`mapid`)

One narrow table per video stream:

| Field | Value |
|-------|-------|
| Codec | H.264 / AVC (avc1) |
| Profile / Level | High / 4.0 |
| Resolution | 1280 × 720 |
| Aspect (DAR / SAR) | 16:9 / 1:1 |
| Frame rate | 23.976 fps (24000/1001) |
| Scan | Progressive |
| Pixel format | yuv420p, 8-bit |
| Dynamic range | SDR |
| Bitrate | 4.5 Mb/s |
| Frames | 17,238 |
| Language | und (undefined) |

- Frame rate: convert the `r_frame_rate`/`fps` fraction to a decimal (24000/1001
  ≈ 23.976), keep the fraction in parentheses.
- Scan: **Progressive** / **Interlaced** (from `field_order`/`interlaced`).
- Dynamic range: infer SDR/HDR from pixel format / color transfer when present;
  omit the row if unknown.

### 🔊 Audio — stream N (`mapid`)

One narrow table per audio stream:

| Field | Value |
|-------|-------|
| Codec | AAC (LC) |
| Channels | 2 (stereo) |
| Sample rate | 48 kHz |
| Bitrate | 160 kb/s |
| Language | und (undefined) |

### 📝 Other streams

If subtitle / data / timecode streams exist, give each its own narrow table
(codec, type, language). If there are chapters, note the count. Omit this
section entirely when there are none.

## After the tables

- State the run facts: job succeeded, `pipeline_wall_sec`, and
  `total_estimated_cost`.
- List the output objects (the `metadata` and `probe` URIs) so the user can
  fetch them again.
- SANITY CHECK the filename against the essence: media filenames often encode a
  source master's spec (e.g. `UHD4k5994_HDR_P3PQ`) that does NOT match the actual
  file (which may be a downscaled SDR proxy). If the resolved video facts differ
  from what the name implies, call this out plainly.

## Notes

- Metadata-only is far cheaper than producing a throwaway proxy — always use the
  single metadata group for "what is this file" questions; never transcode just
  to read facts.
- The full metadata-group field reference (every payload field, the MXF block,
  detail-level semantics) is in
  [../video-processing/references/media_processing_agent.md](../video-processing/references/media_processing_agent.md).
- If `process_media` errors because the gateway isn't connected, tell the user
  to connect the `media-toolkit` MCP server (reload the window or use
  the MCP Servers view) and retry.
- Presigned URLs can expire; if a fetch returns HTTP 403, re-presign and retry.
