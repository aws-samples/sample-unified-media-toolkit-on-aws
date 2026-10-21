---
name: list-ingest-files
description: List source media files in the ingest bucket via the media-toolkit list_files tool, then present them as a markdown table. Use when the user asks to list, browse, or show files in their ingest bucket. Defaults to the first 20 files.
metadata:
  version: 1.0.0
compatibility: Requires the media-toolkit MCP server to be connected. The gateway's media-ingest target must be enabled (IngestBucket configured on the stack).
---

# List ingest bucket files

List source media in the caller's ingest bucket using the media-toolkit
`media-ingest___list_files` tool, and render the result as a markdown table.

## When to use

Trigger this skill when the user asks to list, browse, or show files in their
ingest bucket (for example: "list my ingest files", "show my bucket", "what's in
my ingest bucket").

## How to run

1. Call the gateway tool `media-ingest___list_files`.
   - Default to `max_results: 20` unless the user asks for a different count
     (the tool clamps `max_results` to 1–200).
   - Optional filters, pass through only when the user asks:
     - `prefix` — server-side key prefix, e.g. `movies/` or any folder path.
     - `extension` — a SINGLE extension string, e.g. `mp4` or `mov`. Pass ONE
       extension per call. A list/array of extensions is NOT reliably honored by
       this deployment and can return zero results — if the user wants several
       extensions, make one call per extension and merge the rows.
   - `$ARGUMENTS` (any text after the command) may name a folder, an extension,
     or a count. Interpret it and map it onto `prefix` / `extension` /
     `max_results` accordingly.

2. The tool returns:
   ```json
   {
     "files": [
       { "s3uri": "...", "key": "...", "size": 123, "last_modified": "ISO-8601" }
     ],
     "count": 20,
     "next_token": "<last key scanned, present only when more may follow>"
   }
   ```

3. Render the `files` array as a markdown table with these columns:

   | # | Key | Size | Last Modified | S3 URI |
   |---|-----|------|---------------|--------|

   - `#` — 1-based row index.
   - `Key` — the object `key`.
   - `Size` — humanize `size` bytes (KB / MB / GB, one decimal place).
   - `Last Modified` — the `last_modified` timestamp as returned.
   - `S3 URI` — the full `s3uri` (feeds workflow submit tools directly).

4. Below the table, state the number of rows shown (`count`). If `next_token`
   is present, tell the user more files exist and that they can ask for the next
   page (pass the returned `next_token` back as `next_token` to continue).

## Notes

- This is read-only: the tool lists source media only (no download, no
  presigning). It reads the INGEST bucket, not the proxy/output bucket.
- Results are lexicographic by key, not by date.
- Do not fabricate rows. Only render files actually returned by the tool.
- If the tool errors because the gateway isn't connected, tell the user to
  connect the `media-toolkit` MCP server (reload the window or use the
  MCP Servers view) and retry.
