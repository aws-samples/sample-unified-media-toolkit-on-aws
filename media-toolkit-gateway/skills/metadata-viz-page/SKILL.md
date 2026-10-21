---
name: metadata-viz-page
description: Build a CUSTOM Unified Media Toolkit review page (synced video + findings list + timeline + table view) as one self-contained HTML file, using the toolkit's shared page template and design system. Use when the user asks for a review page that no packaged skill produces — a bespoke combination of workflow outputs, a side-by-side comparison, a custom evidence view. For the READY-MADE page of a completed analyze-video run, use the visualize-metadata skill instead; for compliance review use review-compliance.
metadata:
  version: 1.0.0
---

# Unified Media Toolkit — custom metadata review page

Build a single HTML page that plays an asset's video in sync with the
metadata the micro-workflows produced: a click-to-seek findings list, a
category-lane timeline with a filmstrip, an accessibility table view, and a
dark/light theme. The canonical use is a content-compliance page (visual
`depicted` evidence + spoken `discourse_facts` + a word-list scan), but the
frame fits any per-asset review of timed metadata.

**When to use which skill:** `visualize-metadata` renders the standard
page for a completed analyze-video run; `review-compliance` and
`create-audio-description` render their own specialized pages. THIS
skill is for every page those do not cover: it carries the shared
template and the design rules, and you write a small builder for the
specific evidence the user wants to review — the result matches the
toolkit's visual language instead of inventing a new one.

This skill is self-contained:

- `references/template.html` — the complete working page (CSS + JS) with a
  single `const DATA = /*__DATA__*/;` injection point. **Start from a copy of
  this file; do not rebuild the page from scratch.**
- `references/reference.md` — the design system (palette, layout, mark and
  interaction grammar), the DATA JSON contract, the builder pattern, and the
  delivery checklist. Read it before modifying the template or writing the
  builder.

## Architecture (non-negotiable)

- **One self-contained HTML file that works from `file://`.** All metadata,
  base64 thumbnails, and the WebVTT subtitle text are inlined. Only heavy
  media (the video proxy) streams over a presigned S3 URL.
- **No external dependencies.** No CDN, no framework, no webfonts. Vanilla
  JS, `<canvas>` for the timeline, system fonts.
- **Template + builder split.** A Python builder reads the workflow output
  URIs from a per-asset `config.json`, joins the evidence into the DATA
  shape, presigns the video, and substitutes the `/*__DATA__*/` marker.
  Never hand-edit a generated page — presigned URLs expire with the signing
  session, so regeneration must stay routine.
- If the page needs audio, presign a proxy that has an audio track
  (shot-segmentation proxies are video-only by design).

## Workflow

1. Read `references/reference.md`.
2. Copy `references/template.html`; change page chrome (title text) only if
   the use case demands it.
3. Write/extend the builder: gather outputs → build `DATA` exactly per the
   contract → inject. Evidence semantics matter: the page shows
   *observables, never verdicts*; `attested_none` only for layers that were
   actually asked; quotes re-verified verbatim before showing ✓.
4. Verify against the delivery checklist at the end of `reference.md`
   (file:// works, both themes, every finding in list *and* table, no
   external requests beyond the presigned video).

## Look and feel, in one paragraph

Dark by default with a designed light mode, scoped CSS variables under
`.viz-root`, system-ui at 13px with monospace tabular timecodes. Exactly
three categorical colors (blue `#3987e5`, orange `#d95926`, green `#199e70`)
and they encode the *evidence layer*, never entity identity — entities get
lane position plus an always-visible label. Durations are rounded span bars,
instants are dots, the third layer is a ring; the playhead rides its own
overlay canvas. Everything clicks to seek, hover tooltips name what a click
will do, and the table view ships always because it is the accessibility
relief, not a bonus. If you are tempted to add a fourth color, add a labeled
lane instead — the full rules and the reasons are in `reference.md`.
