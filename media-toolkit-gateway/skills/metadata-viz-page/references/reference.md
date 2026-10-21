# Design system, DATA contract, and builder pattern

Companion to `template.html` in this folder. The template is the working
implementation of everything below — copy it and inject data; read this file
when you need to modify or extend it.

## Design system

All styling is scoped under `.viz-root` with CSS custom properties. Dark is
the default (`<html data-theme="dark">`); the light theme swaps the variable
set. Both modes are designed selections from the same ramps, never
auto-inverted.

### Palette

| Role | Dark | Light |
|---|---|---|
| Page background | `#0d0d0d` | `#f9f9f7` |
| Panel surface | `#1a1a19` | `#fcfcfb` |
| Text primary | `#ffffff` | `#0b0b0b` |
| Text secondary | `#c3c2b7` | `#52514e` |
| Text muted | `#898781` | `#898781` |
| Grid lines | `#2c2c2a` | `#e1e0d9` |
| Baseline | `#383835` | `#c3c2b7` |
| Hairline border | `rgba(255,255,255,.10)` | `rgba(11,11,11,.10)` |
| Series 1 (blue) | `#3987e5` | `#2a78d6` |
| Series 2 (orange) | `#d95926` | `#eb6834` |
| Series 3 (green) | `#199e70` | `#1baf7a` |
| Good | `#0ca30c` | `#006300` |
| Warning | `#fab219` | `#fab219` |

### Color rules

- **Color carries at most one meaning per page**, and only for a *genuine*
  categorical set of **three or fewer** (e.g. evidence layer:
  visual / spoken / word-list). The three series slots are all-pairs
  contrast-validated in both modes; a fourth categorical hue has repeatedly
  failed validation — when you think you need one, the answer is facets:
  split into labeled lanes or sections instead.
- **Entity identity never comes from color.** Categories, speakers, people
  get identity from lane position + an always-visible label. Within a lane,
  the *layer* of a mark may vary by color; the lane itself is named in the
  left gutter.
- Known accessibility gap: light-mode green is ~2.7:1 against the surface
  (below the 3:1 target for graphical objects). The shipped relief is
  structural, not chromatic: every lane is labeled, every finding appears in
  the findings list, and the table view exposes all rows as plain text.
  Keep all three whenever you reuse the palette.

### Typography and chrome

- `system-ui` stack, 13px base. No webfonts.
- Every timecode: `ui-monospace` + `font-variant-numeric: tabular-nums`.
- Section/table labels: 10px, uppercase, `.05em+` letter-spacing, muted.
- Buttons: 6px radius, hairline border, state via `aria-pressed` (pressed =
  primary text + stronger border). Stat pills: 999px radius, hairline border.
- Panes divide with 1px hairlines (a 1px `gap` over a border-colored grid
  background, or `border-top`).

## Layout grammar

A fixed-viewport app frame (`height: 100vh`; each pane manages its own
overflow — the page itself never scrolls):

```
┌ header bar: title+sub · spacer · pills · [Table view] [Light] ┐
├───────────────────────────┬───────────────────────────────────┤
│ player pane (≤52%)        │ findings pane                     │
│  video (≤44vh, clipped)   │  head: filters · spacer · Follow  │
│  timecode · subtitle btn  │  scrollable click-to-seek list    │
│  prose justification      │                                   │
├───────────────────────────┴───────────────────────────────────┤
│ timeline pane: legend · filmstrip · gutter+canvas · note      │
├────────────────────────────────────────────────────────────────┤
│ table view (toggled)                                          │
└────────────────────────────────────────────────────────────────┘
```

Hard-won constraints baked into the template — keep them:

- The player pane has `overflow: hidden` and the video
  `max-height: min(44vh, 100%)`. Without both, opening the table view
  squeezes the pane and the video overflows onto the timeline.
- Canvas left gutter ≈ 116px for lane labels; ruler steps 10/30/60 s chosen
  by total duration.
- The playhead lives on a **separate overlay canvas** stacked on the
  timeline canvas, so per-frame updates never redraw the lanes.
- The tooltip is one fixed-position `div` shared by all canvas hovers,
  clamped to the viewport edge.
- The table view is not optional — it is the accessibility relief channel
  and must render every finding.

## Mark grammar

- **Duration** (has `end`): rounded span bar (2px radius, ~12px tall),
  stroked 2px in the surface color so adjacent spans stay separable.
- **Instant**: filled dot, ≥8px hit size (r≈4.5 + surface stroke).
- **Third layer in the same lane**: ring (hollow) dot.
- While drawing, push `{x0, x1, y0, y1, i}` into a `hitboxes` array; hover
  and click both run the same hit test, so the tooltip names exactly what a
  click will do.

## Interaction contract

- Click a finding/mark → `video.currentTime = t − 0.5; play()`. Click empty
  ruler → proportional seek.
- Follow mode: on `timeupdate`, mark the last list row with `t ≤ now` active
  and `scrollIntoView({block: 'nearest'})`.
- Filters (category, layer) re-render the list; lanes always show all.
- Subtitles: the VTT string is inlined in DATA, attached via
  `URL.createObjectURL(new Blob([vtt], {type:'text/vtt'}))` on a
  `<track default>`, with an on/off button. (A `file://` page cannot load a
  sidecar .vtt — the Blob URL is the workaround.)
- Theme toggle and window resize re-run every canvas draw. The draw code
  reads colors live from CSS variables (`getComputedStyle`), so redrawing
  is all that recoloring requires.

## DATA contract

The template renders `const DATA = /*__DATA__*/;` — the builder replaces the
marker with one JSON object:

```jsonc
{
  "meta": {
    "title": "Asset title",
    "subtitle": "provenance line: run ids · model · row counts",
    "duration_sec": 82.4,
    "note": "data caveats shown under the timeline (see below)"
  },
  "media": { "video": "https://…presigned…" },   // the ONLY remote fetch
  "vtt": "WEBVTT\n\n0\n00:00:01.000 --> …",       // inline subtitles, or null
  "rating": {                                     // header pills + prose block
    "suggested": "…", "recommendation": "…", "profile": "…",
    "justification": "…", "elements": "…"
  },
  "categories": ["VIOLENCE", "SUBSTANCE"],        // lane order, top to bottom
  "attested_none": ["SEXUAL"],                    // shown in the prose block
  "findings": [{
    "t": 12.3,            // seconds; instants have only t
    "end": 15.1,          // present ⇒ drawn as a span bar
    "ts": "00:12",        // preformatted mm:ss
    "cat": "VIOLENCE",    // must match a categories[] lane
    "layer": "visual",    // visual | spoken | word-list → series 1/2/3
    "act": "firearm",     // machine token (kind/act)
    "evidence": "…",      // human-readable detail or verbatim quote
    "src": "shot 41",     // provenance: which shot/cue produced it
    "verified": true      // quote re-verified against source ⇒ ✓ badge
  }],
  "shots": [{ "id": 1, "start": 0.0, "end": 3.2,
              "thumb": "data:image/jpeg;base64,…" }]  // filmstrip; thumb may be null
}
```

Semantics the page must not blur (put them in `meta.note`):

- The page shows **evidence, never a verdict**. Findings are observables;
  classification against a rulebook is the consumer's job.
- `attested_none` means the layer was asked and reported none (`[]`).
  An **absent** key means *not assessed* — never claim "none" for it.
- Absence of a visual finding is not proof of absence (frame sampling can
  miss dark or sub-second events).
- Spoken/word-list evidence should be re-verified verbatim against the
  transcript before being marked `verified`.

## Builder pattern

One Python script per page family; per-asset settings in a small
`config.json` (source URIs + title + rating block). Shape:

```python
# 1. read workflow outputs from S3 (JSON / JSONL)
# 2. join rows into DATA["findings"]: map evidence acts/kinds -> the
#    consumer's categories; attach timestamps from shots.json / cue timings;
#    re-verify quotes against the transcript text
# 3. build the VTT string from the diarized transcript
# 4. thumbnails: fetch, downscale (~height 40px), base64 data-URIs
# 5. presign ONLY the video (s3v4 signature)
# 6. emit: template.read_text().replace("/*__DATA__*/",
#          json.dumps(data, ensure_ascii=False))
```

Rules:

- Never hand-edit a generated page — regenerate. Presigned URLs die when the
  signing session's token expires, regardless of the requested expiry, so
  regeneration is routine; say so in the page or a README.
- An evidence act may map to several consumer categories — emit one finding
  row per (evidence, category) pair, so "attested none" is only claimed for
  categories no evidence maps to.
- Keep the builder deterministic: same inputs, same page.

## Delivery checklist

- Opens from `file://` and works with the network blocked except the
  presigned video.
- Both themes checked; canvases recolor on toggle; resize redraws.
- Every lane labeled; every finding present in list AND table view.
- Timecodes monospace/tabular; counts in header pills; `meta.note` states
  the data caveats.
- No external requests besides the presigned media (no CDN, fonts, or
  analytics).
