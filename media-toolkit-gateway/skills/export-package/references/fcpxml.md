# FCPXML export blueprint

Emit a single-sequence FCPXML timeline that DaVinci Resolve (and Final Cut,
Premiere via converters) imports, with the analysis metadata expressed as the
editor-native constructs each tool understands: clip on the spine, markers,
keyword ranges, and a caption track. Package it in a self-contained folder with
the proxy + stems so it relinks by relative path.

## Package layout

```
outputs/export-package/{ASSET_ID}/
├── {ASSET_ID}.fcpxml            # the timeline; asset src = ABSOLUTE file:// into media/
├── media/
│   ├── proxy.mp4                # downloaded via gateway-presigned URL
│   ├── speech.wav               # audio-stem outputs
│   ├── music.wav
│   └── sfx.wav
├── captions.srt                 # sidecar caption file (belt-and-suspenders)
└── metadata/                    # the raw analysis JSON, for reference
    ├── shots.json scenes.json transcript.json conversation.json structure.json faces.json
```

The `<asset>` `src` MUST be an ABSOLUTE `file://` URL to the downloaded media in
this package's `media/` folder, e.g.
`file:///…/outputs/export-package/{ASSET_ID}/media/proxy.mp4`
(percent-encode spaces). Do NOT use a relative `./media/...` path — verified in
a real import, DaVinci Resolve's FCPXML importer does NOT resolve relative
`media-rep src` and fails with "N of N clips were not yet found". The package is
still portable in practice (media sits beside the FCPXML); if it is ever moved,
the paths are rewritten or the user relinks from `media/`. Download proxy + 3
stems by presigning each with the gateway (`media-outputs___get_download_url`)
and fetching the bytes into `media/`, then write their absolute file:// URLs.

## Frame rate — read it, never hardcode

Take the frame rate from `shots.json` (its fps / `actual_frame_count` ÷
`actual_duration_sec`) or the proxy's `process_media` probe. A 23.976
source is 24000/1001. Express FCPXML durations as rational `N/Ds` on the frame
grid: `frameDuration = 1001/24000s` for 23.976. Round every time to the nearest
frame boundary: `frames = round(t * fps)`, `time = "{frames*1001}/24000s"`.
Derive the ratios from the ACTUAL fps; the 1001/24000 values above are
the 23.976 example only.

## Document skeleton (FCPXML 1.9)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
  <resources>
    <format id="r1" name="FFVideoFormat720p2398" frameDuration="1001/24000s"
            width="1280" height="720"/>
    <asset id="a1" name="{ASSET_ID}" start="0s" hasVideo="1" hasAudio="1"
           format="r1" duration="{DUR}s">
      <media-rep kind="original-media" src="file:///ABS/PATH/{ASSET_ID}/media/proxy.mp4"/>
    </asset>
    <!-- optional stem assets a2/a3/a4 → file:///ABS/PATH/.../media/speech.wav etc. (audio only) -->
  </resources>
  <library>
    <event name="{ASSET_ID} — analyze-video">
      <project name="{ASSET_ID}">
        <sequence format="r1" duration="{DUR}s"
                  tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
          <spine>
            <!-- primary clip = whole proxy; markers/keywords are its children -->
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

Width/height and the format `name` follow the proxy's actual resolution/fps.
Use `tcFormat="NDF"` for whole-frame-rate content, keep it consistent with the
format. `duration` on the sequence = the asset duration rounded to the frame.

## Metadata → timeline mapping

Put the whole proxy as ONE `asset-clip` (or `clip`) on the spine spanning the
programme; hang markers and keyword ranges off it. `offset` is the clip's
position on the sequence; `start`+`duration` address media time; child
`marker`/`keyword` `start` values are in the clip's own time base.

- **Shots** → a `chapter-marker` (or plain `marker`) at each shot start, named
  `Shot {id}` (+ shot_type when not "content"). Shots give an editor cut-point
  navigation. If the user prefers cut-up clips instead of one clip + markers,
  split the spine into one `asset-clip` per shot referencing the same asset with
  each shot's media `start`/`duration` — but the default is one clip + markers
  (cleaner relink, keeps audio continuous).
- **Scenes** → `keyword` ranges (`<keyword start=.. duration=.. value="Scene {id}: {setting}"/>`).
  Keyword ranges show up in Resolve/FCP as coloured spans you can filter by.
- **Conversation chapters** → `chapter-marker` at each segment start, named with
  the segment topic (these become chapter navigation points).
- **PSA break candidates** → `marker completed="0"` (to-do / red markers) named
  `BREAK? score {n}` at each candidate time — visually distinct from chapters.
- **Faces / people** → `keyword` ranges per appearance span, value = person name
  (named) or `Unknown {id}`. One range per appearance; editors can filter the
  timeline by cast member. Cap total keyword elements sanely (merge sub-second
  gaps) so the file stays reasonable.
- **Speech / music / sfx stems** → add each stem WAV as an audio `asset-clip` on
  a secondary lane (`lane="-1"`, `-2`, `-3`) with an audio `role`
  (`dialogue`/`music`/`effects`), `offset="0s"` spanning the programme. Set them
  to a muted/secondary state note in the clip name (e.g. `SPEECH stem`) since the
  proxy already carries the mix — they are there for the editor to solo, not to
  double the mix.
- **Caption track (transcript)** → a `<caption>` per transcript sentence inside
  the primary clip, using a caption format resource, e.g.:
  ```xml
  <clip ...>
    <caption lane="1" start="{t0}" duration="{t1-t0}" role="iTT?captionFormat=ITT.en">
      <text><text-style ref="ts1">{sentence text}</text-style></text>
      <text-style-def id="ts1"><text-style font=".AppleSystem" fontSize="60"/></text-style-def>
    </caption>
  </clip>
  ```
  Resolve imports FCPXML captions into a subtitle track. ALSO always write the
  sidecar `captions.srt` (numbered cues, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, blank
  line between) as a fallback that every NLE reads.

## Validity rules that bite

- Every `start`/`duration`/`offset`/`frameDuration` is a rational on the frame
  grid (`{n}/{d}s`) or `0s`. Never emit float seconds like `12.5s` — snap to
  frames.
- Marker `start` is in the parent clip's time base; if the clip `start` is `0s`
  and `offset` is `0s`, marker time == sequence time == programme time.
- Escape XML entities in every name/text (`& < > " '`).
- Keep marker/keyword/caption ordering by `start` ascending within a parent.
- Test import target is Resolve; if a construct is rejected, prefer the simpler
  form (plain `marker` over `chapter-marker`, one clip + markers over per-shot
  clips) and note the downgrade in the summary.

## SRT sidecar format (always emit)

```
1
00:00:26,978 --> 00:00:29,258
So these guys just disappeared?

2
00:00:29,698 --> 00:00:33,960
Yeah, all on that stretch of road right above El Matador.
```
Times from the transcript sentences (`start`/`end`), comma decimal, millisecond
precision.

## FCP7 XML flavor (Premiere) — always emitted alongside

`{ASSET_ID}.fcp7.xml` is the same timeline as legacy Final Cut Pro 7
`xmeml version="4"`, which Adobe Premiere Pro imports natively (File →
Import → the .xml). Both files always ship; they reference the same
`media/` folder. Mapping downgrades (accepted, user decision
2026-10-07):

- xmeml has NO keyword ranges and NO caption construct. Scenes and
  face appearances become DURATION markers (xmeml markers take
  `<in>`/`<out>`); the transcript rides ONLY the sidecar `captions.srt`
  (Premiere imports SRT natively).
- Premiere ignores most marker styling, so the CATEGORY is encoded in
  the marker NAME prefix — `SHOT` / `SCENE` / `CHAPTER` / `BREAK?` /
  `FACE:` — making markers filterable by search.
- Instant markers use `<out>-1</out>`; range markers carry real out
  frames.
- Frame rate uses the xmeml convention: 23.976 is
  `<timebase>24</timebase><ntsc>TRUE</ntsc>` (never a rational); frames
  are counted on the TRUE rate (`round(t * 24000/1001)`).
- Stems map BETTER here than in FCPXML: each WAV is a real audio
  `<track>` with a full-length clipitem.
