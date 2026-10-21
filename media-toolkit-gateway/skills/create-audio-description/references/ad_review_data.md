# AD review page — build contract

The page UI is FROZEN in `template_ad_review.html` beside this file.
**Start from a copy; never rebuild or restyle** — substitute the
tokens and nothing else.

| token | replace with |
|---|---|
| `/*__DATA__*/` | the DATA JSON (one `json.dumps`, compact) |
| `__ASSET__` | display name of the asset (title) |

Validate afterwards: the embedded JSON parses, no `__` token survives.

## DATA contract

```
asset str · asset_label str
media {video, mix}   presigned urls (mix only when the ad_mix was
                     rendered; the page shows "mix unavailable"
                     gracefully when absent)
view {}              the joined review document below
narration_vtt str    the AD cues as WebVTT (omit when unavailable)
dialogue_vtt str     the dialogue transcript as WebVTT (optional)
```

## view — joined from the generate_ad run outputs

All inputs come from the AD run prefix (presign + fetch each JSON via
the gateway; never fetch at page runtime):

```
turns[]      final_transcript sentences joined with the cast speaker
             map: {start, end, speaker, name, text<=160 chars}
windows[]    windows.json: {window_id, start, end, seconds,
             word_budget, scene_entry, music_dominant,
             people_on_screen}
cues[]       cue_ledger.json joined with render_manifest takes by
             window_id — EVERY state kept (placed/dropped/unwritten):
             {id, window_id, kind, state, at (ledger's, else the
             take's), dur, tempo (ledger's, else take's), text,
             budget, words, necessity, important, why_needed[],
             violations[], disposition, oracle}
gaps[]       ad_gaps.json verbatim
characters[] character_map.json .characters (speakable-from times)
duration_sec max sentence/window end · fps from the manifest
video/mix    booleans: does the media exist
summary      {windows, cues_placed (state=placed & kind=cue),
             cues_dropped, cues_unwritten, narration_sec (sum of
             placed dur), window_sec, duty_windows (scene_entry),
             duty_hit (duty windows with a placed cue), gaps,
             cost (usage sum), usage}
```

Principles: cue text, placement and every violation come **verbatim
from the run's own ledger** — the page is evidence review, never a
re-judgment. A missing input drops its lane by omission; never
fabricate an empty document.
