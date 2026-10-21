# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Releases are cut from `main`: per-workflow `.version` files carry the release
> version, `./build-deploy-s3-dist.sh` publishes the
> five-region artifacts, and the README's **Releases** section records the
> template / one-click / client-kit URLs per version (source code is served
> by the GitHub release).

## [0.1.0] - 2026-10-21

First semantically versioned release (succeeds the `alpha.*` line;
last alpha: `alpha.20261017a`).

### Added

- **audiostem-separation — `speech_safe` output** (opt-in
  `audio_speech_safe.wav`): attenuated original mix patched into
  separator-deleted spans with crossfades — a playback/mixing product that
  restores speech the speech stem deletes (whispers, phone voices,
  background speech). Ships with a new `speech_stem_gaps` lane in
  `stem_activity.json` (mix active ∧ stem silent — an upper bound on
  deleted speech), gateway parameters and tool schema, and
  refuse-before-spend output validation.
- **compare-models skill**: N-model conversation-analysis /
  shot-metadata-enrichment comparison page — scoreboard, per-shot field
  matrix with divergence-highlighted cells, shot-scoped playback
  (validated on a six-model run).
- **Open-source posture**: `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, README
  Security and License (MIT-0) sections; per-workflow README consolidation
  with Design record sections preserving every code-cited § number.
- **Licenses**: Bandit v2 separation weights recorded as CC BY-SA 4.0
  (Zenodo record) alongside the Apache-2.0 code entry.

### Changed

- **speech-diarization accuracy** (pooled 19-episode caption battery
  11.70% → 11.08% WER): VAD defaults moved to the measured knee
  (`seg_threshold` 0.30, `min_speech_duration_sec` 0); word-to-sentence
  assembly is now a strict partition with seam-twin dedup for true decode
  twins; ASR decode-hygiene knobs (`condition_on_previous_text`,
  `no_speech_threshold`, `log_prob_threshold`) exposed at config level
  with defaults pinned to faster-whisper 1.2.1.
- **audiostem gateway**: strict boolean parameter parsing.

### Fixed

- **speech-diarization duplicate boundary words**: a word object assigned
  to two sentences was emitted twice (~0.64% of words since the original
  pipeline); residual duplicates now 0.
- Build scripts: stale `docker/`-era run-from comments and the bundle's
  dirty-tree error message.

### Removed

- Superseded `e2e-media-workflow` skill (the gateway's analyze-video chain
  is the single source of truth).

## [alpha series] — 2026-09-21 … 2026-10-17

Pre-release line, summarized. Full per-release notes lived in the README
Releases section up to v0.1.0.

- **alpha.20261017a** — Bedrock model registry expanded (GPT-6 family
  with the effort knob, Claude Haiku 5.5; Sonnet 5.5 thinking default
  adaptive); inference accounting and limits hardened (implicit-cache
  cost accounting, explicit output ceilings on thinking-capable Claude 5
  entries, plan-time image limits); media-uploads gateway target
  (presigned single/multipart ingest uploads, 5 GiB cap); media
  workflows fail by name on silent/empty inputs (NoAudioStream /
  NoVideoStream); client-kit documentation sweep; METHOD.md;
  ap-northeast-1 region added.
- **alpha.20261014a** — output labels standardized to kebab-case (with
  poll-time aliasing for pre-rename runs); hybrid-search hardening
  (unknown collections refused at submit, pre-embed image-URI sweep at
  zero model cost, admin `replace_all` guard, retrieval paths readable
  from `list_collections`, topic chapters in semantic search); tts
  voice-catalog previews presignable.
- **alpha.20261013a** — speech-diarization pad-boundary word-resolution
  ladder (gated on a 40-run caption ground-truth battery); model
  registry refresh; audio tool-routing guidance; video-processing
  router, index-video and search-media skills in the client kit.
- **alpha.20261009b** — maintenance release.
- **alpha.20260921b** — first multi-region publication
  (ap-southeast-2 added 2026-09-22).
