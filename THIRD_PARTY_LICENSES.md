# Third-Party Models, Packages, and Binaries

Inventory for security/legal review of every third-party component the
Unified Media Toolkit runs or redistributes, with versions where
pinned, licenses, and links to each project's LICENSE file.

Scope notes:

- **Foundation packages deliberately omitted** per review guidance:
  boto3/botocore/s3transfer, numpy, scipy, Pillow/PIL, torch (torch is
  build-time only; no runtime image ships it).
- **Direct dependencies are listed.** Fully pinned transitive sets live
  in each workflow's `requirements.txt` (and `export/requirements.txt`
  for the build-time converters); those files are the authoritative
  manifests for SCA tooling.
- **AWS-managed services are not third-party redistributions** and are
  excluded from the tables: Amazon Rekognition (face-recognition),
  Amazon Bedrock models (Anthropic Claude and Amazon Nova families via
  bedrock-inference), AWS Elemental MediaConvert. They are governed by
  the AWS service terms of the deploying account.
- License links target the project's LICENSE file on its default
  branch; Hugging Face links target the model card (license shown in
  its header).
- **Distribution model:** the toolkit distributes SOURCE (Dockerfiles,
  build scripts, requirements pins). Container images — including
  FFmpeg/OpenH264 and all pip-installed wheels — are built inside the
  deploying account's own CodeBuild at stack creation; this project
  does not redistribute those binaries. The one project-hosted
  redistribution channel is the seeded MODEL archives (converted ONNX
  tarballs served from the project's S3/CloudFront), all of which are
  Apache-2.0/MIT-licensed per the table below.

## 1. ML models (weights we convert, package, and run)

| Model | Version / variant | Format as deployed | Used by | License | License file / card |
|---|---|---|---|---|---|
| DINOv2 ViT-S/14 (with registers) | `dinov2_vits14_reg` | ONNX FP32 (opset 17), baked into image | shot-segmentation (frame embeddings, shot/scene detection) | Apache-2.0 | https://github.com/facebookresearch/dinov2/blob/main/LICENSE |
| OpenAI Whisper | large-v3-turbo (+ `cy`/`ar` variants; `base` multilingual in tts QC) | CTranslate2 INT8 | speech-diarization (ASR), tts (ASR round-trip QC) | MIT (code and weights) | https://github.com/openai/whisper/blob/main/LICENSE · https://huggingface.co/openai/whisper-large-v3-turbo |
| pyannote segmentation (community-1) | community-1 | ONNX | speech-diarization (VAD / powerset segmentation, splice binder) | Weights: **CC-BY-4.0** (verified 2026-10-06 via the HF card; attribution required on redistribution — our seeded tarball must carry the attribution/NOTICE). Toolkit (pyannote-audio): MIT. Card is gated (auto-approve form) — gating is access mechanics, redistribution rights come from CC-BY-4.0 | https://huggingface.co/pyannote/speaker-diarization-community-1 · https://github.com/pyannote/pyannote-audio/blob/develop/LICENSE |
| WeSpeaker speaker embedding | ResNet34 (256-dim) | ONNX | speech-diarization (speaker embeddings for clustering) | Apache-2.0 (toolkit and weights). Training data: VoxCeleb, CC BY 4.0 (verified 2026-10-06) — attribution attaches to dataset redistribution, which this project does not do; cite VoxCeleb in docs as a courtesy | https://github.com/wenet-e2e/wespeaker/blob/master/LICENSE · https://mm.kaist.ac.kr/datasets/voxceleb/ |
| wav2vec 2.0 XLSR-53 (forced-alignment base model) | large-xlsr-53 | ONNX INT8 | speech-diarization (word-level forced alignment) | Apache-2.0 | https://huggingface.co/facebook/wav2vec2-large-xlsr-53 — per-language fine-tunes listed in Appendix A |
| wtpsplit SaT (Segment-any-Text) | SaT | ONNX | speech-diarization (sentence segmentation, 85+ languages) | MIT | https://github.com/segment-any-text/wtpsplit/blob/main/LICENSE |
| CosyVoice 3 — LLM | Qwen2 backbone | ONNX INT4 | tts (text → speech tokens) | Apache-2.0 (Qwen2 small models also Apache-2.0) | https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE |
| CosyVoice 3 — Flow | Matcha-style CFM estimator | ONNX FP32 | tts (speech tokens → mel) | Apache-2.0 (Matcha-TTS lineage MIT) | https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE · https://github.com/shivammehta25/Matcha-TTS/blob/main/LICENSE |
| CosyVoice 3 — HiFT vocoder | — | ONNX | tts (mel → 24 kHz waveform) | Apache-2.0 (HiFi-GAN lineage MIT) | https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE |
| CosyVoice 3 — speech tokenizer | — | ONNX | tts (reference audio → speech tokens at voice registration) | Apache-2.0 | https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE |
| CAMPPlus speaker embedding | 192-dim | ONNX | tts (voice-registration speaker embedding) | Apache-2.0 (3D-Speaker / ModelScope release) | https://github.com/modelscope/3D-Speaker/blob/main/LICENSE |
| Bandit v2 (cinematic audio source separation) | v2, split ONNX | ONNX | audiostem-separation (speech / music / effects stems) | Code: Apache-2.0 (verified 2026-10-06 against the LICENSE file). Weights: CC BY-SA 4.0 (verified 2026-10-18 against the Zenodo record). | https://github.com/kwatcharasupat/bandit-v2/blob/main/LICENSE · https://zenodo.org/records/12701995 |
| DeepFilterNet3 | DFN3 (upstream Rikorose checkpoint) | ONNX (enc / erb_dec / df_dec) | speech-enhancement (in-place dialogue noise suppression) | MIT OR Apache-2.0 (dual) | https://github.com/Rikorose/DeepFilterNet/blob/main/LICENSE-MIT · https://github.com/Rikorose/DeepFilterNet/blob/main/LICENSE-APACHE |

## 2. Runtime Python packages (direct, non-foundation)

| Package | Pinned version(s) | Used by | License | License file |
|---|---|---|---|---|
| onnxruntime | 1.24.2 (diar, audiostem, shot-seg) · 1.18.0 (tts) · 1.24.3 (export) | all ONNX model inference | MIT | https://github.com/microsoft/onnxruntime/blob/main/LICENSE |
| faster-whisper | 1.2.1 (diar) · 1.0.3 (tts) | Whisper inference wrapper | MIT | https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE |
| CTranslate2 (transitive via faster-whisper, load-bearing) | per faster-whisper pin | Whisper INT8 inference engine | MIT | https://github.com/OpenNMT/CTranslate2/blob/master/LICENSE |
| tokenizers | 0.22.2 (diar) · 0.21.4 (tts) | tokenization for ASR/TTS | Apache-2.0 | https://github.com/huggingface/tokenizers/blob/main/LICENSE |
| transformers | 4.51.3 (tts) | CosyVoice LLM tokenizer/config plumbing | Apache-2.0 | https://github.com/huggingface/transformers/blob/main/LICENSE |
| soundfile | 0.13.1 (tts, export) | WAV I/O (bundles libsndfile, LGPL-2.1) | BSD-3-Clause | https://github.com/bastibe/python-soundfile/blob/master/LICENSE |
| pyvips (`[binary]`) | 3.0.0 (SME, frame-inspection, face-recognition) | image compositing/grids (bundles libvips, LGPL-2.1+) | MIT (binding) | https://github.com/libvips/pyvips/blob/master/LICENSE.txt |
| lancedb | 0.33.0 (hybrid-search) | vector store | Apache-2.0 | https://github.com/lancedb/lancedb/blob/main/LICENSE |
| lance-namespace / lance-namespace-urllib3-client | 0.8.2 (hybrid-search) | LanceDB namespace client | Apache-2.0 | https://github.com/lancedb/lance-namespace/blob/main/LICENSE |
| pyarrow | 24.0.0 (hybrid-search) | columnar data interchange | Apache-2.0 | https://github.com/apache/arrow/blob/main/LICENSE.txt |
| pydantic / pydantic_core | 2.13.4 / 2.46.4 (hybrid-search) | schema validation | MIT | https://github.com/pydantic/pydantic/blob/main/LICENSE |
| tqdm | 4.67.3 (diar) · 4.68.0 (hybrid-search) | progress reporting (non-TTY silent) | MPL-2.0 + MIT (dual) | https://github.com/tqdm/tqdm/blob/master/LICENCE |
| opencc-python-reimplemented | 0.1.7 (tts) | Chinese script conversion for TTS text | Apache-2.0 | https://github.com/yichen0831/opencc-python/blob/master/LICENSE.txt |
| deprecation | 2.1.0 (hybrid-search) | API deprecation helper | Apache-2.0 | https://github.com/briancurtin/deprecation/blob/master/LICENSE |
| annotated-types | 0.7.0 (hybrid-search) | pydantic support | MIT | https://github.com/annotated-types/annotated-types/blob/main/LICENSE |
| typing_extensions / typing-inspection | 4.15.0 / 0.4.2 (hybrid-search) | typing backports | PSF-2.0 / MIT | https://github.com/python/typing_extensions/blob/main/LICENSE |
| urllib3 | 2.6.3 (hybrid-search) | HTTP client | MIT | https://github.com/urllib3/urllib3/blob/main/LICENSE.txt |
| python-dateutil | 2.9.0.post0 | date parsing | Apache-2.0 + BSD-3 (dual) | https://github.com/dateutil/dateutil/blob/master/LICENSE |
| jmespath | 1.0.1 | JSON query (boto family) | MIT | https://github.com/jmespath/jmespath.py/blob/develop/LICENSE.txt |
| packaging | 26.x | version parsing | Apache-2.0 + BSD-2 (dual) | https://github.com/pypa/packaging/blob/main/LICENSE |
| six | 1.17.0 | py2/3 compat (transitive) | MIT | https://github.com/benjaminp/six/blob/master/LICENSE |

## 3. Native binaries and system libraries built into container images

| Component | Version | Image(s) | License | License file | Notes |
|---|---|---|---|---|---|
| FFmpeg (ffmpeg, ffprobe) | n6.1.2 | media-processing | LGPL-2.1+ (built `--disable-static`, no GPL components enabled). Built from source in the deploying account's CodeBuild — this project distributes the build recipe, not binaries | https://github.com/FFmpeg/FFmpeg/blob/master/COPYING.LGPLv2.1 | LGPL source archive shipped in-image at `/opt/sources/source_ffmpeg.zip`, satisfying source availability wherever the built image travels |
| OpenH264 | 2.6.0 | media-processing | BSD-2-Clause (copyright). Patent note: Cisco's H.264 patent grant covers only Cisco's pre-compiled binary; this image BUILDS FROM SOURCE in the deploying account's CodeBuild, so any H.264 patent-pool consideration sits with the deployer, not with this project as a distributor | https://github.com/cisco/openh264/blob/master/LICENSE | Source archive shipped in-image (`source_openh264.zip`) |
| BMX (mxf2raw) | `main` | media-processing | BSD-3-Clause | https://github.com/bbc/bmx/blob/main/COPYING | MXF duration/timecode/essence probing |
| uriparser | 1.0.0 | media-processing | BSD-3-Clause | https://github.com/uriparser/uriparser/blob/master/COPYING | BMX dependency |
| YASM | 1.3.0 | media-processing (build stage only) | BSD-2-Clause | https://github.com/yasm/yasm/blob/master/COPYING | assembler for FFmpeg build; not in the runtime layer |
| libvips | via pyvips[binary] 3.0.0 wheels | SME, frame-inspection, face-recognition | LGPL-2.1+ | https://github.com/libvips/libvips/blob/master/LICENSE | bundled in the manylinux wheel |
| libsndfile | via soundfile 0.13.1 wheels | tts | LGPL-2.1 | https://github.com/libsndfile/libsndfile/blob/master/COPYING | bundled in the wheel |
| libgomp | distro package (AL2023) | audiostem | GPL-3.0 with GCC Runtime Library Exception | https://github.com/gcc-mirror/gcc/blob/master/COPYING.RUNTIME | standard GCC OpenMP runtime, installed via microdnf |
| AWS Lambda Python base image | public.ecr.aws/lambda/python 3.13.x | all container workflows | AWS-provided base image | https://gallery.ecr.aws/lambda/python | includes awslambdaric (Apache-2.0) |


## Appendix A — XLSR fine-tuned forced-alignment models (per-language license audit, 2026-10-06)

| lang | model | license | note |
|---|---|---|---|
| ar | [jonatasgrosman/wav2vec2-large-xlsr-53-arabic](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic) | apache-2.0 |  |
| ca | [softcatala/wav2vec2-large-xlsr-catala](https://huggingface.co/softcatala/wav2vec2-large-xlsr-catala) | apache-2.0 |  |
| cs | [comodoro/wav2vec2-xls-r-300m-cs-250](https://huggingface.co/comodoro/wav2vec2-xls-r-300m-cs-250) | apache-2.0 |  |
| de | [jonatasgrosman/wav2vec2-large-xlsr-53-german](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-german) | apache-2.0 |  |
| el | [jonatasgrosman/wav2vec2-large-xlsr-53-greek](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-greek) | apache-2.0 |  |
| en | [jonatasgrosman/wav2vec2-large-xlsr-53-english](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english) | apache-2.0 |  |
| es | [jonatasgrosman/wav2vec2-large-xlsr-53-spanish](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-spanish) | apache-2.0 |  |
| eu | [stefan-it/wav2vec2-large-xlsr-53-basque](https://huggingface.co/stefan-it/wav2vec2-large-xlsr-53-basque) | apache-2.0 |  |
| fa | [jonatasgrosman/wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian) | apache-2.0 |  |
| fi | [jonatasgrosman/wav2vec2-large-xlsr-53-finnish](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-finnish) | apache-2.0 |  |
| fr | [jonatasgrosman/wav2vec2-large-xlsr-53-french](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french) | apache-2.0 |  |
| gl | [ifrz/wav2vec2-large-xlsr-galician](https://huggingface.co/ifrz/wav2vec2-large-xlsr-galician) | IC (Individual Contributor) | no license declared — rights reserved by default, pending author clarification |
| he | [imvladikon/wav2vec2-xls-r-300m-hebrew](https://huggingface.co/imvladikon/wav2vec2-xls-r-300m-hebrew) | IC (Individual Contributor) | no license declared — rights reserved by default, pending author clarification |
| hi | [theainerd/Wav2Vec2-large-xlsr-hindi](https://huggingface.co/theainerd/Wav2Vec2-large-xlsr-hindi) | IC (Individual Contributor) | no license declared — rights reserved by default, pending author clarification |
| hr | [classla/wav2vec2-xls-r-parlaspeech-hr](https://huggingface.co/classla/wav2vec2-xls-r-parlaspeech-hr) | IC (Individual Contributor) | no license declared — rights reserved by default, pending author clarification |
| hu | [jonatasgrosman/wav2vec2-large-xlsr-53-hungarian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-hungarian) | apache-2.0 |  |
| it | [jonatasgrosman/wav2vec2-large-xlsr-53-italian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-italian) | apache-2.0 |  |
| ja | [jonatasgrosman/wav2vec2-large-xlsr-53-japanese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese) | apache-2.0 |  |
| ka | [xsway/wav2vec2-large-xlsr-georgian](https://huggingface.co/xsway/wav2vec2-large-xlsr-georgian) | apache-2.0 |  |
| ko | [kresnik/wav2vec2-large-xlsr-korean](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean) | apache-2.0 |  |
| lv | [jimregan/wav2vec2-large-xlsr-latvian-cv](https://huggingface.co/jimregan/wav2vec2-large-xlsr-latvian-cv) | apache-2.0 |  |
| ml | [gvs/wav2vec2-large-xlsr-malayalam](https://huggingface.co/gvs/wav2vec2-large-xlsr-malayalam) | apache-2.0 |  |
| nl | [jonatasgrosman/wav2vec2-large-xlsr-53-dutch](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-dutch) | apache-2.0 |  |
| nn | [NbAiLab/nb-wav2vec2-300m-nynorsk](https://huggingface.co/NbAiLab/nb-wav2vec2-300m-nynorsk) | apache-2.0 |  |
| no | [NbAiLab/nb-wav2vec2-300m-bokmaal](https://huggingface.co/NbAiLab/nb-wav2vec2-300m-bokmaal) | apache-2.0 |  |
| pl | [jonatasgrosman/wav2vec2-large-xlsr-53-polish](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-polish) | apache-2.0 |  |
| pt | [jonatasgrosman/wav2vec2-large-xlsr-53-portuguese](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-portuguese) | apache-2.0 |  |
| ro | [gigant/romanian-wav2vec2](https://huggingface.co/gigant/romanian-wav2vec2) | apache-2.0 |  |
| ru | [jonatasgrosman/wav2vec2-large-xlsr-53-russian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian) | apache-2.0 |  |
| sk | [comodoro/wav2vec2-xls-r-300m-sk-cv8](https://huggingface.co/comodoro/wav2vec2-xls-r-300m-sk-cv8) | apache-2.0 |  |
| sl | [anton-l/wav2vec2-large-xlsr-53-slovenian](https://huggingface.co/anton-l/wav2vec2-large-xlsr-53-slovenian) | apache-2.0 |  |
| sv | [KBLab/wav2vec2-large-voxrex-swedish](https://huggingface.co/KBLab/wav2vec2-large-voxrex-swedish) | cc0-1.0 | public domain |
| te | [anuragshas/wav2vec2-large-xlsr-53-telugu](https://huggingface.co/anuragshas/wav2vec2-large-xlsr-53-telugu) | apache-2.0 |  |
| tl | [Khalsuu/filipino-wav2vec2-l-xls-r-300m-official](https://huggingface.co/Khalsuu/filipino-wav2vec2-l-xls-r-300m-official) | apache-2.0 |  |
| tr | [mpoyraz/wav2vec2-xls-r-300m-cv7-turkish](https://huggingface.co/mpoyraz/wav2vec2-xls-r-300m-cv7-turkish) | cc-by-4.0 | attribution required |
| uk | [Yehor/wav2vec2-xls-r-300m-uk-with-small-lm](https://huggingface.co/Yehor/wav2vec2-xls-r-300m-uk-with-small-lm) | apache-2.0 |  |
| ur | [kingabzpro/wav2vec2-large-xls-r-300m-Urdu](https://huggingface.co/kingabzpro/wav2vec2-large-xls-r-300m-Urdu) | apache-2.0 |  |
| zh | [jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn) | apache-2.0 |  |
