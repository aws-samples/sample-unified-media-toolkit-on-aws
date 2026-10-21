# lambda_function.py
"""
Lambda entry point — routes invocations to the correct state handler based
on the "operation" field in the input event.

Pipeline overview
─────────────────
  Speech-to-text with speaker diarization.  Sequential pipeline:

  1. StateFixedChunkingForVAD
       Splits audio into fixed-size windows for parallel VAD processing.

  2. StateRunVADModel  (parallel Map, one Lambda per window)
       Runs pyannote ONNX segmentation to detect speech turns.

  3. StateDynamicChunkingBySpeechTurns
       Flattens all VAD turns, forms speech groups (adjacent turns with
       gaps < speech_group_max_gap_sec), partitions groups into dynamic
       chunks for parallel transcription + embedding.

  4. StateDetectLanguageDownloadModels
       Detects prominent language via Whisper probes, downloads whisper,
       alignment, and wtpsplit models to /tmp.

  5. Map processor (parallel, one iteration per dynamic chunk):
       a. StateRunWhisperAlignWtpsplitModels
            Transcribes speech groups via batched Whisper, refines word
            timestamps with wav2vec2 alignment, runs wtpsplit sentence
            splitting once on the full iterator word sequence.
       b. StateRunSpeechEmbeddingModel
            Reads sentences from transcript_N.json, extracts WeSpeaker
            ResNet34 speaker embeddings per sentence (high and low quality).

  6. StateGlobalSpeakersClustering
       Loads all embeddings, clusters HIGH quality embeddings via AHC with
       K-enumeration (dual scoring: elbow + combined), assigns LOW quality
       sentences via nearest centroid, SKIPPED sentences via diarization.

  7. StateMergeSpeakersTranscripts
       Loads all transcript_N.json files + global_speakers.json, assigns
       speakers to sentences, writes final_transcript.json and .vtt.

  8. StateUsageTracking
       Records execution metrics.

Configurable options
────────────────────
  All options are passed via the Step Functions event payload.  Defaults
  are defined in lib/input_schema.py and lib/cluster_sweep.py.

  VAD / Chunking  (event.data.vad)
  ─────────────────────────────────────────────────────────────────────
    fixed_chunk_size_secs         720      Fixed VAD window size (seconds).
    dynamic_speech_chunk_secs     360      Target speech seconds per dynamic chunk.
    min_silence_duration_sec      0.3      Minimum silence gap for VAD to split.
    min_speech_duration_sec       0.5      Minimum speech turn duration.
    speech_group_max_gap_sec      1.0      Max inter-turn gap within a speech group.
                                           Turns closer than this use real audio;
                                           turns farther apart start new groups.
    merge_gap_threshold_sec       0.20     Max gap for merge_short to combine
                                           adjacent short sentences.  Prevents
                                           cross-speaker merging in rapid exchanges.

  ASR  (event.data.asr_cfg per iterator, or event.data.asr)
  ─────────────────────────────────────────────────────────────────────
    model_variant                 "turbo"  Whisper model variant.
    num_workers                   2        ThreadPoolExecutor size for batched
                                           Whisper inference.
    beam_size                     2        Whisper beam search width.
    initial_prompt                ""       Whisper initial prompt (domain vocab).

  Whisper kwargs  (event.data.asr_cfg.whisper_kwargs — passthrough)
  ─────────────────────────────────────────────────────────────────────
    Any key accepted by faster-whisper WhisperModel.transcribe() except
    audio, language, word_timestamps, vad_filter.  Examples:
      temperature, patience, length_penalty, repetition_penalty,
      no_repeat_ngram_size, suppress_tokens, suppress_blank,
      condition_on_previous_text, compression_ratio_threshold,
      log_prob_threshold, no_speech_threshold, max_new_tokens,
      hallucination_silence_threshold, hotwords, prefix

  Embedding  (event.data.vad)
  ─────────────────────────────────────────────────────────────────────
    min_embedding_duration_sec    0.75     Minimum segment duration for HIGH
                                           quality embedding (used in clustering).
    soft_embedding_min_sec        0.30     Minimum segment duration for LOW
                                           quality embedding (nearest-centroid
                                           assignment only, not clustered).
                                           Sentences below this are SKIPPED and
                                           assigned via diarization overlap.

  Clustering  (event.data.clustering)
  ─────────────────────────────────────────────────────────────────────
    algorithm                     "ahc"    Clustering backend: "ahc" or "vbx".
    num_speakers                  null     Hard override for K (skips sweep).
    max_k                         50       Maximum K in sweep range.
    min_turns_per_speaker         3        k_max = N // this value.
    min_cluster_size              2        Clusters smaller than this are merged
                                           into the nearest large cluster.

    K selection (dual scoring):
    k_selection_method            "elbow"  "elbow" or "combined".  Both methods
                                           always run.  If they agree, that K is
                                           used.  If they disagree, this setting
                                           determines which result wins.
    elbow_plateau_pct             0.03     Silhouette within this fraction of
                                           sil_max is considered "plateau".
                                           Elbow = lowest k_act in plateau.

    Combined method weights (used when k_selection_method="combined"):
    sil_weight                    0.6      Silhouette weight in combined score.
    ch_weight                     0.4      Calinski-Harabasz weight.  Recommend
                                           0.0 — CH favors low K with dominant
                                           speakers.
    k_penalty                     0.02     Penalty per unit K/k_max.

    Outlier removal:
    outlier_global_percentile     97.0     Top (100-x)% by mean distance removed.
    single_speaker_threshold      0.20     Dendrogram height below which → K=1.
    unreliable_silhouette         0.20     Warn when best sil < this.

    Speaker model trimming:
    centroid_trim_percentile      90.0     Per-cluster centroid trimming.
    min_turns_for_trim            5        Skip trimming for small clusters.

    Advanced:
    apply_snorm                   false    Symmetric score normalisation.
    fa                            0.3      VBx acoustic weight.
    fb                            17.0     VBx Dirichlet prior.
    lda_dim                       0        LDA projection (0 = skip).
    max_iter                      40       VBx EM iterations.
    init_threshold                0.350    AHC init threshold for VBx.

  Logging  (event.data.log_level)
  ─────────────────────────────────────────────────────────────────────
    log_level                     "INFO"   "DEBUG" | "INFO" | "WARNING" | "ERROR"

Routing table
─────────────
  operation                                → handler module
  ──────────────────────────────────────────────────────────────────────
  StateFixedChunkingForVAD                 → fixed_chunking_for_vad
  StateRunVADModel                         → run_vad_model
  StateDynamicChunkingBySpeechTurns        → dynamic_chunking_by_speech_turns
  StateDetectLanguageDownloadModels        → detect_language_download_models
  StateRunWhisperAlignWtpsplitModels       → run_whisper_align_wtpsplit_models
  StateRunSpeechEmbeddingModel             → run_speech_embedding_model
  StateGlobalSpeakersClustering            → global_speakers_clustering
  StateMergeSpeakersTranscripts            → merge_speakers_transcripts
  StateUsageTracking                       → usage_tracking

Design notes
────────────
  • Handlers are imported lazily (inside the dispatch branch) so that a cold
    start for a lightweight state (e.g. StateFixedChunkingForVAD) does not
    pay the ORT session-load cost incurred by StateRunVADModel,
    StateRunWhisperAlignWtpsplitModels, or StateRunSpeechEmbeddingModel.
  • Each handler module is responsible for its own logging setup and any
    module-level initialisation (ORT cpuinfo fix, session cache, etc.).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(_h)


def lambda_handler(event, context):
    operation = event.get("operation")
    if not operation:
        raise ValueError("Missing 'operation'.")

    log.info(f"Routing: '{operation}'")

    if operation == "StateCheckAndPreprocess":
        from states.check_and_preprocess import lambda_function as handler
    elif operation == "StateSubmitMediaProcessing":
        from states.submit_media_processing import lambda_function as handler
    elif operation == "StateExtractProcessedAudio":
        from states.extract_processed_audio import lambda_function as handler
    elif operation == "StateFixedChunkingForVAD":
        from states.fixed_chunking_for_vad import lambda_function as handler
    elif operation == "StateRunVADModel":
        from states.run_vad_model import lambda_function as handler
    elif operation == "StateDynamicChunkingBySpeechTurns":
        from states.dynamic_chunking_by_speech_turns import lambda_function as handler
    elif operation == "StateDetectLanguageDownloadModels":
        from states.detect_language_download_models import lambda_function as handler
    elif operation == "StateRunWhisperAlignWtpsplitModels":
        from states.run_whisper_align_wtpsplit_models import lambda_function as handler
    elif operation == "StateRunSpeechEmbeddingModel":
        from states.run_speech_embedding_model import lambda_function as handler
    elif operation == "StateGlobalSpeakersClustering":
        from states.global_speakers_clustering import lambda_function as handler
    elif operation == "StateMergeSpeakersTranscripts":
        from states.merge_speakers_transcripts import lambda_function as handler
    elif operation == "StateSeparateSpeakerChannels":
        from states.separate_speaker_channels import lambda_function as handler
    elif operation == "StateUsageTracking":
        from states.usage_tracking import lambda_function as handler
    else:
        raise ValueError(f"Unknown operation: '{operation}'.")

    return handler.lambda_handler(event, context)
