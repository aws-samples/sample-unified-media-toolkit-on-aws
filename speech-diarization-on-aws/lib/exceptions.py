"""Named workflow conditions. The class NAME is the stable code: it lands
in the execution's error field (and Step Functions ErrorEquals matches
class names), so a consumer or agent branches on it without parsing
message text. Message bodies still name the condition in prose —
evidence for a human, code for a machine."""


class NoAudioStream(RuntimeError):
    """The source carries no audio stream — media-processing refused the
    wav extraction by name (NoCompatibleOutputGroups). There is nothing
    to diarize: a terminal content fact, not a retryable fault. Agents
    should probe the source first (process_media `metadata` is ~2 Lambda
    invocations) and skip the audio chain instead of submitting."""


class NoSpeechDetected(RuntimeError):
    """The source has audio but no usable speech survived the pipeline
    (music-only, ambience, silence-with-track): zero high-quality speech
    embeddings reached clustering. The message carries the VAD/embedding
    evidence so a genuinely speechless source is distinguishable from an
    upstream regression that silently dropped speech."""
