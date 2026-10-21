"""Named workflow conditions. The class NAME is the stable code: it lands
in the execution's error field (and Step Functions ErrorEquals matches
class names), so a consumer or agent branches on it without parsing
message text. Message bodies still name the condition in prose —
evidence for a human, code for a machine."""


class NoAudioStream(RuntimeError):
    """The source carries no audio stream — media-processing refused the
    wav extraction by name (NoCompatibleOutputGroups). There is nothing
    to separate: a terminal content fact, not a retryable fault. Agents
    should probe the source first (process_media `metadata` is ~2 Lambda
    invocations) and skip the audio chain instead of submitting."""
