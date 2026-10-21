import time
import os
import gc
from faster_whisper import WhisperModel

# --- CONFIG ---
MODEL_PATH = "./whisper-v3-turbo-arm64-ct2"
AUDIO_FILE = "audio.wav"


def run_whisper_test():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    print(f"📂 Loading Whisper v3 Turbo (ARM64/int8 optimized)...")

    # CHANGE: Switched compute_type to "int8" for maximum ARM64 compatibility
    model = WhisperModel(
        MODEL_PATH,
        device="cpu",
        compute_type="int8",
        cpu_threads=1,
        num_workers=1
    )

    print(f"🎙️  Transcribing: {AUDIO_FILE}...")
    start_time = time.monotonic()

    # segments, info = model.transcribe(
    #     AUDIO_FILE,
    #     beam_size=1,
    #     word_timestamps=True,
    #     vad_filter=True
    # )
    segments, info = model.transcribe(
        AUDIO_FILE,
        beam_size=2,            # Max speed for Turbo
        word_timestamps=True,   # Required for alignment
        vad_filter=True,        # Skip silences
        vad_parameters=dict(min_silence_duration_ms=500),
        # Stability for long files
        condition_on_previous_text=True,  # Prevents "looping" hallucinations
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        # Language
        language="en",          # Explicitly setting language saves language-detection time
    )
    print(f"✨ Language: {info.language}")
    print("-" * 30)

    for segment in segments:
        # Using strip() to keep the output clean in the logs
        print(
            f"[{segment.start:>7.2f}s -> {segment.end:>7.2f}s] {segment.text.strip()}")

    duration = time.monotonic() - start_time
    print("-" * 30)
    print(f"✅ Success! Total Time: {duration:.2f}s")

    del model
    gc.collect()


if __name__ == "__main__":
    run_whisper_test()
