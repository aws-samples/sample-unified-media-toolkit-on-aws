# pyright: reportMissingImports=false, reportMissingModuleSource=false
import time
from embedding_handler import lambda_handler

if __name__ == "__main__":
    context = {}
    event = {
        "operation": "StateSpeechEmbeddingsModel",
        "item_id": 17,
        "item_data": {
            "dyn_chunk_idx": 17,
            "fetch_start_sec": 4905.178,
            "fetch_end_sec": 5319.1,
            "speech_sec": 392.444,
            "num_turns": 10,
            "turn_range": [
                155,
                164
            ],
            "source_vad_chunks": [
                6,
                7
            ],
            "is_remainder": False,
            "is_big_cut": False,
            "zone": "core",
            "bucket": "[proxy_bucket]",
            "audio_key": "[uuid]/audio/0/audio_en.wav",
            "audio_cfg": {
                    "duration_sec": 8966.144,
                    "sample_rate": 16000,
                    "num_channels": 1
            },
            "model_bucket": "[proxy_bucket]",
            "model_prefix": "_model_artefacts",
            "output_prefix": "[uuid]/audio/0/dynamic",
            "vad_cfg": {
                "model_id": "pyannote.community1",
                "fixed_chunk_size_secs": 720,
                "dynamic_speech_chunk_secs": 360,
                "overlap_speech_turns": 3,
                "min_speech_duration_sec": 0.5,
                "min_silence_duration_sec": 0.15
            },
            "transcript_key": None,
            "status": "PENDING"
        }
    }

    t0 = time.monotonic()
    response = lambda_handler(event, context)
    t1 = time.monotonic()

    print("response", response)
    print("============================")
    print(f"TOTAL RUNTIME: {t1 - t0}")
