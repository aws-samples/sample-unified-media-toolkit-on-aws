import numpy as np
import soundfile as sf
import onnxruntime as ort
import json
import os

# --- CONFIG ---
MODEL_PATH = "./assets/models/align/en/quantized/model_quantized.onnx"
VOCAB_PATH = "./assets/models/align/en/quantized/vocab.json"
AUDIO_INPUT = "audio.wav"
CHUNK_SECONDS = 30


def normalize_audio(audio):
    """Standard Wav2Vec2 Normalization (Zero-mean, unit-variance)"""
    return (audio - np.mean(audio)) / (np.sqrt(np.var(audio)) + 1e-7)


def ctc_decode(indices, inv_vocab):
    """
    Standard CTC Decoding:
    1. Collapse consecutive repeated characters.
    2. Remove the padding token (index 0).
    3. Map '|' to space.
    """
    tokens = []
    prev_idx = None

    for idx in indices:
        # 1. Collapse repeats (e.g., 'hhheee' -> 'he')
        if idx == prev_idx:
            continue

        # 2. Skip special tokens: 0 (pad), 1 (bos), 2 (eos), 3 (unk)
        if idx <= 3:
            prev_idx = idx
            continue

        char = inv_vocab.get(idx, "")

        # 3. Handle word separator '|'
        if char == "|":
            tokens.append(" ")
        else:
            tokens.append(char)

        prev_idx = idx

    return "".join(tokens).strip()


def run_pure_onnx_chunked():
    # 1. Load Vocab
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    inv_vocab = {int(v): k for k, v in vocab.items()}

    # 2. Initialize ONNX
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    session = ort.InferenceSession(MODEL_PATH, sess_options, providers=[
                                   'CPUExecutionProvider'])

    # 3. Load Audio
    print(f"📂 Loading {AUDIO_INPUT}...")
    speech, sr = sf.read(AUDIO_INPUT)
    if len(speech.shape) > 1:
        speech = np.mean(speech, axis=1)

    # Resample to 16kHz if necessary
    if sr != 16000:
        print(f"🔄 Resampling from {sr}Hz to 16000Hz...")
        duration = len(speech) / sr
        speech = np.interp(np.linspace(0, duration, int(duration * 16000)),
                           np.linspace(0, duration, len(speech)), speech)

    # 4. Process in Chunks
    total_samples = len(speech)
    chunk_samples = CHUNK_SECONDS * 16000
    full_transcript = []

    print(f"🚀 Processing {total_samples/16000:.2f}s audio...")

    for i in range(0, total_samples, chunk_samples):
        chunk = speech[i: i + chunk_samples]
        if len(chunk) < 1600:
            continue  # Skip tiny fragments

        # Normalize
        input_values = normalize_audio(chunk).astype(np.float32)
        input_values = input_values[np.newaxis, :]

        # Inference
        # emissions shape: (1, frames, 33)
        emissions = session.run(
            None, {session.get_inputs()[0].name: input_values})[0]

        # Get the most likely character index for each frame
        prediction_ids = np.argmax(emissions[0], axis=-1)

        # Decode the chunk
        chunk_text = ctc_decode(prediction_ids, inv_vocab)

        if chunk_text:
            timestamp = i / 16000
            print(f"[{timestamp:0>7.1f}s] {chunk_text[:80]}...")
            full_transcript.append(chunk_text)

    print("\n" + "="*30)
    print("✅ FINAL ALIGNMENT PREVIEW:")
    print(" ".join(full_transcript)[:500])
    print("="*30)


if __name__ == "__main__":
    run_pure_onnx_chunked()
