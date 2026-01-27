
# ============================================================================
# PYTHON USAGE EXAMPLE
# ============================================================================

import numpy as np
import onnxruntime as ort
import librosa

# 1. Load the ONNX model and speaker bank
session = ort.InferenceSession("speaker_backbone.onnx")
speaker_bank = np.load("./speaker_backbone_speaker_bank.npy")

# 2. Preprocess audio to log-mel spectrogram
def preprocess_audio(audio_path, sr=16000, n_mels=64, n_fft=2048, hop_length=512):
    """Convert audio file to log-mel spectrogram."""
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    logmel = librosa.power_to_db(mel, ref=np.max)

    # Shape: [1, 1, T, F]
    logmel = logmel.T[np.newaxis, np.newaxis, :, :]
    return logmel.astype(np.float32)

# 3. Run inference
def verify_speaker(audio_path, threshold=0.7):
    """Verify if speaker is authorized."""
    # Preprocess
    spectrogram = preprocess_audio(audio_path)
    lengths = np.array([spectrogram.shape[2]], dtype=np.int64)

    # Get embedding
    outputs = session.run(None, {
        "input_spectrogram": spectrogram,
        "input_lengths": lengths
    })
    embedding = outputs[0]

    # Compare to speaker bank (cosine similarity)
    similarities = embedding @ speaker_bank.T
    best_idx = np.argmax(similarities)
    best_score = similarities[0, best_idx]

    # Decision
    is_authorized = best_score >= threshold

    return {
        "speaker_id": int(best_idx),
        "confidence": float(best_score),
        "authorized": bool(is_authorized)
    }

# Example usage
result = verify_speaker("../test-recording.wav")
print(f"Speaker ID: {result['speaker_id']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Access: {'GRANTED' if result['authorized'] else 'DENIED'}")
