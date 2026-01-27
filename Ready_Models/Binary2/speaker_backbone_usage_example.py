import numpy as np
import onnxruntime as ort
import librosa
import glob
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_FILE = "speaker_backbone.onnx"
BANK_FILE = "speaker_backbone_speaker_bank.npy"
# TEST_AUDIO_FOLDER = "../../Recordings_1/Aleksander"
TEST_AUDIO_FOLDER = "../"

# Threshold: Since we fix the math, 0.5 - 0.7 will work again
THRESHOLD = 0.6 

# ============================================================================
# UTILITIES
# ============================================================================

def normalize(v):
    """Normalize a matrix of vectors to unit length (L2 norm)."""
    # axis=1 ensures we normalize each row (each speaker) independently
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norm + 1e-9)

def preprocess_audio(audio_path, sr=16000, chunk_duration=3.0, stride=2.0):
    """
    Reads audio, removes silence, and splits into 3.0s chunks.
    Matches the logic of your working PyTorch script.
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except:
        return None, None

    # 1. Trim Silence (Critical for accurate embeddings)
    y, _ = librosa.effects.trim(y, top_db=20)

    # 2. Volume Normalization
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    # 3. Chunking (Split into 3s segments)
    chunk_len = int(chunk_duration * sr)
    step = int(stride * sr)
    
    chunks = []
    if len(y) < chunk_len:
        # Pad short files
        y = np.pad(y, (0, chunk_len - len(y)))
        chunks.append(y)
    else:
        # Create overlapping chunks
        for i in range(0, len(y) - chunk_len + 1, step):
            chunks.append(y[i : i + chunk_len])

    if not chunks: chunks = [y] # Fallback

    # 4. Mel Spectrograms
    specs = []
    for c in chunks:
        if len(c) < chunk_len: 
            c = np.pad(c, (0, chunk_len - len(c)))
            
        mel = librosa.feature.melspectrogram(y=c, sr=sr, n_mels=64, n_fft=2048, hop_length=512)
        logmel = librosa.power_to_db(mel, ref=np.max)
        specs.append(logmel.T) # [Time, Freq]

    # Batch: [Batch, 1, Time, Freq]
    batch_x = np.array(specs, dtype=np.float32)[:, np.newaxis, :, :]
    
    # Lengths: [Batch]
    batch_l = np.array([batch_x.shape[2]] * len(batch_x), dtype=np.int64)
    
    return batch_x, batch_l

# ============================================================================
# MAIN INFERENCE
# ============================================================================

# 1. Load ONNX
session = ort.InferenceSession(MODEL_FILE)

# 2. Load & Normalize Bank (CRITICAL FIX)
# We must normalize the reference vectors to get valid Cosine Similarity
speaker_bank = np.load(BANK_FILE)
speaker_bank = normalize(speaker_bank) 

def verify_speaker(audio_path):
    # Preprocess (Get batch of 3s chunks)
    spectrograms, lengths = preprocess_audio(audio_path)
    if spectrograms is None: return {"speaker_id": -1, "confidence": 0.0, "authorized": False}

    # Run Inference
    outputs = session.run(None, {
        "input_spectrogram": spectrograms,
        "input_lengths": lengths
    })
    
    # embeddings shape: [Batch_Size, 256]
    embeddings = outputs[0]

    # Normalize input embeddings (The model output should be normalized, but good to be safe)
    embeddings = normalize(embeddings)

    # Calculate Similarity for each chunk
    # [Batch, 256] @ [256, NumSpeakers] = [Batch, NumSpeakers]
    chunk_scores = embeddings @ speaker_bank.T
    
    # Average the scores across all chunks
    avg_scores = np.mean(chunk_scores, axis=0)
    
    # Find best speaker
    best_idx = np.argmax(avg_scores)
    best_score = avg_scores[best_idx]

    # Decision
    # Class 1 is "Member", Class 0 is "Outsider" (Binary Mode)
    # Or match specific ID for Speaker ID mode





    # is_authorized = (best_idx == 1 and best_score >= THRESHOLD)
    is_authorized = (best_idx == 1 )




 

    return {
        "speaker_id": int(best_idx),
        "confidence": float(best_score),
        "authorized": is_authorized
    }

# Run Test
audio_files = glob.glob(os.path.join(TEST_AUDIO_FOLDER, "*.wav"))
print("-" * 85)
print(f"{'FILE':<35} | {'ID':<3} | {'NAME':<12} | {'CONF':<8} | {'ACCESS'}")
print("-" * 85)

for file_path in audio_files:
    result = verify_speaker(file_path)
    
    fname = os.path.basename(file_path)
    status = "✅ GRANTED" if result['authorized'] else "❌ DENIED"

    
    # Formatting confidence to %
    print(f"{fname[:33]:<35} | {result['speaker_id']:<3} | {result['confidence']:.1%}   | {status}")