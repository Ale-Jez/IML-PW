import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import argparse
import os
import sys

# ============================================================================
# 1. MODEL ARCHITECTURE (Exact copy from voice_recognition_rnn_cnn.ipynb)
# ============================================================================

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, T, F = x.shape
        s = x.mean(dim=(2, 3))
        w = self.fc(s).view(B, C, 1, 1)
        return x * w

class Backbone(nn.Module):
    def __init__(self, no_mels, embed_dim, rnn_hidden, rnn_layers, bidir):
        super().__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            SEBlock(32, reduction=8), nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            SEBlock(64, reduction=8), nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            SEBlock(128, reduction=8), nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.rnn_hidden = rnn_hidden
        self.rnn = nn.GRU(
            input_size=128 * (no_mels // 8),
            hidden_size=self.rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=bidir,
            batch_first=True,
            dropout=0.2
        )
        out_dim = (2 if bidir else 1) * rnn_hidden
        self.rnn_ln = nn.LayerNorm(out_dim)
        self.att = nn.Sequential(
            nn.Linear(out_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, 256),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x, lengths=None):
        h = self.cnn_block(x)
        B, C, T, Fp = h.shape
        h = h.permute(0, 2, 1, 3).contiguous().view(B, T, C * Fp)

        if lengths is not None:
            # We must clamp lengths to avoid errors if calc is slightly off
            lengths = lengths.clamp(max=T)
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            Tmax = rnn_out.size(1)
            mask = torch.arange(Tmax, device=rnn_out.device).unsqueeze(0).expand(B, Tmax) < lengths.unsqueeze(1)
        else:
            rnn_out, _ = self.rnn(h)
            mask = torch.ones(rnn_out.size(0), rnn_out.size(1), dtype=torch.bool, device=rnn_out.device)

        rnn_out = self.rnn_ln(rnn_out)
        a = self.att(rnn_out).squeeze(-1).masked_fill(~mask, float('-inf'))
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        mean = torch.sum(w * rnn_out, dim=1)
        var = torch.sum(w * (rnn_out - mean.unsqueeze(1))**2, dim=1)
        std = torch.sqrt(var + 1e-5)
        stats = torch.cat([mean, std], 1)
        z = self.proj(stats)
        return F.normalize(z, p=2, dim=1)

class AAMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.20):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

class SpeakerClassifier(nn.Module):
    def __init__(self, backbone, num_speakers, aamsm_scaler=30.0, aamsm_margin=0.2):
        super().__init__()
        self.backbone = backbone
        self.aamsm = AAMSoftmax(backbone.proj[-1].out_features, num_speakers, aamsm_scaler, aamsm_margin)
        self.bank = None

    def eval(self):
        super().eval()
        if self.bank is None:
            with torch.no_grad():
                self.bank = F.normalize(self.aamsm.weight, dim=1)
        return self

# ============================================================================
# 2. PREPROCESSING (Matching prepare_h5.ipynb)
# ============================================================================

class InferencePreprocessor:
    """Matches the exact steps from prepare_h5.ipynb"""
    def __init__(self, sr=16000, n_mels=64, n_fft=2048, hop_length=512, chunk_duration=1.0):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_duration = chunk_duration # Training used 1.0s chunks

    def process(self, audio_path):
        # 1. Load Audio
        try:
            y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None

        # 2. Trim Silence (CRITICAL: Training data had no silence)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # 3. Normalize Volume (CRITICAL: Training data was normalized)
        max_val = np.max(np.abs(y)) + 1e-9
        y = y / max_val

        # 4. Chunking (Split long audio into 1s segments)
        # This matches the training data format exactly.
        chunk_len = int(self.chunk_duration * self.sr)
        
        # If audio is shorter than 1 chunk, pad it
        if len(y) < chunk_len:
            y = np.pad(y, (0, chunk_len - len(y)))
            chunks = [y]
        else:
            # Create overlapping chunks (optional overlap, here using 0.5s overlap for better coverage)
            # Training likely used non-overlapping, but inference benefits from overlap
            stride = chunk_len // 2
            chunks = [y[i:i+chunk_len] for i in range(0, len(y) - chunk_len + 1, stride)]
            
            # If no chunks created (edge case), take whole
            if not chunks: 
                chunks = [y]

        # 5. Convert chunks to Log-Mel Spectrograms
        batch_logmels = []
        batch_lengths = []
        
        for chunk in chunks:
            if len(chunk) < chunk_len: # pad last chunk if needed
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
                
            mel = librosa.feature.melspectrogram(
                y=chunk, sr=self.sr, n_mels=self.n_mels, 
                n_fft=self.n_fft, hop_length=self.hop_length
            )
            logmel = librosa.power_to_db(mel, ref=np.max)
            
            # Transpose to (Time, Freq) as expected by Backbone input pipeline
            logmel = logmel.T # Shape: [T, F]
            
            batch_logmels.append(logmel)
            batch_lengths.append(logmel.shape[0])

        # Stack into batch: [Batch, 1, Time, Freq]
        batch_tensor = torch.tensor(np.array(batch_logmels), dtype=torch.float32).unsqueeze(1)
        lengths_tensor = torch.tensor(batch_lengths, dtype=torch.long)
        
        return batch_tensor, lengths_tensor

# ============================================================================
# 3. INFERENCE LOGIC
# ============================================================================

def load_trained_model(checkpoint_path, num_speakers, device):
    print(f"⏳ Loading model from {checkpoint_path}...")
    # Initialize Architecture
    backbone = Backbone(no_mels=64, embed_dim=256, rnn_hidden=256, rnn_layers=2, bidir=True)
    model = SpeakerClassifier(backbone, num_speakers=num_speakers)
    
    # Load Weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/train1/best_model.pt")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "speaker_id"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Config based on Mode
    # Binary: 0=Outsider, 1=Member (2 classes)
    # SpkID: 58 speakers (from your logs)
    num_speakers = 2 if args.mode == "binary" else 58 
    
    # 2. Load Model
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
        
    model = load_trained_model(args.checkpoint, num_speakers, device)
    
    # 3. Preprocess
    preprocessor = InferencePreprocessor(chunk_duration=1.0) # Matches training chunk size
    data = preprocessor.process(args.audio)
    
    if data is None:
        sys.exit(1)
        
    batch_x, batch_len = data
    batch_x = batch_x.to(device)
    batch_len = batch_len.to(device)
    
    # 4. Inference (Batched)
    print(f"ℹ️  Processing {len(batch_x)} chunks...")
    with torch.no_grad():
        # Get embeddings for all chunks
        embeddings = model.backbone(batch_x, lengths=batch_len) # [Batch, 256]
        
        # Average the embeddings (Mean Pooling over time chunks)
        # This creates a single robust vector for the whole file
        avg_embedding = F.normalize(embeddings.mean(dim=0, keepdim=True), p=2, dim=1)
        
        # Compare to Bank
        bank = model.bank.to(device)
        sims = avg_embedding @ bank.T # [1, NumClasses]
        
        # Get score
        score, pred_id = sims.max(dim=1)
        final_score = score.item()
        final_id = pred_id.item()

    # 5. Report
    print("-" * 40)
    print(f"FILE:       {os.path.basename(args.audio)}")
    print(f"MODE:       {args.mode.upper()}")
    
    threshold = 0.7
    
    if args.mode == "binary":
        # Class 1 = Member, Class 0 = Outsider
        label = "MEMBER" if final_id == 1 else "OUTSIDER"
        authorized = (final_id == 1 and final_score > threshold)
        
        # Special logic: If predicted Outsider, score is similarity to Outsider cluster
        # If we want 'Confidence of being Member', we can check sims[0, 1]
        member_confidence = sims[0, 1].item()
        print(f"PREDICTION: {label}")
        print(f"CONFIDENCE: {final_score:.1%} (Match to class {final_id})")
        print(f"MBR SCORE:  {member_confidence:.1%} (Similarity to Member class)")
        
    else:
        label = f"Speaker {final_id}"
        authorized = (final_score > threshold)
        print(f"PREDICTION: {label}")
        print(f"CONFIDENCE: {final_score:.1%}")

    status = "✅ ACCESS GRANTED" if authorized else "❌ ACCESS DENIED"
    print(f"RESULT:     {status}")
    print("-" * 40)