import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

# ==========================================
# 1. CONFIGURATION
# ==========================================

CHECKPOINT_PATH = "best_model.pt"
N_MELS = 64
EMBED_DIM = 256
NUM_SPEAKERS = 2
IN_GROUP_IDS = [1, 27, 29, 38, 40]
CONFIDENCE_THRESHOLD = 0.60

# Global variables to store the loaded model and device
# This allows predict_speaker to access them without passing arguments
_MODEL = None
_DEVICE = None


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================

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
        self.rnn = nn.GRU(input_size=128 * (no_mels // 8), hidden_size=self.rnn_hidden,
                          num_layers=rnn_layers, bidirectional=bidir, batch_first=True, dropout=0.2)
        out_dim = (2 if bidir else 1) * rnn_hidden
        self.rnn_ln = nn.LayerNorm(out_dim)
        self.att = nn.Sequential(nn.Linear(out_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.proj = nn.Sequential(nn.Linear(out_dim * 2, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                  nn.Linear(256, embed_dim))

    def forward(self, x):
        h = self.cnn_block(x)
        B, C, T, Freq = h.shape
        h = h.permute(0, 2, 1, 3).contiguous().view(B, T, C * Freq)

        rnn_out, _ = self.rnn(h)
        rnn_out = self.rnn_ln(rnn_out)
        a = self.att(rnn_out).squeeze(-1)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        mean = torch.sum(w * rnn_out, dim=1)
        var = torch.sum(w * (rnn_out - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var + 1e-5)
        stats = torch.cat([mean, std], 1)
        z = self.proj(stats)
        return F.normalize(z, p=2, dim=1)


class AAMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.20):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb):
        W = F.normalize(self.weight, dim=1)
        return emb @ W.T * self.s


class SpeakerClassifier(nn.Module):
    def __init__(self, backbone, num_speakers):
        super().__init__()
        self.backbone = backbone
        self.aamsm = AAMSoftmax(256, num_speakers)

    def forward(self, x):
        emb = self.backbone(x)
        return self.aamsm(emb)


# ==========================================
# 3. HELPER FUNCTIONS (Setup & Preprocessing)
# ==========================================

def load_model():
    """
    Initializes the model, loads weights from checkpoint, and sets the global device.
    """
    global _MODEL, _DEVICE

    # 1. Determine Device
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device selected: {_DEVICE}")

    # 2. Build Architecture
    print(f"⏳ Building Model ({NUM_SPEAKERS} Classes)...")
    backbone = Backbone(no_mels=N_MELS, embed_dim=EMBED_DIM, rnn_hidden=256, rnn_layers=2, bidir=True)
    model = SpeakerClassifier(backbone, num_speakers=NUM_SPEAKERS)

    # 3. Load Weights
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=_DEVICE)
        model.load_state_dict(state_dict, strict=False)
        model.to(_DEVICE)
        model.eval()
        print("✅ Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    _MODEL = model
    return _MODEL, _DEVICE


def preprocess_file(file_path):
    """Reads audio, trims silence, normalizes volume, and chunks it."""
    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)

        if len(y) < 1000: return None

        y = y / (np.max(np.abs(y)) + 1e-9)

        chunk_len = int(1.0 * 16000)
        stride = int(2.0 * 16000)

        chunks = []
        if len(y) < chunk_len:
            y = np.pad(y, (0, chunk_len - len(y)))
            chunks.append(y)
        else:
            for i in range(0, len(y) - chunk_len + 1, stride):
                chunks.append(y[i: i + chunk_len])

        mels = []
        for c in chunks:
            mel = librosa.feature.melspectrogram(
                y=c, sr=16000, n_fft=2048, hop_length=512, n_mels=N_MELS
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            mels.append(log_mel.T)

        if not mels: return None
        return torch.tensor(np.array(mels), dtype=torch.float32).unsqueeze(1)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ==========================================
# 4. EXPORTED FUNCTION
# ==========================================

def predict_speaker(file_path):
    """
    Takes an audio file path.
    Automatically loads the model if it hasn't been loaded yet.

    Returns:
        prediction (str): "✅ MEMBER", "❌ OUTSIDER", or "❓ LOW CONF (1)"
        pred_id (int): The predicted speaker ID
        avg_conf (float): The confidence score (0.0 to 1.0)
    """
    global _MODEL, _DEVICE

    # Auto-initialization logic
    if _MODEL is None:
        load_model()

    # 1. Preprocess
    batch = preprocess_file(file_path)

    if batch is None:
        return "ERROR", -1, 0.0

    # 2. Inference
    batch = batch.to(_DEVICE)

    with torch.no_grad():
        logits = _MODEL(batch)
        probs = torch.softmax(logits, dim=1)

        # Aggregate chunks
        chunk_conf, chunk_ids = torch.max(probs, dim=1)
        votes = chunk_ids.cpu().tolist()

        if not votes:
            return "ERROR", -1, 0.0

        pred_id = max(set(votes), key=votes.count)
        avg_conf = chunk_conf.mean().item()

    # 3. Decision Logic
    if pred_id in IN_GROUP_IDS:
        if avg_conf >= CONFIDENCE_THRESHOLD:
            prediction = "✅ MEMBER"
        else:
            prediction = "❓ LOW CONF (1)"
    else:
        prediction = "❌ OUTSIDER"

    return prediction, pred_id, avg_conf
