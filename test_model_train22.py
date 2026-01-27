"""
Evaluate the checkpoint from checkpoints/train22 on the test split of the fixed dataset.
Run from IML folder: python IML-PW/test_model_train22.py
Or from IML-PW folder: python test_model_train22.py
"""
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_utils import pad_collate, build_h5_loaders

# Get script directory to construct absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "outputs", "logmels_fixed_split.h5")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "train24", "best_model.pt")

# ---------------------------------------------------------------------
# Model definition (matches the training notebook)
# ---------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, t, f = x.shape
        s = x.mean(dim=(2, 3))          # [B, C]
        w = self.fc(s).view(b, c, 1, 1) # [B, C, 1, 1]
        return x * w


class Backbone(nn.Module):
    def __init__(self, no_mels: int, embed_dim: int, rnn_hidden: int, rnn_layers: int, bidir: bool):
        super().__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEBlock(32, reduction=8),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64, reduction=8),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128, reduction=8),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        self.rnn_hidden = rnn_hidden
        self.rnn = nn.GRU(
            input_size=128 * (no_mels // 8),
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=bidir,
            batch_first=True,
            dropout=0.2,
        )

        out_dim = (2 if bidir else 1) * rnn_hidden
        self.rnn_ln = nn.LayerNorm(out_dim)
        self.att = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x, lengths=None, mc_dropout: bool | None = None):
        if mc_dropout is None:
            mc_dropout = self.training

        h = self.cnn_block(x)
        if mc_dropout:
            h = F.dropout(h, p=0.3, training=True)

        b, c, t, f = h.shape
        h = h.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.rnn(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            tmax = rnn_out.size(1)
            mask = (torch.arange(tmax, device=rnn_out.device).unsqueeze(0).expand(b, tmax)) < lengths.unsqueeze(1)
        else:
            rnn_out, _ = self.rnn(h)
            mask = torch.ones(rnn_out.size(0), rnn_out.size(1), dtype=torch.bool, device=rnn_out.device)

        if mc_dropout:
            rnn_out = F.dropout(rnn_out, p=0.3, training=True)

        rnn_out = self.rnn_ln(rnn_out)
        a = self.att(rnn_out).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1).unsqueeze(-1)

        mean = torch.sum(w * rnn_out, dim=1)
        var = torch.sum(w * (rnn_out - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var + 1e-5)
        stats = torch.cat([mean, std], 1)

        if mc_dropout:
            stats = F.dropout(stats, p=0.3, training=True)

        z = self.proj(stats)
        z = F.normalize(z, p=2, dim=1)
        return z


class AAMSoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.20):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels):
        w = F.normalize(self.weight, dim=1)
        cos_theta = emb @ w.t()
        theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=w.size(0)).float()
        output = cos_theta * (1 - one_hot) + target_logits * one_hot
        return output * self.s


class SpeakerClassifier(nn.Module):
    def __init__(self, backbone, num_speakers, aamsm_scaler, aamsm_margin):
        super().__init__()
        self.backbone = backbone
        self.aamsm = AAMSoftmax(backbone.proj[-1].out_features, num_speakers, aamsm_scaler, aamsm_margin)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, labels=None, lengths=None):
        emb = self.backbone(x, lengths=lengths, mc_dropout=None)
        if labels is not None:
            logits = self.aamsm(emb, labels)
            return logits, emb
        return emb

    def eval(self):
        super().eval()
        if not hasattr(self, "bank") or self.bank is None:
            with torch.no_grad():
                self.bank = F.normalize(self.aamsm.weight, dim=1)
        return self

    @torch.no_grad()
    def embed(self, x, lengths=None):
        z = self.backbone(x, lengths=lengths)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def verify_any(self, x, lengths=None, threshold=None, bank=None, return_index=False):
        if bank is None:
            if not hasattr(self, "bank") or self.bank is None:
                raise RuntimeError("Bank not built; call eval() first")
            bank = self.bank
        bank = F.normalize(bank, dim=1)
        probe = self.embed(x, lengths=lengths)
        bank = bank.to(probe.device)
        sims = probe @ bank.t()
        scores, idx = sims.max(dim=1)
        thr = 0.7 if threshold is None else threshold
        decisions = scores >= thr
        if return_index:
            return decisions, scores, idx
        return decisions, scores

# ---------------------------------------------------------------------
# Utility: read dataset meta
# ---------------------------------------------------------------------
def get_yaml_meta(h5_path):
    import h5py

    with h5py.File(h5_path, "r") as f:
        raw = f["/meta/file_description.yaml"][()].decode("utf-8")
        meta_yaml = yaml.safe_load(raw) or {}
        meta_grp = f["/meta"]
        attrs = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in meta_grp.attrs.items()}
        meta_yaml.update(attrs)
        if "total_speakers" not in meta_yaml:
            if "speaker_mapping.yaml" in meta_grp:
                sm_raw = meta_grp["speaker_mapping.yaml"][()].decode("utf-8")
                sm = yaml.safe_load(sm_raw) or {}
                speakers = sm.get("speakers", {})
                meta_yaml["total_speakers"] = len(speakers)
    return meta_yaml

# Config already set at top of file using __file__
BATCH_SIZE = 256
NUM_WORKERS = 2  # safe on Windows; set 0 if issues

# Main
def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    meta = get_yaml_meta(DATASET_PATH)
    no_mels = meta["preprocessing_config"]["n_mels"]
    no_speakers = meta["total_speakers"]

    print("\n" + "="*70)
    print("EVALUATION: train22 on TEST split")
    print("="*70)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Speakers: {no_speakers}")
    print(f"Mel bands: {no_mels}")

    loaders = build_h5_loaders(
        DATASET_PATH,
        splits=("test",),
        feature_key="logmel",
        label_key="label",
        length_key="length",
        time_dim="F",
        batch_sizes={"test": BATCH_SIZE},
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        shuffle_train=False,
        remap_labels=True,
    )
    test_loader = loaders["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = Backbone(no_mels=no_mels, embed_dim=256, rnn_hidden=256, rnn_layers=2, bidir=True)
    model = SpeakerClassifier(backbone, num_speakers=no_speakers, aamsm_scaler=30.0, aamsm_margin=0.25).to(device)

    state = torch.load(CHECKPOINT_PATH, map_location=device)
    # Remove keys not in current model architecture (score_alpha, score_beta from old training)
    state = {k: v for k, v in state.items() if k in model.state_dict()}
    model.load_state_dict(state)
    model.eval()

    total, correct = 0, 0
    all_preds, all_labels = [], []
    start = time.perf_counter()

    with torch.no_grad():
        for batch in test_loader:
            x, y, lengths = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            logits, _ = model(x, y, lengths=lengths)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.numel()
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    elapsed = time.perf_counter() - start
    acc = correct / total if total else 0.0

    print(f"\nTest samples: {total}")
    print(f"Test accuracy: {acc*100:.2f}%")
    print(f"Time: {elapsed:.2f}s")

    # Optional: confusion matrix
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(torch.cat(all_labels), torch.cat(all_preds))
        print("\nConfusion matrix:")
        print(cm)
    except Exception:
        pass

    print("\n" + "="*70)
    print("Evaluation complete")
    print("="*70)


if __name__ == "main" or __name__ == "__main__":
    main()
