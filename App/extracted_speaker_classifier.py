"""
Extracted Speaker Verification Model Classes
Extracted from: voice_recognition_rnn_cnn.ipynb

Contains:
- SEBlock: Squeeze-and-Excitation block for channel attention
- Backbone: CNN-RNN feature extractor for speaker embeddings
- AAMSoftmax: Additive Angular Margin Softmax loss head
- SpeakerClassifier: Complete model unifying Backbone with AAMSoftmax

Ready to use in any Python environment with PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Learns to scale channel responses based on global context.
    """
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
        # x: [B, C, T, F]
        B, C, T, F = x.shape
        # Global average pooling over time + frequency
        s = x.mean(dim=(2, 3))              # [B, C]
        w = self.fc(s)                      # [B, C]
        w = w.view(B, C, 1, 1)              # [B, C, 1, 1]
        return x * w


class Backbone(nn.Module):
    """
    Backbone of model, unifies entry CNN block with RNN block.
    
    Processes log-mel spectrograms [B, 1, T, F] through:
    1. CNN feature extraction with SE attention blocks
    2. GRU-based temporal modeling
    3. Attentive Statistics Pooling (ASP)
    4. Speaker embedding projection
    
    Args:
        no_mels: Number of mel-frequency bins (frequency dimension F)
        embed_dim: Output embedding dimension
        rnn_hidden: GRU hidden state dimension
        rnn_layers: Number of GRU layers
        bidir: Whether GRU is bidirectional
    """
    def __init__(self, no_mels, embed_dim, rnn_hidden, rnn_layers, bidir):
        super().__init__()

        # CNN BLOCK: Extract spectral features
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

        # RNN BLOCK: Temporal modeling
        self.rnn_hidden = rnn_hidden  # for easier reconfiguration in the future
        self.rnn = nn.GRU(  # Gated Recurrent Unit (RNN)
            input_size=128 * (no_mels // 8),  # 8 = 2^3 because 3 pooling by 2
            hidden_size=self.rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=bidir,  # if true the GRU learns in both directions
            batch_first=True,
            dropout=0.2
        )

        # EMBEDDING HEAD
        out_dim = (2 if bidir else 1) * rnn_hidden

        self.rnn_ln = nn.LayerNorm(out_dim)

        # Simple attention for ASP (Attentive Statistics Pooling)
        # essentially ignores boring, uninteresting moments
        self.att = nn.Sequential(
            nn.Linear((2 if bidir else 1) * rnn_hidden, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Linear(256, embed_dim)   # embedding to given embedding dimension
        )

    def forward(self, x, lengths: torch.Tensor | None = None, mc_dropout: bool | None = None):
        if mc_dropout is None:
            mc_dropout = self.training
        
        # PIPELINE
        h = self.cnn_block(x)  # process through CNN
        if mc_dropout:
            h = F.dropout(h, p=0.3, training=True)  # monte carlo dropout applied
        
        # shape after cnn_block [B, C, T, F (pooled)]
        B, C, T, Fp = h.shape
        h = h.permute(0, 2, 1, 3).contiguous().view(B, T, C * Fp)  # reshape for time sequence analysis

        # RNN processing with optional sequence packing
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                h,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.rnn(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            Tmax = rnn_out.size(1)
            mask = (torch.arange(Tmax, device=rnn_out.device).unsqueeze(0).expand(B, Tmax)) < lengths.unsqueeze(1)
        else:
            rnn_out, _ = self.rnn(h)  # process reshaped cnn output through rnn, outputs [B, T, H×2]
            mask = torch.ones(rnn_out.size(0), rnn_out.size(1),
                            dtype=torch.bool, device=rnn_out.device)

        if mc_dropout:
            rnn_out = F.dropout(rnn_out, p=0.3, training=True)  # monte carlo dropout applied

        rnn_out = self.rnn_ln(rnn_out)

        # STATISTICS: Attentive Statistics Pooling
        a = self.att(rnn_out).squeeze(-1)  # attention weights over time, dim: BxTx1
        a = a.masked_fill(~mask, float('-inf'))
        w = torch.softmax(a, dim=1).unsqueeze(-1)  # dim: BxTx1

        mean = torch.sum(w * rnn_out, dim=1)  # dim: BxH
        var = torch.sum(w * (rnn_out - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var + 1e-5)
        stats = torch.cat([mean, std], 1)

        if mc_dropout:
            stats = F.dropout(stats, p=0.3, training=True)

        z = self.proj(stats)                  # B×emb_dim
        z = nn.functional.normalize(z, p=2, dim=1)

        return z


class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax head.
    
    Enhances class identification using additive margins.
    Used only for training; discarded at inference time.
    
    Achieves:
    - Tighter intra-class clusters
    - Larger inter-class gaps
    
    Args:
        in_features: Embedding dimension
        out_features: Number of speaker classes
        s: Temperature scaling factor (default: 30.0)
        m: Additive angular margin (default: 0.20)
    
    Reference: https://medium.com/@zhaomin.chen/additive-margin-softmax-loss-3c78e37b08ed
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.20):
        super().__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels):
        # Normalize weight vectors (embedding is already normalized)
        W = F.normalize(self.weight, dim=1)

        # Get cosine similarities using matrix product
        cos_theta = emb @ W.T

        # Increase the margins
        theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)

        # One-hot encoding, substituting only for true speaker
        one_hot = F.one_hot(labels, num_classes=W.size(0)).float()
        output = cos_theta * (1 - one_hot) + target_logits * one_hot

        return output * self.s


class SpeakerClassifier(nn.Module):
    """
    Complete speaker verification/classification model.
    
    Unifies Backbone with AAMSoftmax head for training and inference.
    
    At training:
    - Outputs speaker classification logits using AAMSoftmax loss
    - Uses speaker embeddings for loss computation
    
    At inference:
    - Returns speaker embeddings
    - Builds speaker prototype bank from AAMSoftmax weights
    - Compares input embeddings against prototype bank
    
    Args:
        backbone: Backbone instance for feature extraction
        num_speakers: Number of speaker classes
        aamsm_scaler: AAMSoftmax temperature scaling (s parameter)
        aamsm_margin: AAMSoftmax additive margin (m parameter)
    """
    def __init__(self, backbone, num_speakers, aamsm_scaler, aamsm_margin):
        super().__init__()
        self.backbone = backbone
        self.aamsm = AAMSoftmax(
            backbone.proj[-1].out_features,
            num_speakers,
            aamsm_scaler,
            aamsm_margin
        )
        self._inference_prepared = False
        self.score_alpha = nn.Parameter(torch.tensor(1.0))  # scale
        self.score_beta = nn.Parameter(torch.tensor(0.0))   # bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bank = None
        self.inference_threshold = 0.7

    def forward(self, x, labels=None, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input spectrograms [B, 1, T, F]
            labels: Speaker labels [B] (required for training, optional for inference)
            lengths: Valid time lengths [B] (optional)
        
        Returns:
            If labels provided: (logits, embeddings)
            If labels not provided: embeddings
        """
        emb = self.backbone(x, lengths=lengths, mc_dropout=None)
        if labels is not None:
            logits = self.aamsm(emb, labels)
            return logits, emb
        else:
            return emb

    def eval(self):
        """Set model to evaluation mode and prepare speaker bank."""
        super().eval()   # keeps original behaviour

        # Build bank for inference if it's not already built
        if not self._inference_prepared:
            with torch.no_grad():
                self.bank = F.normalize(self.aamsm.weight, dim=1)
            self._inference_prepared = True

        return self

    @torch.no_grad()
    def embed(self, x, lengths=None):
        """
        Generate speaker embedding.
        
        Args:
            x: Input spectrogram [B, 1, T, F]
            lengths: Valid time lengths [B] (optional)
        
        Returns:
            Normalized embedding [B, D]
        """
        z = self.backbone(x, lengths=lengths)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def mc_embed(self, x, lengths=None, n_samples: int = 10):
        """
        Monte Carlo dropout embeddings for uncertainty estimation.
        
        Args:
            x: Input spectrogram [B, 1, T, F]
            lengths: Valid time lengths [B] (optional)
            n_samples: Number of MC samples
        
        Returns:
            Stack of embeddings [n_samples, B, D]
        """
        embs = []
        for _ in range(n_samples):
            z = self.backbone(x, lengths=lengths, mc_dropout=True)
            z = F.normalize(z, dim=1)
            embs.append(z)
        return torch.stack(embs, dim=0)

    @torch.no_grad()
    def build_bank_from_aam(self):
        """Build speaker prototype bank from AAMSoftmax weights."""
        return F.normalize(self.aamsm.weight, dim=1)

    def set_default_inference_threshold(self, threshold):
        """Set default similarity threshold for speaker verification."""
        self.inference_threshold = threshold

    @torch.no_grad()
    def verify_any(self, x, lengths=None, *, threshold=None, bank=None, return_index=False):
        """
        Compare input embeddings against a speaker bank.
        
        Args:
            x: Input spectrograms [B, 1, T, F]
            lengths: Valid time lengths [B] (optional)
            threshold: Similarity threshold (uses default if None)
            bank: Speaker prototype bank (uses self.bank if None)
            return_index: Whether to return speaker indices
        
        Returns:
            decisions: [B] bool - whether score >= threshold
            scores: [B] float - max cosine similarity
            (optional) indices: [B] long - argmax speaker index
        """
        if bank is None:
            if self.bank is None:
                raise RuntimeError(
                    'The bank has not been built before inference. '
                    'Make sure the model has been set to eval mode!'
                )
            else:
                bank = self.bank
        bank = F.normalize(bank, dim=1)

        probe = self.embed(x, lengths=lengths)      # [B, D]
        bank = bank.to(probe.device)                # Ensure bank is on same device as probe
        sims = probe @ bank.T                       # [B, S]
        scores, idx = sims.max(dim=1)               # max over speakers

        scores = self.score_alpha * scores + self.score_beta

        thr = self.inference_threshold if threshold is None else threshold
        decisions = scores >= thr

        if return_index:
            return decisions, scores, idx
        return decisions, scores

    @torch.no_grad()
    def mc_verify_any(self, x, lengths=None, n_samples: int = 10,
                      threshold=None, bank=None, return_index=False):
        """
        Monte Carlo Dropout version of verify_any with uncertainty estimation.
        
        Args:
            x: Input spectrograms [B, 1, T, F]
            lengths: Valid time lengths [B] (optional)
            n_samples: Number of MC samples for uncertainty
            threshold: Similarity threshold (uses default if None)
            bank: Speaker prototype bank (uses self.bank if None)
            return_index: Whether to return speaker indices
        
        Returns:
            decisions: [B] bool - based on mean score
            mean_scores: [B] float
            var_scores: [B] float - uncertainty estimate
            (optional) pred_idx: [B] long - predicted speaker index from mean scores
        """
        self.eval()
        if bank is None:
            if self.bank is None:
                raise RuntimeError(
                    'The bank has not been built before inference. '
                    'Make sure the model has been set to eval mode!'
                )
            else:
                bank = self.bank
        bank = F.normalize(bank, dim=1)

        embs = self.mc_embed(x, lengths=lengths, n_samples=n_samples)  # [K, B, D]
        bank = bank.to(embs.device)                   # Ensure bank is on same device as embs
        sims = torch.einsum("kbd,sd->kbs", embs, bank)  # [K, B, S]
        mean_sims = sims.mean(dim=0)  # [B, S]
        var_sims = sims.var(dim=0)    # [B, S]
        mean_scores, pred_idx = mean_sims.max(dim=1)  # [B]

        mean_scores = self.score_alpha * mean_scores + self.score_beta
        score_var = var_sims.gather(1, pred_idx.view(-1, 1)).squeeze(1)  # [B]

        thr = self.inference_threshold if threshold is None else threshold
        decisions = mean_scores >= thr

        if return_index:
            return decisions, mean_scores, score_var, pred_idx
        return decisions, mean_scores, score_var

    @torch.no_grad()
    def infer(self, x, lengths=None, threshold=None):
        """
        High-level inference method.
        
        Args:
            x: Input spectrogram [B, 1, T, F] (log-mel spectrograms)
            lengths: Valid time lengths [B] (optional)
            threshold: Similarity threshold (optional)
        
        Returns:
            pred_ids: [B] long - predicted speaker index (0..num_speakers-1)
            scores: [B] float - cosine similarity to predicted prototype
            decisions: [B] bool - whether score >= threshold
        """
        decisions, scores, pred_ids = self.verify_any(
            x, lengths=lengths, threshold=threshold, return_index=True
        )
        return pred_ids, scores, decisions

    @torch.no_grad()
    def test(self, test_loader):
        """
        Test the model on a test DataLoader.
        
        Args:
            test_loader: Either a DataLoader or a dict with 'test' key containing a DataLoader
        
        Returns:
            List of (predicted_speaker_id, similarity_score, true_label) tuples
        """
        self.eval()
        device = self.device if self.device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(device)
        results = []

        # Handle both dict and direct DataLoader
        if isinstance(test_loader, dict):
            test_loader = test_loader['test']

        with torch.no_grad():
            for batch in test_loader:
                X, y, lengths = batch                   # unpack batch

                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)

                for i in range(X.size(0)):              # per-item verify
                    x_single = X[i:i + 1]                 # keep batch dim [1,1,T,F]
                    l_single = lengths[i:i + 1]           # keep batch dim [1]
                    pred_ids, scores, decisions = self.infer(
                        x=x_single,
                        lengths=l_single,
                        threshold=0.7
                    )
                    results.append((
                        int(pred_ids.item()),         # predicted speaker ID
                        float(scores.item()),         # similarity score
                        int(y[i].item())              # true label for this item
                    ))
        
        return results


# Example usage:
if __name__ == "__main__":
    # Create model
    backbone = Backbone(
        no_mels=128,
        embed_dim=256,
        rnn_hidden=256,
        rnn_layers=2,
        bidir=True
    )
    
    model = SpeakerClassifier(
        backbone=backbone,
        num_speakers=10,  # Example: 10 speakers
        aamsm_scaler=30.0,
        aamsm_margin=0.20
    )
    
    # Prepare for inference
    model.eval()
    
    # Create dummy input
    x = torch.randn(2, 1, 100, 128)  # [batch_size, channels, time_steps, mel_bins]
    
    # Inference
    pred_ids, scores, decisions = model.infer(x)
    print(f"Predicted speaker IDs: {pred_ids}")
    print(f"Similarity scores: {scores}")
    print(f"Decisions: {decisions}")
