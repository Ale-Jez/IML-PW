"""
Speaker Classification Model Classes
Extracted from voice_recognition_rnn_cnn.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import Tuple, Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Backbone(nn.Module):
    """Speaker embedding extractor backbone"""
    def __init__(self, embedding_dim=256, **kwargs):
        super(Backbone, self).__init__()
        self.embedding_dim = embedding_dim
        
        # CNN block
        self.cnn_block = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEBlock(32, reduction=8),
            nn.MaxPool2d((2, 4)),
            
            # Layer 2
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SEBlock(64, reduction=8),
            nn.MaxPool2d((2, 4)),
            
            # Layer 3
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128, reduction=8),
            nn.MaxPool2d((2, 4)),
        )
        
        # Calculate CNN output size
        self.cnn_out_dim = 128 * 8  # freq_bins / (4*4*4) * channels
        
        # RNN block
        self.rnn = nn.GRU(
            input_size=self.cnn_out_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        
        self.rnn_ln = nn.LayerNorm(512)
        
        # Attention pooling
        self.att = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        
        # Projection
        self.proj = nn.Sequential(
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: input logmel (batch, 1, freq, time)
            lengths: sequence lengths for packing
        Returns:
            embeddings: (batch, embedding_dim)
        """
        # CNN
        x = self.cnn_block(x)  # (batch, 128, freq_compressed, time_compressed)
        
        # Reshape for RNN
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, time, 128, freq)
        x = x.view(b, t, -1)  # (batch, time, 128*freq)
        
        # RNN
        if lengths is not None and lengths.min().item() > 0:
            # Clamp lengths to actual sequence length
            lengths = torch.clamp(lengths, min=1, max=t)
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            x, _ = self.rnn(x)
            x, _ = pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.rnn(x)
        
        x = self.rnn_ln(x)
        
        # Attention pooling
        attn = self.att(x)  # (batch, time, 1)
        attn = F.softmax(attn, dim=1)
        x_mean = (x * attn).sum(dim=1)  # (batch, 512)
        x_std = torch.sqrt(((x ** 2) * attn).sum(dim=1) + 1e-5)  # (batch, 512)
        x = torch.cat([x_mean, x_std], dim=1)  # (batch, 1024)
        
        # Projection
        embedding = self.proj(x)  # (batch, embedding_dim)
        
        return embedding


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax"""
    def __init__(self, embedding_dim=256, num_speakers=5, scale=30.0, margin=0.25):
        super(AAMSoftmax, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        self.scale = scale
        self.margin = margin
        
        self.weight = nn.Parameter(torch.randn(num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        """
        Args:
            x: embeddings (batch, embedding_dim)
            labels: speaker labels (batch,)
        Returns:
            logits: (batch, num_speakers)
            embeddings: (batch, embedding_dim)
        """
        # Normalize
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        logits = torch.mm(x_norm, w_norm.t()) * self.scale  # (batch, num_speakers)
        
        if labels is not None and self.training:
            # Add margin to correct class
            batch_size = x.size(0)
            m_hot = torch.zeros_like(logits)
            m_hot.scatter_(1, labels.view(-1, 1), self.margin)
            logits = logits - m_hot
        
        return logits


class SpeakerClassifier(nn.Module):
    """Full speaker classification model"""
    def __init__(self, embedding_dim=256, num_speakers=5, aamsm_scale=30.0, aamsm_margin=0.25):
        super(SpeakerClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        
        self.backbone = Backbone(embedding_dim=embedding_dim)
        self.classifier = AAMSoftmax(embedding_dim=embedding_dim, num_speakers=num_speakers,
                                    scale=aamsm_scale, margin=aamsm_margin)

    def forward(self, x, lengths=None, labels=None):
        """
        Args:
            x: logmels (batch, 1, freq, time)
            lengths: sequence lengths
            labels: speaker labels (optional, for training)
        Returns:
            logits: (batch, num_speakers)
        """
        embeddings = self.backbone(x, lengths)
        logits = self.classifier(embeddings, labels)
        return logits, embeddings

    def embed(self, x, lengths=None):
        """Extract embeddings"""
        with torch.no_grad():
            embeddings = self.backbone(x, lengths)
        return embeddings

    def infer(self, x, lengths=None, threshold=0.5):
        """Speaker identification"""
        with torch.no_grad():
            embeddings = self.backbone(x, lengths)  # (1, embedding_dim)
            # Normalize
            embedding_norm = F.normalize(embeddings, p=2, dim=1)  # (1, embedding_dim)
            weight_norm = F.normalize(self.classifier.weight, p=2, dim=1)  # (num_speakers, embedding_dim)
            # Handle case where there are no speakers
            if weight_norm.shape[0] == 0:
                raise ValueError("No enrolled speakers: classifier weight has zero rows.")
            # Cosine similarities
            similarities = torch.mm(embedding_norm, weight_norm.t())  # (1, num_speakers)
            if similarities.shape[1] == 0:
                raise ValueError("No enrolled speakers: similarity matrix has zero columns.")
            speaker_id = similarities.argmax(dim=1).item()
            confidence = similarities[0, speaker_id].item()
            return speaker_id, confidence
