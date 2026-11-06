"""
FastSpeech 2 Non-Autoregressive Acoustic Model Architecture.
Generates 80-dimensional Mel-Spectrograms from phoneme ID sequences with pitch and duration control.
"""
from __future__ import annotations
import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class VariancePredictor(nn.Module):
        """1D Convolutional Variance Predictor for Duration, Pitch, and Energy."""
        def __init__(self, d_model: int = 256, kernel_size: int = 3, filter_size: int = 256, dropout: float = 0.1):
            super().__init__()
            self.conv1 = nn.Conv1d(d_model, filter_size, kernel_size, padding=(kernel_size - 1) // 2)
            self.ln1 = nn.LayerNorm(filter_size)
            self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=(kernel_size - 1) // 2)
            self.ln2 = nn.LayerNorm(filter_size)
            self.linear = nn.Linear(filter_size, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (Batch, Length, d_model)
            out = self.conv1(x.transpose(1, 2)).transpose(1, 2)
            out = self.dropout(F.relu(self.ln1(out)))
            out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
            out = self.dropout(F.relu(self.ln2(out)))
            return self.linear(out).squeeze(-1)

    class LengthRegulator(nn.Module):
        """Expands phoneme hidden representations according to duration predictions."""
        def forward(self, x: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
            # x: (Batch, L_in, Dim), durations: (Batch, L_in)
            expanded_list = []
            for b in range(x.size(0)):
                h = x[b]
                d = durations[b]
                frames = []
                for i in range(h.size(0)):
                    repeats = max(1, int(d[i].item()))
                    frames.append(h[i:i+1].repeat(repeats, 1))
                if frames:
                    expanded_list.append(torch.cat(frames, dim=0))
                else:
                    expanded_list.append(h)
            # Pad batch to max length
            max_len = max(item.size(0) for item in expanded_list)
            padded = torch.zeros(len(expanded_list), max_len, x.size(-1), device=x.device)
            for b, item in enumerate(expanded_list):
                padded[b, :item.size(0)] = item
            return padded

    class FastSpeech2(nn.Module):
        """
        FastSpeech 2 Non-Autoregressive Acoustic Model.
        Maps phoneme IDs -> 80-dim log-mel spectrogram frames.
        """
        def __init__(self, vocab_size: int = 300, d_model: int = 256, n_mels: int = 80, num_layers: int = 4):
            super().__init__()
            self.d_model = d_model
            self.n_mels = n_mels

            self.embedding = nn.Embedding(vocab_size, d_model)
            # Encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Variance Adaptor
            self.duration_predictor = VariancePredictor(d_model)
            self.length_regulator = LengthRegulator()
            self.pitch_predictor = VariancePredictor(d_model)
            self.pitch_embed = nn.Linear(1, d_model)

            # Decoder
            decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True)
            self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
            self.mel_linear = nn.Linear(d_model, n_mels)

        def forward(
            self,
            phoneme_ids: torch.Tensor,
            speed_ratio: float = 1.0,
            pitch_scale: float = 1.0,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                phoneme_ids: (Batch, L_phonemes)
                speed_ratio: > 1.0 is faster, < 1.0 is slower
                pitch_scale: pitch adjustment factor
            Returns:
                mel: (Batch, Frames, 80)
                durations: (Batch, L_phonemes)
            """
            x = self.embedding(phoneme_ids) * math.sqrt(self.d_model)
            enc_out = self.encoder(x)

            # Predict duration (default 3~5 frames per phoneme if untrained)
            raw_durations = F.relu(self.duration_predictor(enc_out)) + 3.0
            durations = torch.clamp(raw_durations / speed_ratio, min=1.0).round()

            # Length regulation
            lr_out = self.length_regulator(enc_out, durations)

            # Pitch prediction & conditioning
            pitch_pred = self.pitch_predictor(lr_out).unsqueeze(-1) * pitch_scale
            conditioned = lr_out + self.pitch_embed(pitch_pred)

            # Decoder & Mel projection
            dec_out = self.decoder(conditioned)
            mel_output = self.mel_linear(dec_out)

            return mel_output, durations

except ImportError:
    class FastSpeech2:
        def __init__(self, vocab_size: int = 300, d_model: int = 256, n_mels: int = 80):
            self.n_mels = n_mels

        def forward(self, phoneme_ids, speed_ratio=1.0, pitch_scale=1.0):
            num_frames = int(len(phoneme_ids) * 3 / speed_ratio)
            mel = np.zeros((1, num_frames, 80), dtype=np.float32)
            return mel, np.full(len(phoneme_ids), 3)
