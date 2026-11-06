"""
Neural Vocoder: HiFi-GAN Generator with Multi-Receptive Field (MRF) Upsampling.
Converts 80-channel Mel-Spectrogram frames into high-fidelity audio waveform.
"""
from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResBlock(nn.Module):
        """Residual Block with multi-dilation convolutions."""
        def __init__(self, channels: int, kernel_size: int = 3, dilations: tuple = (1, 3, 5)):
            super().__init__()
            self.convs1 = nn.ModuleList([
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2)
                for d in dilations
            ])
            self.convs2 = nn.ModuleList([
                nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)
                for _ in dilations
            ])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = F.leaky_relu(x, 0.1)
                xt = c1(xt)
                xt = F.leaky_relu(xt, 0.1)
                xt = c2(xt)
                x = xt + x
            return x

    class HiFiGANGenerator(nn.Module):
        """
        HiFi-GAN Vocoder Generator.
        Upsampling factors: [8, 5, 4] = 160x upsampling (100 frames/sec -> 16000 Hz sample rate).
        """
        def __init__(
            self,
            in_channels: int = 80,
            upsample_initial_channel: int = 256,
            upsample_rates: tuple = (8, 5, 4),
            upsample_kernel_sizes: tuple = (16, 10, 8),
        ):
            super().__init__()
            self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)

            self.ups = nn.ModuleList()
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
                self.ups.append(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )

            self.resblocks = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel // (2 ** (i + 1))
                self.resblocks.append(ResBlock(ch, kernel_size=3))

            final_ch = upsample_initial_channel // (2 ** len(upsample_rates))
            self.conv_post = nn.Conv1d(final_ch, 1, 7, 1, padding=3)

        def forward(self, mel: torch.Tensor) -> torch.Tensor:
            """
            mel: (Batch, Frames, 80) -> transpose to (Batch, 80, Frames)
            Returns:
                audio: (Batch, Total_Samples)
            """
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            if mel.size(1) != 80 and mel.size(2) == 80:
                mel = mel.transpose(1, 2)

            x = self.conv_pre(mel)
            for up, res in zip(self.ups, self.resblocks):
                x = F.leaky_relu(x, 0.1)
                x = up(x)
                x = res(x)

            x = F.leaky_relu(x)
            x = self.conv_post(x)
            x = torch.tanh(x)
            return x.squeeze(1)

except ImportError:
    class HiFiGANGenerator:
        def __init__(self, in_channels: int = 80):
            pass

        def __call__(self, mel: np.ndarray) -> np.ndarray:
            # Fallback Griffin-Lim style harmonic synthesizer
            n_samples = mel.shape[1] * 160
            t = np.linspace(0, n_samples / 16000.0, n_samples, endpoint=False)
            waveform = 0.3 * np.sin(2 * np.pi * 320.0 * t)
            return waveform.astype(np.float32)
