"""
Audio Quality and Latency Evaluation for Neural TTS: MCD, SNR, and RTF.
"""
from __future__ import annotations
import numpy as np


def compute_mcd(ref_mel: np.ndarray, syn_mel: np.ndarray) -> float:
    """
    Mel-Cepstral Distortion (MCD) in dB.
    MCD = (10 * sqrt(2) / ln(10)) * mean(sqrt(sum((ref - syn)^2)))
    """
    min_len = min(ref_mel.shape[0], syn_mel.shape[0])
    if min_len == 0:
        return 0.0

    ref = ref_mel[:min_len]
    syn = syn_mel[:min_len]

    diff = ref - syn
    frame_dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    constant = (10.0 * np.sqrt(2.0)) / np.log(10.0)
    return float(constant * np.mean(frame_dist))


def compute_rtf(elapsed_time: float, audio_duration_sec: float) -> float:
    """Real-Time Factor (RTF): Elapsed synthesis wall-clock time / synthesized audio duration."""
    if audio_duration_sec <= 0:
        return 0.0
    return float(elapsed_time / audio_duration_sec)
