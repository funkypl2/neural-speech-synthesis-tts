"""
Unit tests for Neural Speech Synthesis (TTS) Suite.
"""
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
import numpy as np
from frontend.text_normalizer import TextNormalizer
from frontend.g2p import G2PConverter
from streaming.streaming_synthesizer import StreamingTTSSynthesizer
from metrics.audio_eval import compute_mcd, compute_rtf


def test_text_normalizer():
    tn = TextNormalizer()
    res1 = tn.normalize("2026年5月")
    assert "二零二六年" in res1
    assert "五月" in res1

    res2 = tn.normalize("准确率达到95.5%")
    assert "百分之九十五点五" in res2

    res3 = tn.normalize("你好，世界！")
    assert "#2" in res3
    assert "#3" in res3


def test_g2p_converter():
    g2p = G2PConverter()
    norm = "你好 #2 世界 #3"
    phonemes = g2p.text_to_phonemes(norm)
    assert "<sil>" in phonemes
    assert "<sp>" in phonemes
    ids = g2p.text_to_ids(norm)
    assert len(ids) == len(phonemes)
    assert all(isinstance(i, int) for i in ids)


def test_streaming_tts_synthesizer():
    engine = StreamingTTSSynthesizer(sample_rate=16000)
    text = "语音合成，测试通过！"
    chunks = list(engine.synthesize_stream(text))
    assert len(chunks) >= 2
    for clause, chunk, lat in chunks:
        assert len(chunk) > 0
        assert lat >= 0.0

    wav, elapsed = engine.synthesize_full(text)
    assert len(wav) > 0
    assert elapsed > 0.0


def test_audio_eval_metrics():
    mel1 = np.ones((20, 80), dtype=np.float32)
    mel2 = np.ones((20, 80), dtype=np.float32)
    mcd_zero = compute_mcd(mel1, mel2)
    assert mcd_zero == 0.0

    rtf = compute_rtf(elapsed_time=0.1, audio_duration_sec=2.0)
    assert rtf == 0.05
