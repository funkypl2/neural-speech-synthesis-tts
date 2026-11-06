"""
CLI for Neural Speech Synthesis (TTS).
Synthesizes text into high-fidelity .wav files with controllable pitch and speed.
"""
from __future__ import annotations
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import argparse
import struct
import numpy as np
from streaming.streaming_synthesizer import StreamingTTSSynthesizer
from metrics.audio_eval import compute_rtf


def save_wav(waveform: np.ndarray, path: str, sample_rate: int = 16000):
    """Save 1D float32 waveform (-1.0 to 1.0) into 16-bit PCM WAV."""
    int_data = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(int_data) * 2))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(int_data) * 2))
        f.write(int_data.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Neural Speech Synthesis (TTS) CLI")
    parser.add_argument("--text", type=str, default="欢迎使用神经语音合成系统，支持极速流式分块发音！", help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output_syn.wav", help="Output WAV path")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier (>1.0 faster, <1.0 slower)")
    parser.add_argument("--pitch", type=float, default=1.0, help="Pitch scale multiplier")
    parser.add_argument("--stream", action="store_true", help="Demonstrate clause-level streaming synthesis")
    args = parser.parse_args()

    engine = StreamingTTSSynthesizer(sample_rate=16000)

    print("=================================================================")
    print("      Neural Speech Synthesis (TTS) Studio                       ")
    print("=================================================================")
    print(f"[*] Input Text: '{args.text}'")
    print(f"[*] Speed Multiplier: {args.speed}x | Pitch Multiplier: {args.pitch}x\n")

    if args.stream:
        print("[*] Streaming Clause-by-Clause Synthesis Mode:")
        total_chunks = []
        for i, (clause, chunk, lat_ms) in enumerate(engine.synthesize_stream(args.text, speed_ratio=args.speed, pitch_scale=args.pitch)):
            total_chunks.append(chunk)
            dur_sec = len(chunk) / 16000.0
            print(f"  [Chunk {i+1}] '{clause}' -> {dur_sec:.2f}s audio (Time-to-Audio: {lat_ms:.1f}ms)")
        full_wav = np.concatenate(total_chunks) if total_chunks else np.array([], dtype=np.float32)
    else:
        full_wav, elapsed = engine.synthesize_full(args.text, speed_ratio=args.speed, pitch_scale=args.pitch)
        audio_dur = len(full_wav) / 16000.0
        rtf = compute_rtf(elapsed, audio_dur)
        print(f"[+] Synthesis Complete!")
        print(f"    - Audio Duration: {audio_dur:.2f}s")
        print(f"    - Processing Time: {elapsed*1000:.1f}ms")
        print(f"    - Real-Time Factor (RTF): {rtf:.4f}")

    if len(full_wav) > 0:
        save_wav(full_wav, args.output, sample_rate=16000)
        print(f"\n[+] Audio successfully saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
