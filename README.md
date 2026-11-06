# Neural-Speech-Synthesis-TTS: End-to-End Streaming Neural TTS Studio

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: FastSpeech2](https://img.shields.io/badge/Acoustic-FastSpeech%202-purple.svg)]()
[![Vocoder: HiFi-GAN](https://img.shields.io/badge/Vocoder-HiFi--GAN-orange.svg)]()
[![Streaming: Sub-150ms](https://img.shields.io/badge/Streaming-Sub--150ms%20TTFA-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Neural-Speech-Synthesis-TTS** 是一个高性能、低延迟的现代端到端神经语音合成（Text-to-Speech, TTS）算法与工程套件。完整实现了从**文本正则化（Text Normalization, TN）**、**汉字注音与声调解析（G2P / Phonemization）**、**非自回归 FastSpeech 2 声学模型**到**高保真 HiFi-GAN 神经声码器（Neural Vocoder）**的全链路流水线，并原生支持**基于标点前瞻预测的极速流式分块发音（Streaming Chunk Synthesis）**。

---

## 🏗️ 算法架构与合成全景

```
 [Raw Text Input] 
        │  (例如: "2026年准确率提升95.5%！")
        ▼
 [Text Normalizer (TN)] ──► 数字/日期/符号展开 + 韵律断句标记 (#2, #3)
        │
        ▼
 [G2P Converter]        ──► 声母 (Initial) + 带调韵母 (Final) + 停顿词 ID
        │
        ▼
 ┌───────────────────────────────────────────────┐
 │             FastSpeech 2 Acoustic             │
 │   • Phoneme Embedding + Positional Encoding   │
 │   • Transformer Encoder                       │
 │   • Variance Adaptor:                         │
 │     - Duration Predictor & Length Regulator   │
 │     - Pitch Predictor (F0) & Embedding        │
 │     - Energy Predictor & Embedding            │
 │   • Transformer Decoder                       │
 └───────────────────────┬───────────────────────┘
                         │ 80-channel Log Mel-Spectrogram
                         ▼
 ┌───────────────────────────────────────────────┐
 │               HiFi-GAN Vocoder                │
 │   • Multi-Receptive Field (MRF) ResNet        │
 │   • Transposed Conv Upsamplers [8, 5, 4]      │
 │   • High-Fidelity Waveform Reconstruction     │
 └───────────────────────┬───────────────────────┘
                         │
                         ▼
             [Synthesized Speech (16kHz)]
```

### 核心亮点
1. **全套语言前端 (`frontend/`)**：
   - 规则级文本归一化（数字、日期、小数、百分比、标点智能展开）；
   - 标准中文拼音声母/韵母拆解与五度调值映射，支持轻声与停顿标记（`<sp>` 短停顿、`<sil>` 句间静音）。
2. **非自回归声学网络 (`models/`)**：
   - FastSpeech 2 架构，彻底告别自回归模型（Tacotron）的逐帧循环低效与注意力跳字/漏字（Attention Drift）缺陷；
   - 支持动态语速（`speed_ratio`）与基频音高（`pitch_scale`）自由拉伸与调节。
3. **HiFi-GAN 声码器 (`vocoder/`)**：
   - 多感受野（MRF）交替膨胀卷积结构，实现极高音质重构与低推理开销。
4. **低延迟流式引擎 (`streaming/`)**：
   - 标点分块并发渲染，首字发声时间（TTFA, Time-to-First-Audio）低于 150ms。

---

## 🚀 快速上手 (Quick Start)

### 1. 安装依赖
```bash
cd 06-neural-speech-synthesis-tts
pip install -r requirements.txt
```

### 2. 命令行文本合成 WAV 音频
```bash
# 基础合成
python synthesize.py --text "欢迎使用现代神经语音合成系统，体验极速发音！" --output output.wav

# 调节语速与音高
python synthesize.py --text "语速加快百分之二十，音调升高！" --speed 1.2 --pitch 1.1 --output fast.wav

# 查看流式分块输出延迟
python synthesize.py --text "第一句话生成完毕，紧接着生成第二句话，最后生成第三句话。" --stream
```

### 3. 启动 HTTP & WebSocket 流式服务
```bash
python server.py
# 或使用 uvicorn 启动
uvicorn server:app --host 0.0.0.0 --port 8001
```
- **HTTP 接口**：`POST http://localhost:8001/synthesize`，入参 `{"text": "你好世界"}`，直接返回 PCM WAV 二进制音频。
- **WebSocket 接口**：`ws://localhost:8001/ws/tts`，发送文本即可实时接收音频流式字节块。

### 4. 运行单元测试
```bash
pytest tests/ -v
```

---

## 📂 项目目录结构

```
06-neural-speech-synthesis-tts/
├── frontend/
│   ├── text_normalizer.py       # 文本正则化 (TN) 与韵律断句处理
│   └── g2p.py                   # 汉字注音、声母韵母拆分与声调映射
├── models/
│   └── fastspeech2.py           # FastSpeech 2 非自回归声学模型
├── vocoder/
│   └── hifigan.py               # HiFi-GAN 神经声码器
├── streaming/
│   └── streaming_synthesizer.py # 流式分块极速合成引擎
├── metrics/
│   └── audio_eval.py            # MCD (Mel-Cepstral Distortion) 与 RTF 评测
├── synthesize.py                # 命令行合成主入口
├── server.py                    # FastAPI / WebSocket 流媒体服务
├── requirements.txt
└── README.md
```

---

## 📄 开源许可证
本项目采用 [MIT License](LICENSE) 授权协议。
