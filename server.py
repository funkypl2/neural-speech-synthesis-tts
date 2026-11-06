"""
FastAPI & WebSocket Server for Streaming Neural Speech Synthesis.
"""
from __future__ import annotations
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import io
import struct
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel
from streaming.streaming_synthesizer import StreamingTTSSynthesizer

app = FastAPI(title="Neural TTS Streaming Gateway")
engine = StreamingTTSSynthesizer(sample_rate=16000)


class SynthesisRequest(BaseModel):
    text: str
    speed: float = 1.0
    pitch: float = 1.0


@app.get("/health")
def health():
    return {"status": "ok", "service": "neural-speech-synthesis-tts"}


@app.post("/synthesize")
def synthesize_http(req: SynthesisRequest):
    wav, _ = engine.synthesize_full(req.text, speed_ratio=req.speed, pitch_scale=req.pitch)
    int_data = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(int_data) * 2))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, 16000, 16000 * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(int_data) * 2))
    buf.write(int_data.tobytes())

    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.websocket("/ws/tts")
async def websocket_tts_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            for clause, chunk, lat_ms in engine.synthesize_stream(text):
                int_data = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                await websocket.send_bytes(int_data.tobytes())
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
