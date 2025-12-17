import asyncio
import websockets
import soundfile as sf
import json
import time

WS_URL = "ws://localhost:8762"
FRAME_MS = 20
SAMPLE_RATE = 16000
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

async def stream_wav(path):
    audio, sr = sf.read(path, dtype="int16")

    if sr != SAMPLE_RATE:
        raise ValueError("WAV must be 16kHz")

    async with websockets.connect(WS_URL) as ws:
        print("[CLIENT] Connected")

        for i in range(0, len(audio), FRAME_SAMPLES):
            frame = audio[i:i + FRAME_SAMPLES]
            if len(frame) == 0:
                continue

            await ws.send(frame.tobytes())
            await asyncio.sleep(FRAME_MS / 1000)

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                print("[ASR]", msg)
            except asyncio.TimeoutError:
                pass

        await asyncio.sleep(1)

asyncio.run(stream_wav("<<PATH TO AUDIO>>"))
