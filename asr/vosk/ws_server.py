import os
import yaml
from pathlib import Path
import asyncio
import json
import websockets
from vosk import Model, KaldiRecognizer
import soundfile as sf
import io

# ------------------------------------------------------------
#  Locate project root (directory that contains /config folder)
# ------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]     # go up: vosk → asr → root

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# ------------------------------------------------------------
#  Load config safely
# ------------------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

VOSK_CONFIG = CONFIG["asr"]["vosk"]
SAMPLE_RATE = VOSK_CONFIG.get("sample_rate", 16000)
HOST = VOSK_CONFIG.get("host", "0.0.0.0")
PORT = VOSK_CONFIG.get("port", 8765)


class VoskWebSocketServer:
    global SAMPLE_RATE, HOST, PORT
    def __init__(self, model_path):
        print("Loading Vosk Model...")
        self.model = Model(model_path)
        print("Model loaded.")

    async def process_client(self, websocket):
        print(f"Client connected: {websocket.remote_address}")

        # Create recognizer per client
        rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        rec.SetWords(True)

        try:
            async for message in websocket:
                # -----------------------------
                #  1) BINARY AUDIO MESSAGE
                # -----------------------------
                if isinstance(message, bytes):
                    await self.handle_audio_bytes(websocket, rec, message)

                # -----------------------------
                #  2) JSON TEXT MESSAGE
                # -----------------------------
                else:
                    await self.handle_json_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")

    async def handle_audio_bytes(self, websocket, rec, audio_bytes):
        """
        Incoming audio bytes → decode → feed recognizer → return result.
        """
        try:
            audio_io = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_io, dtype='int16')

            if sr != SAMPLE_RATE:
                await websocket.send(json.dumps({
                    "status": "error",
                    "error": f"Sample rate mismatch. Expected {SAMPLE_RATE}, got {sr}"
                }))
                return

            # Feed raw PCM into recognizer
            pcm_bytes = audio.tobytes()

            if rec.AcceptWaveform(pcm_bytes):
                res = json.loads(rec.Result())
                await websocket.send(json.dumps({
                    "status": "final",
                    "result": res
                }))
            else:
                res = json.loads(rec.PartialResult())
                await websocket.send(json.dumps({
                    "status": "partial",
                    "result": res
                }))

        except Exception as e:
            await websocket.send(json.dumps({
                "status": "error",
                "error": str(e)
            }))

    async def handle_json_message(self, websocket, text):
        """Handles JSON messages like ping/config."""
        try:
            data = json.loads(text)

            if data.get("type") == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

            elif data.get("type") == "config":
                await websocket.send(json.dumps({
                    "status": "config_received",
                    "config": data.get("config", {})
                }))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "status": "error",
                "error": "Invalid JSON received"
            }))

    async def start(self, host=HOST, port=PORT):
        print(f"Starting Vosk WebSocket server on ws://{host}:{port}")
        async with websockets.serve(self.process_client, host, port):
            await asyncio.Future() 


if __name__ == "__main__":
    server = VoskWebSocketServer(
        model_path=VOSK_CONFIG["model_path"]

    )
    asyncio.run(server.start())
