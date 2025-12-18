import asyncio
import json
import yaml
from pathlib import Path
import websockets
from vosk import Model, KaldiRecognizer

# ------------------------------------------------------------
# Locate project root
# ------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

VOSK_CONFIG = CONFIG["asr"]["vosk"]
SAMPLE_RATE = VOSK_CONFIG.get("sample_rate", 16000)

# ------------------------------------------------------------
# Vosk Streaming Server
# ------------------------------------------------------------
class VoskWebSocketServer:
    def __init__(self, model_path: str):
        print("[ASR] Loading Vosk model...")
        self.model = Model(model_path)
        print("[ASR] Model loaded")

    async def process_client(self, websocket):
        print(f"[ASR] Client connected: {websocket.remote_address}")

        rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        rec.SetWords(True)

        try:
            async for message in websocket:

                # ------------------------------------------------
                # AUDIO: raw PCM16 bytes ONLY
                # ------------------------------------------------
                if isinstance(message, bytes):
                    await self.handle_pcm(websocket, rec, message)

                # ------------------------------------------------
                # CONTROL / JSON
                # ------------------------------------------------
                else:
                    await self.handle_json(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            print(f"[ASR] Client disconnected: {websocket.remote_address}")

    async def handle_pcm(self, websocket, rec, pcm_bytes: bytes):
        """
        Handle raw PCM16 audio frames.
        """
        if rec.AcceptWaveform(pcm_bytes):
            result = json.loads(rec.Result())
            await websocket.send(json.dumps({
                "type": "final",
                "text": result.get("text", ""),
                "result": result
            }))
        else:
            partial = json.loads(rec.PartialResult())
            await websocket.send(json.dumps({
                "type": "partial",
                "text": partial.get("partial", "")
            }))

    async def handle_json(self, websocket, text_msg: str):
        """
        Handle control messages.
        """
        try:
            data = json.loads(text_msg)

            if data.get("type") == "ping":
                await websocket.send(json.dumps({"type": "pong"}))

            elif data.get("type") == "end_of_utterance":
                await websocket.send(json.dumps({
                    "type": "final",
                    "text": json.loads(rec.FinalResult()).get("text", "")
                }))

        except Exception as e:
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e)
            }))

    async def start(self, host="0.0.0.0", port=8763):
        print(f"[ASR] Streaming server started at ws://{host}:{port}")
        async with websockets.serve(self.process_client, host, port):
            await asyncio.Future()


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    server = VoskWebSocketServer(
        model_path= VOSK_CONFIG["model_path"]
    )
    asyncio.run(server.start())
