import asyncio
import json
import base64
import io
from pathlib import Path

import websockets
import soundfile as sf
from neuttsair.neutts import NeuTTSAir


# ============ HARDCODED REFERENCE CONFIGURATION ============
REF_AUDIO_PATH = "samples/dave.wav"
REF_TEXT_PATH = "samples/dave.txt"
# ===========================================================


class TTSWebSocketServer:
    def __init__(self, host="0.0.0.0", port=8764):
        self.host = host
        self.port = port
        self.tts = None
        self.ref_codes = None
        self.ref_text = None
    
    def initialize_tts(self):
        """Initialize the TTS model and load reference."""
        print("Loading TTS models...")
        self.tts = NeuTTSAir(
            backbone_repo="neuphonic/neutts-air",
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )
        print("TTS models loaded successfully.")
        
        # Load hardcoded reference
        print(f"Loading reference audio: {REF_AUDIO_PATH}")
        self.ref_text = Path(REF_TEXT_PATH).read_text().strip()
        self.ref_codes = self.tts.encode_reference(REF_AUDIO_PATH)
        print("Reference voice loaded successfully.")
    
    async def handle_client(self, websocket):
        """Handle incoming WebSocket connections."""
        client_addr = websocket.remote_address
        print(f"Client connected: {client_addr}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.process_request(data)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "Invalid JSON"
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": str(e)
                    }))
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {client_addr}")
    
    async def process_request(self, data: dict) -> dict:
        """Process a TTS request."""
        action = data.get("action", "synthesize")
        
        if action == "synthesize":
            return await self.synthesize(data)
        elif action == "ping":
            return {"status": "success", "message": "pong"}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
    
    async def synthesize(self, data: dict) -> dict:
        """Synthesize speech from text."""
        text = data.get("text")
        output_format = data.get("format", "wav")
        sample_rate = data.get("sample_rate", 24000)
        
        if not text:
            return {"status": "error", "message": "text is required"}
        
        # Synthesize audio using pre-loaded reference
        loop = asyncio.get_event_loop()
        wav = await loop.run_in_executor(
            None,
            self.tts.infer,
            text,
            self.ref_codes,
            self.ref_text
        )
        
        # Convert to bytes
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wav, sample_rate, format=output_format.upper())
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
        
        # Encode as base64 for JSON transport
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        return {
            "status": "success",
            "audio": audio_base64,
            "format": output_format,
            "sample_rate": sample_rate,
            "duration": len(wav) / sample_rate
        }
    
    async def start(self):
        """Start the WebSocket server."""
        self.initialize_tts()
        
        print(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever


def main():
    server = TTSWebSocketServer(host="0.0.0.0", port=8764)
    asyncio.run(server.start())


if __name__ == "__main__":
    main()