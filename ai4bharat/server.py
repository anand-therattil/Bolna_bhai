import asyncio
import os
import io
import numpy as np
import soundfile as sf
from huggingface_hub import login
import websockets
import json

from model import ASR
from vad import SileroVAD


class ASRWebSocketServer:
    def __init__(self):
        token = os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            login(token=token)
        
        self.asr_model = ASR()
        self.vad_model = SileroVAD()
    
    async def process_audio(self, websocket):
        """Handle WebSocket connections and process audio streams"""
        print(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Received binary audio data
                    await self.handle_audio_data(websocket, message)
                else:
                    # Received text message (JSON)
                    await self.handle_text_message(websocket, message)
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            print(f"Error: {e}")
            await websocket.send(json.dumps({
                "error": str(e),
                "status": "error"
            }))
    
    async def handle_audio_data(self, websocket, audio_bytes):
        """Process incoming audio data"""
        try:
            # Convert bytes to audio array
            audio_io = io.BytesIO(audio_bytes)
            audio_array, sr = sf.read(audio_io, dtype='float32')
            
            # Run VAD to detect speech segments
            timestamps = self.vad_model.has_speech(audio_array)
            
            if timestamps:
                
                # Run ASR
                result = self.asr_model(audio_array, lang="hi", sr=sr, timestamps=timestamps)
                
                # Clean up temp file
                # os.remove(temp_path)
                
                # Send result back to client
                response = {
                    "status": "success",
                    "transcription": result,
                    "timestamps": timestamps,
                    "sample_rate": sr
                }
            else:
                response = {
                    "status": "no_speech",
                    "message": "No speech detected in audio"
                }
            
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            await websocket.send(json.dumps({
                "status": "error",
                "error": str(e)
            }))
    
    async def handle_text_message(self, websocket, message):
        """Handle text/JSON messages from client"""
        try:
            data = json.loads(message)
            
            if data.get("type") == "ping":
                await websocket.send(json.dumps({"type": "pong"}))
            elif data.get("type") == "config":
                # Handle configuration updates
                response = {
                    "status": "config_received",
                    "config": data.get("config", {})
                }
                await websocket.send(json.dumps(response))
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "status": "error",
                "error": "Invalid JSON"
            }))
    
    async def start(self, host="0.0.0.0", port=8765):
        """Start the WebSocket server"""
        print(f"Starting ASR WebSocket server on ws://{host}:{port}")
        async with websockets.serve(self.process_audio, host, port):
            await asyncio.Future()  # Run forever


if __name__ == "__main__":
    server = ASRWebSocketServer()
    asyncio.run(server.start())