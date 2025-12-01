import asyncio
import os
import io
import numpy as np
from huggingface_hub import login
import websockets
import json

from model import ASR


class ASRWebSocketServer:
    def __init__(self):
        token = os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            login(token=token)
        
        self.asr_model = ASR()
        self.client_configs = {}  # ← ADDED: Track per-client configuration
    
    async def process_audio(self, websocket):
        """Handle WebSocket connections and process audio streams"""
        print(f"Client connected: {websocket.remote_address}")
        
        # ← ADDED: Track this client's configuration
        client_id = id(websocket)
        self.client_configs[client_id] = {
            "sample_rate": 16000,
            "language": "hi"
        }
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Received binary audio data
                    await self.handle_audio_data(websocket, message, client_id)  # ← CHANGED: Added client_id
                else:
                    # Received text message (JSON)
                    await self.handle_text_message(websocket, message, client_id)  # ← CHANGED: Added client_id
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
            print(f"Error: {e}")
            await websocket.send(json.dumps({
                "error": str(e),
                "status": "error"
            }))
        finally:
            # ← ADDED: Cleanup client configuration
            if client_id in self.client_configs:
                del self.client_configs[client_id]
    
    async def handle_audio_data(self, websocket, audio_bytes, client_id):  # ← CHANGED: Added client_id parameter
        """Process incoming raw PCM audio data"""
        try:
            # ← CHANGED: Get client configuration instead of reading from file
            config = self.client_configs.get(client_id, {"sample_rate": 16000, "language": "hi"})
            sr = config["sample_rate"]
            lang = config["language"]
            
            # ← CHANGED: Convert raw PCM bytes to numpy array
            # Pipecat sends int16 PCM format, not WAV files with headers
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # ← CHANGED: Convert to float32 (normalize to [-1, 1])
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # ← ADDED: Debug logging
            print(f"Received audio: {len(audio_array)} samples at {sr}Hz")
            

            result = self.asr_model(audio_array, lang=lang, sr=sr)
            
            # Send result back to client
            response = {
                "status": "success",
                "transcription": result['text'],
                "sample_rate": sr
            }
            # ← ADDED: Debug logging
            print(f"{response=}")
            await websocket.send(json.dumps(response))
            
        except Exception as e:
            # ← CHANGED: Better error logging
            print(f"Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            await websocket.send(json.dumps({
                "status": "error",
                "error": str(e)
            }))
    
    async def handle_text_message(self, websocket, message, client_id):  # ← CHANGED: Added client_id parameter
        """Handle text/JSON messages from client"""
        try:
            data = json.loads(message)
            
            if data.get("type") == "ping":
                await websocket.send(json.dumps({"type": "pong"}))
            elif data.get("type") == "config":
                # ← CHANGED: Handle configuration updates and store them
                config_data = data.get("config", {})
                self.client_configs[client_id].update(config_data)
                
                response = {
                    "status": "config_received",
                    "config": self.client_configs[client_id]
                }
                await websocket.send(json.dumps(response))
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "status": "error",
                "error": "Invalid JSON"
            }))
    
    async def start(self, host="0.0.0.0", port=8761):
        """Start the WebSocket server"""
        print(f"Starting ASR WebSocket server on ws://{host}:{port}")
        async with websockets.serve(self.process_audio, host, port):
            await asyncio.Future()  # Run forever


if __name__ == "__main__":
    server = ASRWebSocketServer()
    asyncio.run(server.start())