import asyncio
import json
import os
import io
import base64
from dataclasses import dataclass
from typing import Optional

import websockets
from websockets.server import serve
import soundfile as sf

from helper import load_text_to_speech, timer, sanitize_filename, load_voice_style


@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8764
    onnx_dir: str = "assets/onnx"
    use_gpu: bool = False
    default_total_step: int = 5
    default_speed: float = 1.05
    default_voice_style: str = "assets/voice_styles/M1.json"
    save_dir: str = "results"
    save_to_disk: bool = False


class TTSWebSocketServer:
    """WebSocket server for Text-to-Speech synthesis."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.text_to_speech = None
        self.default_style = None
        self.clients: set = set()
        
    def initialize(self):
        """Initialize TTS model and default voice style."""
        print("=== TTS WebSocket Server ===\n")
        print("Loading TTS model...")
        self.text_to_speech = load_text_to_speech(
            self.config.onnx_dir, 
            self.config.use_gpu
        )
        print("Loading default voice style...")
        self.default_style = load_voice_style(
            [self.config.default_voice_style], 
            verbose=True
        )
        print("Server initialized successfully!\n")
        
    async def handle_synthesize(self, request: dict) -> dict:
        """Handle a synthesis request."""
        text = request.get("text", "")
        if not text:
            return {"error": "No text provided"}
        
        # Get optional parameters with defaults
        total_step = request.get("total_step", self.config.default_total_step)
        speed = request.get("speed", self.config.default_speed)
        voice_style_path = request.get("voice_style")
        
        # Load voice style (use default if not specified)
        if voice_style_path:
            try:
                style = load_voice_style([voice_style_path], verbose=False)
            except Exception as e:
                return {"error": f"Failed to load voice style: {str(e)}"}
        else:
            style = self.default_style
        
        try:
            # Synthesize speech
            with timer("Generating speech"):
                wav, duration = self.text_to_speech(text, style, total_step, speed)
            
            # Trim to actual duration
            sample_rate = self.text_to_speech.sample_rate
            trimmed_wav = wav[0, :int(sample_rate * duration[0].item())]
            
            # Encode audio to base64
            buffer = io.BytesIO()
            sf.write(buffer, trimmed_wav, sample_rate, format="WAV")
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            
            # Optionally save to disk
            if self.config.save_to_disk:
                if not os.path.exists(self.config.save_dir):
                    os.makedirs(self.config.save_dir)
                fname = f"{sanitize_filename(text, 20)}.wav"
                sf.write(
                    os.path.join(self.config.save_dir, fname),
                    trimmed_wav,
                    sample_rate
                )
                print(f"Saved: {self.config.save_dir}/{fname}")
            
            return {
                "status": "success",
                "audio": audio_base64,
                "sample_rate": sample_rate,
                "duration": float(duration[0].item()),
                "format": "wav"
            }
            
        except Exception as e:
            return {"error": f"Synthesis failed: {str(e)}"}
    
    async def handle_batch_synthesize(self, request: dict) -> dict:
        """Handle a batch synthesis request."""
        texts = request.get("texts", [])
        if not texts:
            return {"error": "No texts provided"}
        
        voice_style_paths = request.get("voice_styles", [])
        
        # Use default style for all if not specified
        if not voice_style_paths:
            voice_style_paths = [self.config.default_voice_style] * len(texts)
        
        if len(voice_style_paths) != len(texts):
            return {"error": "Number of voice styles must match number of texts"}
        
        total_step = request.get("total_step", self.config.default_total_step)
        speed = request.get("speed", self.config.default_speed)
        
        try:
            style = load_voice_style(voice_style_paths, verbose=False)
            
            with timer("Batch generating speech"):
                wav, duration = self.text_to_speech.batch(texts, style, total_step, speed)
            
            sample_rate = self.text_to_speech.sample_rate
            results = []
            
            for i, text in enumerate(texts):
                trimmed_wav = wav[i, :int(sample_rate * duration[i].item())]
                
                buffer = io.BytesIO()
                sf.write(buffer, trimmed_wav, sample_rate, format="WAV")
                buffer.seek(0)
                audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                
                results.append({
                    "text": text,
                    "audio": audio_base64,
                    "duration": float(duration[i].item())
                })
                
                if self.config.save_to_disk:
                    if not os.path.exists(self.config.save_dir):
                        os.makedirs(self.config.save_dir)
                    fname = f"{sanitize_filename(text, 20)}.wav"
                    sf.write(
                        os.path.join(self.config.save_dir, fname),
                        trimmed_wav,
                        sample_rate
                    )
            
            return {
                "status": "success",
                "results": results,
                "sample_rate": sample_rate,
                "format": "wav"
            }
            
        except Exception as e:
            return {"error": f"Batch synthesis failed: {str(e)}"}
    
    async def handle_message(self, websocket, message: str) -> dict:
        """Parse and route incoming messages."""
        try:
            request = json.loads(message)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}
        
        action = request.get("action", "synthesize")
        
        if action == "synthesize":
            return await self.handle_synthesize(request)
        elif action == "batch_synthesize":
            return await self.handle_batch_synthesize(request)
        elif action == "ping":
            return {"status": "pong"}
        elif action == "info":
            return {
                "status": "ok",
                "sample_rate": self.text_to_speech.sample_rate,
                "default_speed": self.config.default_speed,
                "default_total_step": self.config.default_total_step,
                "gpu_enabled": self.config.use_gpu
            }
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def handler(self, websocket):
        """Handle a WebSocket connection."""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"Client connected: {client_addr}")
        
        try:
            async for message in websocket:
                print(f"Received request from {client_addr}")
                response = await self.handle_message(websocket, message)
                await websocket.send(json.dumps(response))
                
        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {client_addr}")
        finally:
            self.clients.discard(websocket)
    
    async def start(self):
        """Start the WebSocket server."""
        self.initialize()
        
        print(f"Starting WebSocket server on ws://{self.config.host}:{self.config.port}")
        print("Waiting for connections...\n")
        
        async with serve(self.handler, self.config.host, self.config.port):
            await asyncio.Future()  # Run forever


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS WebSocket Server")
    
    # Server settings
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8764,
        help="Port to listen on (default: 8764)"
    )
    
    # Device settings
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="Use GPU for inference (default: CPU)"
    )
    
    # Model settings
    parser.add_argument(
        "--onnx-dir", type=str, default="assets/onnx",
        help="Path to ONNX model directory"
    )
    
    # Default synthesis parameters
    parser.add_argument(
        "--default-total-step", type=int, default=5,
        help="Default number of denoising steps"
    )
    parser.add_argument(
        "--default-speed", type=float, default=1.05,
        help="Default speech speed"
    )
    parser.add_argument(
        "--default-voice-style", type=str,
        default="assets/voice_styles/M1.json",
        help="Default voice style file path"
    )
    
    # Output settings
    parser.add_argument(
        "--save-dir", type=str, default="results",
        help="Output directory for saved files"
    )
    parser.add_argument(
        "--save-to-disk", action="store_true",
        help="Save generated audio to disk"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    config = ServerConfig(
        host=args.host,
        port=args.port,
        onnx_dir=args.onnx_dir,
        use_gpu=args.use_gpu,
        default_total_step=args.default_total_step,
        default_speed=args.default_speed,
        default_voice_style=args.default_voice_style,
        save_dir=args.save_dir,
        save_to_disk=args.save_to_disk
    )
    
    server = TTSWebSocketServer(config)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nServer stopped.")