"""
WebSocket-based Text-to-Speech Server using Indri TTS model.

This server accepts text via WebSocket connections and streams back audio data.
"""

import asyncio
import json
import io
import base64
import logging
import tempfile
import os
from typing import Optional

import torch
import torchaudio
from transformers import pipeline
import websockets
from websockets.asyncio.server import serve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TTSServer:
    """WebSocket TTS Server using Indri model."""
    
    def __init__(
        self,
        model_id: str = '11mlabs/indri-0.1-124m-tts',
        device: str = 'cpu',
        default_speaker: str = '[spkr_63]',
        sample_rate: int = 24000
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.default_speaker = default_speaker
        self.sample_rate = sample_rate
        self.pipe: Optional[pipeline] = None
        
    def load_model(self):
        """Load the TTS model pipeline."""
        logger.info(f"Loading model: {self.model_id}")
        self.pipe = pipeline(
            'indri-tts',
            model=self.model_id,
            device=self.device,
            trust_remote_code=True
        )
        logger.info("Model loaded successfully")
        
    def synthesize(self, text: str, speaker: Optional[str] = None) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: The text to convert to speech
            speaker: Optional speaker ID (defaults to self.default_speaker)
            
        Returns:
            WAV audio data as bytes
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        speaker = speaker or self.default_speaker
        
        # Generate audio
        output = self.pipe([text], speaker=speaker)
        audio_tensor = output[0]['audio'][0]
        
        # Use temporary file since torchaudio with TorchCodec doesn't support BytesIO
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            torchaudio.save(tmp_path, audio_tensor, sample_rate=self.sample_rate)
            with open(tmp_path, 'rb') as f:
                audio_bytes = f.read()
        finally:
            os.unlink(tmp_path)
        
        return audio_bytes
    
    async def handle_client(self, websocket):
        """
        Handle a WebSocket client connection.
        
        Expected message format (JSON):
        {
            "text": "Text to synthesize",
            "speaker": "[spkr_63]"  // optional
        }
        
        Response format (JSON):
        {
            "status": "success" | "error",
            "audio": "<base64-encoded WAV data>",  // on success
            "sample_rate": 24000,
            "error": "error message"  // on error
        }
        """
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")
        
        try:
            async for message in websocket:
                try:
                    # Parse the request
                    request = json.loads(message)
                    text = request.get('text', '')
                    speaker = request.get('speaker', self.default_speaker)
                    
                    if not text:
                        response = {
                            'status': 'error',
                            'error': 'No text provided'
                        }
                        await websocket.send(json.dumps(response))
                        continue
                    
                    logger.info(f"Synthesizing: '{text[:50]}...' with speaker {speaker}")
                    
                    # Run synthesis in a thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    audio_bytes = await loop.run_in_executor(
                        None, 
                        self.synthesize, 
                        text, 
                        speaker
                    )
                    
                    # Encode audio as base64 for JSON transmission
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    
                    response = {
                        'status': 'success',
                        'audio': audio_b64,
                        'sample_rate': self.sample_rate
                    }
                    
                    await websocket.send(json.dumps(response))
                    logger.info(f"Sent audio response ({len(audio_bytes)} bytes)")
                    
                except json.JSONDecodeError:
                    response = {
                        'status': 'error',
                        'error': 'Invalid JSON message'
                    }
                    await websocket.send(json.dumps(response))
                    
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    response = {
                        'status': 'error',
                        'error': str(e)
                    }
                    await websocket.send(json.dumps(response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
            
    async def start(self, host: str = 'localhost', port: int = 8765):
        """Start the WebSocket server."""
        self.load_model()
        
        logger.info(f"Starting WebSocket TTS server on ws://{host}:{port}")
        
        async with serve(self.handle_client, host, port):
            await asyncio.Future()  # Run forever


class TTSServerBinary(TTSServer):
    """
    Alternative WebSocket TTS Server that sends binary audio data directly.
    
    This is more efficient for large audio files as it avoids base64 encoding overhead.
    """
    
    async def handle_client(self, websocket):
        """
        Handle client with binary audio responses.
        
        Request format (JSON):
        {
            "text": "Text to synthesize",
            "speaker": "[spkr_63]"  // optional
        }
        
        Response: Binary WAV data (or JSON error message)
        """
        client_addr = websocket.remote_address
        logger.info(f"Client connected (binary mode): {client_addr}")
        
        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    text = request.get('text', '')
                    speaker = request.get('speaker', self.default_speaker)
                    
                    if not text:
                        response = {
                            'status': 'error',
                            'error': 'No text provided'
                        }
                        await websocket.send(json.dumps(response))
                        continue
                    
                    logger.info(f"Synthesizing: '{text[:50]}...' with speaker {speaker}")
                    
                    loop = asyncio.get_event_loop()
                    audio_bytes = await loop.run_in_executor(
                        None, 
                        self.synthesize, 
                        text, 
                        speaker
                    )
                    
                    # Send binary audio data directly
                    await websocket.send(audio_bytes)
                    logger.info(f"Sent binary audio ({len(audio_bytes)} bytes)")
                    
                except json.JSONDecodeError:
                    response = {
                        'status': 'error',
                        'error': 'Invalid JSON message'
                    }
                    await websocket.send(json.dumps(response))
                    
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    response = {
                        'status': 'error',
                        'error': str(e)
                    }
                    await websocket.send(json.dumps(response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='WebSocket TTS Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8760, help='Port to bind to')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--binary', action='store_true', help='Use binary mode for audio')
    parser.add_argument('--speaker', default='[spkr_63]', help='Default speaker ID')
    
    args = parser.parse_args()
    
    if args.binary:
        server = TTSServerBinary(
            device=args.device,
            default_speaker=args.speaker
        )
    else:
        server = TTSServer(
            device=args.device,
            default_speaker=args.speaker
        )
    
    asyncio.run(server.start(args.host, args.port))


if __name__ == '__main__':
    main()