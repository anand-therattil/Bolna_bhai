import asyncio
import json
import io
import base64
import logging
from typing import Optional

import torch
import soundfile as sf
import websockets
from websockets.server import serve

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables
model = None
tokenizer = None
description_tokenizer = None
device = None


def load_model():
    """Load the TTS model and tokenizers."""
    global model, tokenizer, description_tokenizer, device
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    logger.info("Loading Indic Parler TTS model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts"
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )
    
    logger.info("Model loaded successfully!")


def generate_audio(prompt: str, description: Optional[str] = None) -> bytes:
    """
    Generate audio from text prompt.
    
    Args:
        prompt: The text to synthesize
        description: Voice description (optional, uses default if not provided)
    
    Returns:
        WAV audio data as bytes
    """
    if description is None:
        description = (
            "A female speaker delivers a slightly expressive and animated speech "
            "with a moderate speed and pitch. The recording is of very high quality, "
            "with the speaker's voice sounding clear and very close up."
        )
    
    # Tokenize inputs
    description_input_ids = description_tokenizer(
        description, return_tensors="pt"
    ).to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate audio
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask,
        )
    
    audio_arr = generation.cpu().numpy().squeeze()
    
    # Convert to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio_arr, model.config.sampling_rate, format="WAV")
    buffer.seek(0)
    
    return buffer.read()


async def handle_client(websocket):
    """Handle WebSocket client connections."""
    client_id = id(websocket)
    logger.info(f"Client connected: {client_id}")
    
    try:
        async for message in websocket:
            try:
                # Parse incoming message
                data = json.loads(message)
                prompt = data.get("prompt", "")
                description = data.get("description")
                response_format = data.get("format", "base64")  # base64 or binary
                
                if not prompt:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "message": "No prompt provided"
                    }))
                    continue
                
                logger.info(f"Generating audio for: {prompt[:50]}...")
                
                # Generate audio
                audio_bytes = generate_audio(prompt, description)
                
                # Send response based on requested format
                if response_format == "binary":
                    # Send binary audio directly
                    await websocket.send(audio_bytes)
                else:
                    # Send as base64 encoded JSON
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                    await websocket.send(json.dumps({
                        "status": "success",
                        "audio": audio_base64,
                        "sample_rate": model.config.sampling_rate,
                        "format": "wav"
                    }))
                
                logger.info(f"Audio sent to client {client_id}")
                
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                await websocket.send(json.dumps({
                    "status": "error",
                    "message": str(e)
                }))
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_id}")


async def main(host: str = "0.0.0.0", port: int = 8765):
    """Start the WebSocket server."""
    # Load model before starting server
    load_model()
    
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    
    async with serve(handle_client, host, port):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Indic Parler TTS WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8763, help="Port to listen on")
    
    args = parser.parse_args()
    
    asyncio.run(main(args.host, args.port))