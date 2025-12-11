"""
Example WebSocket client for the TTS server.

Usage:
    python tts_websocket_client.py --text "Hello, world!"
    python tts_websocket_client.py --text "Hello" --speed 1.2 --output greeting.wav
    python tts_websocket_client.py --batch --texts "Hello" "Goodbye"
"""

import asyncio
import argparse
import json
import base64

import websockets


async def synthesize_single(
    uri: str,
    text: str,
    speed: float = 1.05,
    total_step: int = 5,
    voice_style: str = None,
    output_file: str = "output.wav"
):
    """Synthesize a single text."""
    async with websockets.connect(uri) as websocket:
        request = {
            "action": "synthesize",
            "text": text,
            "speed": speed,
            "total_step": total_step
        }
        if voice_style:
            request["voice_style"] = voice_style
        
        await websocket.send(json.dumps(request))
        response = json.loads(await websocket.recv())
        
        if "error" in response:
            print(f"Error: {response['error']}")
            return
        
        # Decode and save audio
        audio_data = base64.b64decode(response["audio"])
        with open(output_file, "wb") as f:
            f.write(audio_data)
        
        print(f"Audio saved to: {output_file}")
        print(f"Duration: {response['duration']:.2f}s")
        print(f"Sample rate: {response['sample_rate']} Hz")


async def synthesize_batch(
    uri: str,
    texts: list,
    speed: float = 1.05,
    total_step: int = 5,
    output_prefix: str = "output"
):
    """Synthesize multiple texts in batch."""
    async with websockets.connect(uri) as websocket:
        request = {
            "action": "batch_synthesize",
            "texts": texts,
            "speed": speed,
            "total_step": total_step
        }
        
        await websocket.send(json.dumps(request))
        response = json.loads(await websocket.recv())
        
        if "error" in response:
            print(f"Error: {response['error']}")
            return
        
        for i, result in enumerate(response["results"]):
            audio_data = base64.b64decode(result["audio"])
            output_file = f"{output_prefix}_{i+1}.wav"
            with open(output_file, "wb") as f:
                f.write(audio_data)
            print(f"Saved: {output_file} (duration: {result['duration']:.2f}s)")


async def get_server_info(uri: str):
    """Get server information."""
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"action": "info"}))
        response = json.loads(await websocket.recv())
        print("Server Info:")
        for key, value in response.items():
            print(f"  {key}: {value}")


async def ping_server(uri: str):
    """Ping the server."""
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"action": "ping"}))
        response = json.loads(await websocket.recv())
        print(f"Server response: {response['status']}")


def parse_args():
    parser = argparse.ArgumentParser(description="TTS WebSocket Client")
    
    parser.add_argument(
        "--server", type=str, default="ws://localhost:8764",
        help="WebSocket server URI"
    )
    parser.add_argument(
        "--text", type=str,
        help="Text to synthesize"
    )
    parser.add_argument(
        "--texts", type=str, nargs="+",
        help="Multiple texts for batch synthesis"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Use batch synthesis mode"
    )
    parser.add_argument(
        "--speed", type=float, default=1.05,
        help="Speech speed"
    )
    parser.add_argument(
        "--total-step", type=int, default=5,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--voice-style", type=str,
        help="Voice style file path"
    )
    parser.add_argument(
        "--output", type=str, default="output.wav",
        help="Output file name"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Get server info"
    )
    parser.add_argument(
        "--ping", action="store_true",
        help="Ping the server"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.ping:
        asyncio.run(ping_server(args.server))
    elif args.info:
        asyncio.run(get_server_info(args.server))
    elif args.batch and args.texts:
        asyncio.run(synthesize_batch(
            args.server,
            args.texts,
            args.speed,
            args.total_step,
            args.output.replace(".wav", "")
        ))
    elif args.text:
        asyncio.run(synthesize_single(
            args.server,
            args.text,
            args.speed,
            args.total_step,
            args.voice_style,
            args.output
        ))
    else:
        print("Please provide --text, --texts with --batch, --info, or --ping")