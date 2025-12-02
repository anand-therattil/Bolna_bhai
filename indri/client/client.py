"""
WebSocket TTS Client for testing the TTS server.

Usage:
    python websocket_tts_client.py "Text to synthesize" --output output.wav
    python websocket_tts_client.py "Text to synthesize" --binary --output output.wav
"""

import asyncio
import json
import base64
import argparse
import logging
from pathlib import Path

import websockets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def synthesize_text(
    text: str,
    output_path: str,
    host: str = 'localhost',
    port: int = 8765,
    speaker: str = '[spkr_63]',
    binary_mode: bool = False
) -> bool:
    """
    Connect to TTS server and synthesize text.
    
    Args:
        text: Text to synthesize
        output_path: Path to save the output WAV file
        host: Server host
        port: Server port
        speaker: Speaker ID
        binary_mode: Whether server uses binary mode
        
    Returns:
        True if successful, False otherwise
    """
    uri = f"ws://{host}:{port}"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send request
            request = {
                'text': text,
                'speaker': speaker
            }
            
            logger.info(f"Sending request: {text[:50]}...")
            await websocket.send(json.dumps(request))
            
            # Receive response
            response = await websocket.recv()
            
            if binary_mode:
                # Binary mode: response is raw audio bytes
                if isinstance(response, bytes):
                    with open(output_path, 'wb') as f:
                        f.write(response)
                    logger.info(f"Saved audio to {output_path} ({len(response)} bytes)")
                    return True
                else:
                    # JSON error message
                    data = json.loads(response)
                    logger.error(f"Error: {data.get('error', 'Unknown error')}")
                    return False
            else:
                # JSON mode: response is JSON with base64-encoded audio
                data = json.loads(response)
                
                if data['status'] == 'success':
                    audio_bytes = base64.b64decode(data['audio'])
                    with open(output_path, 'wb') as f:
                        f.write(audio_bytes)
                    logger.info(f"Saved audio to {output_path} ({len(audio_bytes)} bytes)")
                    return True
                else:
                    logger.error(f"Error: {data.get('error', 'Unknown error')}")
                    return False
                    
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return False


async def interactive_mode(
    host: str = 'localhost',
    port: int = 8765,
    speaker: str = '[spkr_63]',
    binary_mode: bool = False,
    output_dir: str = '.'
):
    """
    Interactive mode for continuous TTS.
    
    Type text and press Enter to synthesize. Type 'quit' to exit.
    """
    uri = f"ws://{host}:{port}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    counter = 0
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info(f"Connected to {uri}")
            logger.info("Enter text to synthesize (or 'quit' to exit):")
            
            while True:
                try:
                    text = await asyncio.get_event_loop().run_in_executor(
                        None, input, "> "
                    )
                    
                    if text.lower() in ('quit', 'exit', 'q'):
                        break
                        
                    if not text.strip():
                        continue
                    
                    # Send request
                    request = {
                        'text': text,
                        'speaker': speaker
                    }
                    await websocket.send(json.dumps(request))
                    
                    # Receive response
                    response = await websocket.recv()
                    
                    output_path = output_dir / f"output_{counter:04d}.wav"
                    counter += 1
                    
                    if binary_mode:
                        if isinstance(response, bytes):
                            with open(output_path, 'wb') as f:
                                f.write(response)
                            logger.info(f"Saved: {output_path}")
                        else:
                            data = json.loads(response)
                            logger.error(f"Error: {data.get('error')}")
                    else:
                        data = json.loads(response)
                        if data['status'] == 'success':
                            audio_bytes = base64.b64decode(data['audio'])
                            with open(output_path, 'wb') as f:
                                f.write(audio_bytes)
                            logger.info(f"Saved: {output_path}")
                        else:
                            logger.error(f"Error: {data.get('error')}")
                            
                except EOFError:
                    break
                    
    except Exception as e:
        logger.error(f"Connection error: {e}")


def main():
    parser = argparse.ArgumentParser(description='WebSocket TTS Client')
    parser.add_argument('text', nargs='?', help='Text to synthesize')
    parser.add_argument('--output', '-o', default='output.wav', help='Output file path')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8760, help='Server port')
    parser.add_argument('--speaker', default='[spkr_63]', help='Speaker ID')
    parser.add_argument('--binary', action='store_true', help='Use binary mode')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--output-dir', default='.', help='Output directory for interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        asyncio.run(interactive_mode(
            host=args.host,
            port=args.port,
            speaker=args.speaker,
            binary_mode=args.binary,
            output_dir=args.output_dir
        ))
    elif args.text:
        success = asyncio.run(synthesize_text(
            text=args.text,
            output_path=args.output,
            host=args.host,
            port=args.port,
            speaker=args.speaker,
            binary_mode=args.binary
        ))
        exit(0 if success else 1)
    else:
        parser.print_help()
        exit(1)


if __name__ == '__main__':
    main()