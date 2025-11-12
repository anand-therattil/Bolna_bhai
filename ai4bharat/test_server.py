import asyncio
import websockets
import json
import soundfile as sf


async def send_audio_file(uri, audio_path):
    """
    Send an audio file to the WebSocket server
    
    Args:
        uri: WebSocket server URI (e.g., "ws://localhost:8765")
        audio_path: Path to the audio file
    """
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        
        # Read audio file
        audio_array, sr = sf.read(audio_path, dtype='float32')
        
        # Convert to bytes (WAV format)
        import io
        audio_io = io.BytesIO()
        sf.write(audio_io, audio_array, sr, format='WAV')
        audio_bytes = audio_io.getvalue()
        
        print(f"Sending audio file: {audio_path}")
        
        # Send audio data
        await websocket.send(audio_bytes)
        
        # Wait for response
        response = await websocket.recv()
        result = json.loads(response)
        
        print("\n=== Transcription Result ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return result


async def stream_audio_chunks(uri, audio_path, chunk_duration=1.0):
    """
    Stream audio in chunks to simulate real-time processing
    
    Args:
        uri: WebSocket server URI
        audio_path: Path to the audio file
        chunk_duration: Duration of each chunk in seconds
    """
    async with websockets.connect(uri) as websocket:
        print(f"Connected to {uri}")
        
        # Read audio file
        audio_array, sr = sf.read(audio_path, dtype='float32')
        
        # Calculate chunk size
        chunk_size = int(sr * chunk_duration)
        
        # Send audio in chunks
        for i in range(0, len(audio_array), chunk_size):
            chunk = audio_array[i:i + chunk_size]
            
            # Convert chunk to bytes
            import io
            audio_io = io.BytesIO()
            sf.write(audio_io, chunk, sr, format='WAV')
            chunk_bytes = audio_io.getvalue()
            
            print(f"Sending chunk {i//chunk_size + 1}...")
            await websocket.send(chunk_bytes)
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                result = json.loads(response)
                
                if result.get("status") == "success":
                    print(f"Transcription: {result.get('transcription')}")
                elif result.get("status") == "no_speech":
                    print("No speech detected in this chunk")
                    
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
            
            # Small delay between chunks
            await asyncio.sleep(0.1)


async def test_ping(uri):
    """Test server connectivity with ping"""
    async with websockets.connect(uri) as websocket:
        print(f"Testing connection to {uri}")
        
        # Send ping
        await websocket.send(json.dumps({"type": "ping"}))
        
        # Wait for pong
        response = await websocket.recv()
        result = json.loads(response)
        
        if result.get("type") == "pong":
            print("✓ Server is responding")
        
        return result


if __name__ == "__main__":
    import sys
    
    # Configuration
    SERVER_URI = "ws://localhost:8765"
    
    if len(sys.argv) < 2:
        print("Usage: python websocket_asr_client.py <audio_file.wav> [mode]")
        print("Modes: 'file' (default) or 'stream'")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "file"
    
    # Test connection first
    print("Testing server connection...")
    asyncio.run(test_ping(SERVER_URI))
    
    print(f"\n--- Processing in {mode} mode ---\n")
    
    if mode == "stream":
        asyncio.run(stream_audio_chunks(SERVER_URI, audio_file))
    else:
        asyncio.run(send_audio_file(SERVER_URI, audio_file))