import asyncio
import websockets
import json
import base64
import io
import soundfile as sf

async def generate_audio(text):
    uri = "ws://localhost:8000/generate"
    
    async with websockets.connect(uri) as websocket:
        # Send text
        await websocket.send(json.dumps({"text": text}))
        
        # Receive audio
        response = json.loads(await websocket.recv())
        audio_bytes = base64.b64decode(response["audio"])
        
        # Save audio
        with open("output.wav", "wb") as f:
            f.write(audio_bytes)
        
        print(f"Audio saved! Sampling rate: {response['sampling_rate']}")

# Run
asyncio.run(generate_audio("what is this ? How can something like this happpen today? I am not angry with this i am just sad  "))