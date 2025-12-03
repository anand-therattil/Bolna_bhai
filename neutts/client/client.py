import asyncio
import json
import base64
from pathlib import Path

import websockets


class TTSWebSocketClient:
    def __init__(self, uri="ws://localhost:8765"):
        self.uri = uri
        self.websocket = None
    
    async def connect(self):
        """Connect to the WebSocket server."""
        self.websocket = await websockets.connect(self.uri)
        print(f"Connected to {self.uri}")
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            print("Disconnected")
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def send_request(self, data: dict) -> dict:
        """Send a request and wait for response."""
        await self.websocket.send(json.dumps(data))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def ping(self) -> dict:
        """Ping the server."""
        return await self.send_request({"action": "ping"})
    
    async def synthesize(self, text: str) -> dict:
        """Synthesize speech from text."""
        return await self.send_request({
            "action": "synthesize",
            "text": text
        })
    
    async def synthesize_and_save(self, text: str, output_path: str) -> bool:
        """Synthesize speech and save to file."""
        response = await self.synthesize(text)
        
        if response["status"] == "success":
            audio_bytes = base64.b64decode(response["audio"])
            Path(output_path).write_bytes(audio_bytes)
            print(f"Audio saved to {output_path}")
            print(f"Duration: {response['duration']:.2f}s")
            return True
        else:
            print(f"Error: {response.get('message', 'Unknown error')}")
            return False


async def main():
    """Example usage of the TTS WebSocket client."""
    async with TTSWebSocketClient(uri="ws://localhost:8764") as client:
        # Ping the server
        response = await client.ping()
        print(f"Ping response: {response}")
        
        # Synthesize speech - just send text!
        print("\nSynthesizing speech...")
        await client.synthesize_and_save(
            text="My name is Dave, and um, I'm from London.",
            output_path="test_output.wav"
        )


if __name__ == "__main__":
    asyncio.run(main())