"""
Sample WebSocket client for Indic Parler TTS Server

This demonstrates how to connect to the TTS WebSocket server
and receive generated audio.
"""

import asyncio
import json
import base64
import websockets


async def text_to_speech(
    prompt: str,
    description: str = None,
    server_url: str = "ws://localhost:8765",
    output_file: str = "output.wav"
):
    """
    Send a text prompt to the TTS server and save the audio response.
    
    Args:
        prompt: Text to synthesize
        description: Optional voice description
        server_url: WebSocket server URL
        output_file: Output filename for the audio
    """
    async with websockets.connect(server_url) as websocket:
        # Prepare request
        request = {"prompt": prompt}
        if description:
            request["description"] = description
        
        # Send request
        await websocket.send(json.dumps(request))
        print(f"Sent request: {prompt[:50]}...")
        
        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        
        if data["status"] == "success":
            # Decode and save audio
            audio_bytes = base64.b64decode(data["audio"])
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
            print(f"Audio saved to {output_file}")
            print(f"Sample rate: {data['sample_rate']} Hz")
        else:
            print(f"Error: {data['message']}")


async def interactive_session(server_url: str = "ws://localhost:8765"):
    """
    Interactive session - type text and get audio responses.
    """
    async with websockets.connect(server_url) as websocket:
        print("Connected to TTS server. Type text to synthesize (Ctrl+C to exit):")
        
        count = 0
        while True:
            try:
                prompt = input("\nEnter text: ").strip()
                if not prompt:
                    continue
                
                # Send request
                await websocket.send(json.dumps({"prompt": prompt}))
                
                # Receive response
                response = await websocket.recv()
                data = json.loads(response)
                
                if data["status"] == "success":
                    count += 1
                    filename = f"output_{count}.wav"
                    audio_bytes = base64.b64decode(data["audio"])
                    with open(filename, "wb") as f:
                        f.write(audio_bytes)
                    print(f"Audio saved to {filename}")
                else:
                    print(f"Error: {data['message']}")
                    
            except KeyboardInterrupt:
                print("\nDisconnecting...")
                break


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TTS WebSocket Client")
    parser.add_argument("--server", default="ws://localhost:8763", help="Server URL")
    parser.add_argument("--text", help="Text to synthesize (omit for interactive mode)")
    parser.add_argument("--output", default="output.wav", help="Output filename")
    parser.add_argument("--description", help="Voice description")
    
    args = parser.parse_args()
    
    if args.text:
        # Single request mode
        asyncio.run(text_to_speech(
            prompt=args.text,
            description=args.description,
            server_url=args.server,
            output_file=args.output
        ))
    else:
        # Interactive mode
        asyncio.run(interactive_session(args.server))
        