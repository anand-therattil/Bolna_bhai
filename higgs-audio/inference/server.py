# server_simple.py
import asyncio
import json
import torch
import torchaudio
import base64
import io
from websockets.server import serve
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message

# Paths
AUDIO_TOKENIZER_PATH = "/home/user/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-tokenizer/snapshots/9d4988fbd4ad07b4cac3a5fa462741a41810dbec"
MODEL_PATH = "/home/user/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-generation-3B-base/snapshots/10840182ca4ad5d9d9113b60b9bb3c1ef1ba3f84"

# Initialize
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing Higgs Audio on {device}...")
serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)
print("✓ Higgs Audio initialized")

system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)

async def handle_client(websocket):
    print(f"Client connected from {websocket.remote_address}")
    try:
        async for message in websocket:
            print(f"Received message: {message[:100]}...")
            
            # Parse incoming message
            data = json.loads(message)
            text_input = data.get("text", "")
            
            print(f"Text input: {text_input}")
            
            # Prepare messages
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=text_input)
            ]
            
            print("Generating audio...")
            # Generate audio
            output: HiggsAudioResponse = serve_engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.95,
                top_k=50,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )
            
            print(f"✓ Generated {len(output.audio)} samples at {output.sampling_rate}Hz")
            
            # Convert audio to base64
            audio_tensor = torch.from_numpy(output.audio)[None, :]
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_tensor, output.sampling_rate, format="wav")
            audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Send back audio
            response = json.dumps({
                "audio": audio_base64,
                "sampling_rate": output.sampling_rate
            })
            
            print(f"Sending response ({len(response)} bytes)")
            await websocket.send(response)
            print("✓ Response sent")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Client disconnected")

async def main():
    print("=" * 60)
    print("WebSocket server running on ws://0.0.0.0:8000")
    print("=" * 60)
    async with serve(handle_client, "0.0.0.0", 8000):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())