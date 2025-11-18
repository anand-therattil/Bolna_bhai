import asyncio
import json
import torch
import torchaudio
import base64
import io
import websockets
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message

# Paths
AUDIO_TOKENIZER_PATH = "/home/user/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-tokenizer/snapshots/9d4988fbd4ad07b4cac3a5fa462741a41810dbec"
MODEL_PATH = "/home/user/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-generation-3B-base/snapshots/10840182ca4ad5d9d9113b60b9bb3c1ef1ba3f84"

# Initialize
device = "cuda" if torch.cuda.is_available() else "cpu"
serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

system_prompt = (
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
)

async def handle_client(websocket):
    print("Client connected")
    try:
        async for message in websocket:
            # Parse incoming message
            data = json.loads(message)
            text_input = data.get("text", "")
            
            # Prepare messages
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=text_input)
            ]
            
            # Generate audio
            output: HiggsAudioResponse = serve_engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.95,
                top_k=50,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )
            
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
            await websocket.send(response)
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handle_client, "0.0.0.0", 8000):
        print("WebSocket server running on ws://0.0.0.0:8000")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())