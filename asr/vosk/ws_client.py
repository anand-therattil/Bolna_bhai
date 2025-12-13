import asyncio
import websockets
import json
import soundfile as sf
import io


async def send_full_audio(uri, audio_path):
    async with websockets.connect(uri) as ws:
        audio, sr = sf.read(audio_path, dtype='int16')

        bio = io.BytesIO()
        sf.write(bio, audio, sr, format='WAV')

        print("Sending full audio file...")
        await ws.send(bio.getvalue())

        response = await ws.recv()
        print("Response:", response)


async def stream_audio(uri, audio_path, chunk_seconds=1.0):
    audio, sr = sf.read(audio_path, dtype='int16')
    chunk_size = int(sr * chunk_seconds)

    async with websockets.connect(uri) as ws:
        print("Streaming audio...")

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]

            bio = io.BytesIO()
            sf.write(bio, chunk, sr, format='WAV')

            await ws.send(bio.getvalue())

            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(msg)

                if data["status"] == "partial":
                    print("Partial:", data["result"]["partial"])
                elif data["status"] == "final":
                    print("Final:", data["result"]["text"])

            except asyncio.TimeoutError:
                pass


async def ping_server(uri):
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "ping"}))
        print("Ping →", await ws.recv())


if __name__ == "__main__":
    import sys
    uri = "ws://localhost:8765"

    if len(sys.argv) < 2:
        print("Usage: python client.py audio.wav [stream]")
        exit()

    audio = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "file"

    asyncio.run(ping_server(uri))

    if mode == "stream":
        asyncio.run(stream_audio(uri, audio))
    else:
        asyncio.run(send_full_audio(uri, audio))
