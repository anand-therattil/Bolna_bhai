# Bolna Bhai

WebSocket servers for plugging open-source ASR, LLM and TTS models into [Pipecat](https://github.com/pipecat-ai/pipecat), with a focus on Indian languages and accents.

Each model runs as its own WebSocket server. A small Pipecat client sits in front of it and turns it into a pipeline service, so swapping one model for another is mostly a URL change in the config.

This is a working repo, not a packaged library. Some pieces are better tested than others.

## What's here

| Stage | Folder | Model |
|---|---|---|
| ASR | `ai4bharat/` | AI4Bharat Indic Conformer, with Silero VAD |
| ASR | `asr/vosk/` | Vosk (en-us and en-in models included) |
| LLM | `qwen_model/` | Qwen2.5-7B-Instruct served with vLLM |
| LLM | `gpt_model/`, `training/gpt2/` | GPT-2 server, plus fine-tuning and inference scripts |
| TTS | `indic_parler_tts/` | Indic Parler-TTS |
| TTS | `indri/` | Indri |
| TTS | `supertonic/` | Supertonic (ONNX) |
| TTS | `higgs-audio/` | Higgs Audio v2 (upstream training repo, included as is) |
| Transport | `communication_layer/smallwebrtc/` | SmallWebRTC |

## Config

`config/config.yaml` holds the server URLs and ports, the Qwen model name and vLLM settings, and the system prompt.

The Vosk server and the GPT-2 server both listen on port 8763 by default, so change one of them if you run both.

## Running

Every server is a plain Python script. Install the requirements for the pieces you want, start those servers, then start the SmallWebRTC server, which runs the bot.

```bash
# ASR
pip install -r ai4bharat/server/requirements.txt
python ai4bharat/server/server_pipecat.py       # or server.py for a plain WebSocket server

pip install -r asr/vosk/requirements.txt
python asr/vosk/ws_server.py

# LLM
python qwen_model/ws_server.py
python gpt_model/ws_server.py

# TTS
python indri/server/server.py --port 8760 --device cuda
python indic_parler_tts/server/server.py --port 8763
python supertonic/server/py/server.py           # defaults to port 8764

# Transport and bot
pip install -r communication_layer/smallwebrtc/requirement.txt
python communication_layer/smallwebrtc/server.py
```

Each folder with a `client.py` or `pipecat_client.py` also has a way to test that server on its own. The Vosk model path in `config/config.yaml` needs to point at a model on your machine.

## Next

Not done yet:

- LLM: gpt-oss-20b, Llama, Mistral
- ASR: NVIDIA Canary-Qwen
- TTS: Orpheus, NeuTTS, AI4Bharat IndicF5
- LLM + ASR in one model: Ultravox, Voxtral
- Transport: FastAPI WebSocket, SmallWebRTCTransport, a plain WebSocket server

Built on by [MANN](https://github.com/anand-therattil/MANN), a mental-health triage voice agent.
