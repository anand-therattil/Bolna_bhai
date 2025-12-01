# ai4bharat Indic Conformer — Pipecat Integration

**Short description**

This repository contains a lightweight WebSocket-based server and integration glue to run the **ai4bharat Indic Conformer** model for streaming ASR with Pipecat. It bundles four core modules:

* `model.py` — the main ai4bharat model wrapper (load, inference, preprocessing/postprocessing).
* `server.py` — a generic WebSocket server that accepts streaming audio, runs VAD and ASR, and replies with transcriptions.
* `server_pipecat.py` — Pipecat-focused server glue: message formats, session handling and any Pipecat-specific behavior.
* `vad.py` — Silero VAD helper used to detect speech segments and discard non-speech.

This README documents how to install, run, and maintain the project. It assumes you already have the ai4bharat model artifacts available (or know how to obtain them) and basic familiarity with Python, WebSockets and Docker.

---

## Quick start

1. Create a Python virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> If you don't have `requirements.txt`, add at least `torch`, `torchaudio`, `websockets`, `numpy`, `soundfile`, and `silero-vad` (or the Silero VAD dependency used in `vad.py`).

2. Prepare model artifacts (point `MODEL_PATH` to your ai4bharat model directory or file).

3. Start the server (example):

```bash
python server.py
```

Or run the Pipecat-facing server:

```bash
python server_pipecat.py
```

Open your WebSocket client and stream audio (PCM16, mono, the sample rate used by the model).

---

## Repository structure

```
ai4bharat
├─server
    ├─ model.py              # ai4bharat wrapper: load model, preprocess audio, decode logits -> text
    ├─ server.py             # generic WebSocket server for streaming ASR
    ├─ server_pipecat.py     # Pipecat-specific server (message formats/session handling)
    ├─ vad.py                # Silero VAD wrapper and helpers
    ├─ requirements.txt      # (recommended) python deps
    └─ README.md
├─client
    ├─ pipecat_client.py     # ai4bharat wrapper: to be place in the env for the pipecat/services/ai4bharat (if ai4bharat is not there then create a folder and rename it as stt.py)
    ├─ ws_client.py          # generic WebSocket client for ASR

```

---

## Requirements

Recommended software & hardware:

* Python 3.10+ (3.10/3.11 preferable)
* PyTorch compatible with your CUDA (or CPU-only for testing).
* `torchaudio` for audio handling.
* `websockets` or `aiohttp` for WebSocket server (this project uses `websockets` by default).
* Silero VAD (or the exact VAD package imported by `vad.py`).
* A GPU for production-grade throughput and low latency (NVIDIA with CUDA is recommended). CPU is fine for prototyping but expect higher latency.

---

## Usage

### Run the generic WebSocket server (`server.py`)

`server.py` is the straightforward WebSocket entrypoint. It expects clients to send raw audio frames (PCM16 mono) or base64-encoded audio payloads depending on how you implemented the message handling.

**Example command**

`python server.py`

### Run the Pipecat server (`server_pipecat.py`)

`server_pipecat.py` contains Pipecat-specific integration and message handling. It may perform the following extra tasks:
`python server.py`

**Notes**

* If Pipecat expects HTTP callbacks or a specific handshake, `server_pipecat.py` implements that glue — check the top of the file for expected request/response examples.
* Treat `server_pipecat.py` as the production entrypoint when integrating with Pipecat; use `server.py` for testing or raw WebSocket clients.

---
