# Higgs Audio Inference

WebSocket-based audio generation using Higgs Audio v2 models.

## 📁 Files

- **`server.py`** - WebSocket server for audio generation
- **`websocket_client.py`** - Simple WebSocket client
- **`pipecat_client.py`** - Advanced client using Pipecat framework

## 🚀 Quick Start

### 1. Start the Server

```bash
python server.py
```

The server will run on `ws://localhost:8000` and initialize the Higgs Audio model.

### 2. Generate Audio

**Option A: Simple Client**
```bash
python websocket_client.py
```

**Option B: Pipecat Client** (with pipeline processing)
```bash
python pipecat_client.py
```

Both clients will:
- Connect to the server
- Send text input
- Receive generated audio
- Save to `output.wav`


## 🔧 Configuration

### Server Settings
- **Host**: `0.0.0.0`
- **Port**: `8000`
- **Device**: Auto-detects CUDA/CPU
- **Model**: Higgs Audio v2 Generation 3B Base
- **Tokenizer**: Higgs Audio v2 Tokenizer

### Generation Parameters
- `max_new_tokens`: 1024
- `temperature`: 0.3
- `top_p`: 0.95
- `top_k`: 50

## 📦 Requirements

```bash
# Core dependencies
torch
torchaudio
websockets
soundfile

# For Pipecat client
pipecat-ai
numpy
```

## 🎯 When to Use Which Client?

- **`websocket_client.py`**: Simple, straightforward audio generation
- **`pipecat_client.py`**: When you need pipeline processing, frame handling, or integration with Pipecat workflows

## 📄 API Format

### Request
```json
{
  "text": "Your input text here"
}
```

### Response
```json
{
  "audio": "base64_encoded_wav_file",
  "sampling_rate": 24000
}
```

## 💡 Notes

- Server must be running before starting clients
- Audio is returned as complete WAV files (base64 encoded)
- Default sampling rate: 24kHz
- System prompt configures quiet room acoustics