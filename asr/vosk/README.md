# Vosk ASR (Automatic Speech Recognition)

This directory contains a complete Vosk-based speech recognition system with WebSocket server/client architecture and Pipecat integration for real-time speech-to-text processing.

## Overview

The Vosk ASR system provides:
- **WebSocket Server** ([`ws_server.py`](ws_server.py)) - Real-time speech recognition server
- **WebSocket Client** ([`ws_client.py`](ws_client.py)) - Test client for audio file processing
- **Pipecat Integration** ([`pipecat_client.py`](pipecat_client.py)) - STT service for Pipecat framework

## What's Included

### 1. WebSocket Server (`ws_server.py`)
- **Real-time ASR server** using Vosk models
- **Dual message handling**: Binary audio data and JSON control messages
- **Audio format support**: WAV files with configurable sample rates
- **Response types**: Partial and final transcription results
- **Error handling**: Sample rate validation and processing error management
- **Configuration**: Loads settings from [`config/config.yaml`](../config/config.yaml)

### 2. WebSocket Client (`ws_client.py`)
- **Full audio file processing**: Send complete audio files for transcription
- **Streaming mode**: Send audio in chunks for real-time processing
- **Server health check**: Ping functionality to test server connectivity
- **Flexible usage**: Command-line interface with streaming/file modes

### 3. Pipecat STT Service (`pipecat_client.py`)
- **Pipecat framework integration**: Complete STT service implementation
- **Frame-based processing**: Handles [`AudioRawFrame`](pipecat_client.py:14), [`UserStartedSpeakingFrame`](pipecat_client.py:21), [`UserStoppedSpeakingFrame`](pipecat_client.py:22)
- **Audio buffering**: Collects audio between speaking events
- **Connection management**: Persistent or per-utterance connections
- **Error handling**: Comprehensive error reporting via [`ErrorFrame`](pipecat_client.py:17)

## Requirements

### Dependencies
```bash
# Core Vosk dependencies
pip install vosk websockets

# Audio processing
pip install soundfile numpy

# Configuration management
pip install pyyaml

# Logging (for Pipecat client)
pip install loguru

# Pipecat framework (for pipecat_client.py)
pip install pipecat-ai
```

### Vosk Model
- **Required**: Download a Vosk model for your target language
- **Current config**: English-Indian model at `/Users/xxxxxxx/Bolna_Bhai/vosk/models/vosk-model-small-en-in-0.4`
- **Download from**: https://alphacephei.com/vosk/models
- **Update path**: Modify [`model_path`](../config/config.yaml:50) in config.yaml

## Configuration

The system uses [`config/config.yaml`](../config/config.yaml) for configuration:

```yaml
asr:
  vosk:
    ws_url: "ws://localhost:8765"           # WebSocket server URL
    model_path: "/path/to/vosk-model"       # Path to Vosk model directory
    sample_rate: 16000                      # Audio sample rate (Hz)
    persistent_connection: true             # Keep connections alive
```

## Usage

### 1. Start the WebSocket Server
```bash
cd vosk
python ws_server.py
```
Server starts on `ws://localhost:8765` (configurable in config.yaml)

### 2. Test with WebSocket Client

**Process full audio file:**
```bash
python ws_client.py audio.wav
```

**Stream audio in chunks:**
```bash
python ws_client.py audio.wav stream
```

**Test server connectivity:**
```bash
python ws_client.py audio.wav  # Includes automatic ping test
```

### 3. Use with Pipecat Framework

```python
from vosk.pipecat_client import VoskSTTService

# Create STT service
stt = VoskSTTService(
    ws_url="ws://localhost:8765",
    sample_rate=16000,
    persistent_connection=True
)

# Use in Pipecat pipeline
pipeline = Pipeline([
    # ... other processors
    stt,
    # ... other processors
])
```

## API Reference

### WebSocket Server Messages

**Audio Input (Binary):**
- Send raw audio bytes (WAV format)
- Must match configured sample rate (16000 Hz default)

**JSON Control Messages:**
```json
// Ping server
{"type": "ping"}

// Configure session
{"type": "config", "config": {"sample_rate": 16000}}
```

**Server Responses:**
```json
// Partial transcription
{"status": "partial", "result": {"partial": "hello wor"}}

// Final transcription
{"status": "final", "result": {"text": "hello world"}}

// Error response
{"status": "error", "error": "Sample rate mismatch"}

// Ping response
{"type": "pong"}
```

### Pipecat Integration

**Frame Types Handled:**
- [`AudioRawFrame`](pipecat_client.py:14) - Raw audio data
- [`UserStartedSpeakingFrame`](pipecat_client.py:21) - Speech detection start
- [`UserStoppedSpeakingFrame`](pipecat_client.py:22) - Speech detection end

**Output Frames:**
- [`TranscriptionFrame`](pipecat_client.py:20) - Final transcription results
- [`ErrorFrame`](pipecat_client.py:17) - Error conditions

## Architecture

```
┌─────────────────┐    WebSocket     ┌──────────────────┐
│   Audio Client  │ ────────────────▶│  Vosk WS Server  │
│                 │                  │                  │
│ • File upload   │◀────────────────│ • Model loading  │
│ • Streaming     │    JSON/Binary   │ • ASR processing │
│ • Ping/Config   │                  │ • Error handling │
└─────────────────┘                  └──────────────────┘
                                               │
                                               ▼
                                     ┌──────────────────┐
                                     │   Vosk Model     │
                                     │                  │
                                     │ • Speech-to-text │
                                     │ • Language model │
                                     │ • Acoustic model │
                                     └──────────────────┘
```

## Troubleshooting

### Common Issues

1. **Model not found**
   - Download Vosk model from https://alphacephei.com/vosk/models
   - Update [`model_path`](../config/config.yaml:50) in config.yaml

2. **Sample rate mismatch**
   - Ensure audio files match configured sample rate (16000 Hz)
   - Convert audio: `ffmpeg -i input.wav -ar 16000 output.wav`

3. **Connection refused**
   - Verify server is running on correct port
   - Check firewall settings
   - Confirm WebSocket URL in config

4. **Poor transcription quality**
   - Use appropriate language model
   - Ensure clean audio input
   - Check audio format and quality

### Performance Tips

- Use persistent connections for multiple requests
- Process audio in appropriate chunk sizes (1-2 seconds)
- Monitor server memory usage with large models
- Consider GPU acceleration for larger Vosk models

## Integration Notes

This Vosk ASR system is designed to work with:
- **Pipecat framework** for real-time voice applications
- **WebRTC communication layer** for browser-based audio
- **Multi-modal AI systems** requiring speech input
- **Voice assistants** and conversational AI applications

The modular design allows easy integration into larger voice processing pipelines while maintaining high performance and reliability.