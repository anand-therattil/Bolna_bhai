import os
import yaml
from pathlib import Path
import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

# ------------------------------------------------------------
#  Locate project root (directory that contains /config folder)
# ------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]     # go up: vosk → asr → root

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# ------------------------------------------------------------
#  Load config safely
# ------------------------------------------------------------
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

VOSK_CONFIG = CONFIG["asr"]["vosk"]


try:
    import websockets
except ModuleNotFoundError:
    logger.error("You need to `pip install websockets` to use this STT service.")
    raise


class _AudioBuf:
    """Buffer audio between start/stop speaking frames."""
    def __init__(self):
        self.frames: List[AudioRawFrame] = []
        self.processing = False


class VoskSTTService(STTService):
    """Pipecat STT service that communicates with your Vosk WebSocket server."""

    def __init__(
        self,
        *,
        ws_url="ws://localhost:8765",
        sample_rate=16000,
        persistent_connection=True,
        **kwargs
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._ws_url = ws_url
        self._persist = persistent_connection

        self._websocket = None
        self._lock = asyncio.Lock()
        self._buf = _AudioBuf()

    # ------------------------
    # Connection Handling
    # ------------------------

    def _is_ws_open(self):
        return self._websocket and not self._websocket.closed

    async def _connect(self):
        if self._is_ws_open():
            return

        async with self._lock:
            if self._is_ws_open():
                return

            logger.info(f"Connecting to Vosk WebSocket server: {self._ws_url}")
            self._websocket = await websockets.connect(
                self._ws_url,
                max_size=None,
                ping_interval=20,
                ping_timeout=20,
            )
            logger.info("Connected to Vosk server.")

            # Send config (optional but supported by your server)
            await self._websocket.send(json.dumps({
                "type": "config",
                "config": {"sample_rate": self.sample_rate}
            }))

            # Ignore config response
            try:
                await asyncio.wait_for(self._websocket.recv(), timeout=1)
            except asyncio.TimeoutError:
                pass

    async def _disconnect(self):
        if self._websocket:
            try:
                await self._websocket.close()
            except:
                pass
        self._websocket = None

    # ------------------------
    # Pipecat lifecycle hooks
    # ------------------------

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._persist:
            await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._flush()
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    # ------------------------
    # Frame processing
    # ------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.info("User started speaking → buffer reset.")
            self._buf = _AudioBuf()

        elif isinstance(frame, AudioRawFrame):
            self._buf.frames.append(frame)

        elif isinstance(frame, UserStoppedSpeakingFrame):
            if not self._buf.processing and self._buf.frames:
                await self.process_generator(self._process_audio())
                return

        # Other frames simply pass through
        await self.push_frame(frame, direction)

    # ------------------------
    # Audio Processing
    # ------------------------

    async def _flush(self):
        if self._buf.frames and not self._buf.processing:
            await self.process_generator(self._process_audio())

    async def _process_audio(self) -> AsyncGenerator[Frame, None]:
        """Convert buffered frames → PCM bytes → send to server → return transcription."""
        self._buf.processing = True

        try:
            # Merge int16 PCM from frames
            pcm_chunks = []
            for f in self._buf.frames:
                pcm = np.frombuffer(f.audio, dtype=np.int16)
                pcm_chunks.append(pcm)

            if not pcm_chunks:
                yield ErrorFrame(error="No audio!")
                return

            pcm = np.concatenate(pcm_chunks)
            audio_bytes = pcm.tobytes()

            # Connect to server
            await self._connect()

            # Send final chunk (full audio for this utterance)
            await self._websocket.send(audio_bytes)

            # --------------------------
            # Wait for Vosk Final Result
            # --------------------------
            while True:
                raw = await self._websocket.recv()
                data = json.loads(raw)

                status = data.get("status")

                if status == "partial":
                    # optional: log partial
                    logger.debug(f"Partial: {data['result'].get('partial')}")
                    continue

                if status == "final":
                    text = data["result"].get("text", "").strip()
                    lang = Language.ENGLISH  # Vosk English model
                    yield TranscriptionFrame(
                        text=text,
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        language=lang,
                    )
                    logger.info(f"Final transcription → {text}")
                    break

                if status == "error":
                    yield ErrorFrame(error=data.get("error"))
                    break

        except Exception as e:
            logger.exception(e)
            yield ErrorFrame(error=str(e))

        finally:
            self._buf.processing = False
            self._buf.frames = []

            if not self._persist:
                await self._disconnect()
