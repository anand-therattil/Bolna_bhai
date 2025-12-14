import asyncio
import json
import io
import soundfile as sf
import websockets
from typing import AsyncGenerator

from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TranscriptionFrame,
    ErrorFrame,
    UserStoppedSpeakingFrame,
    UserStartedSpeakingFrame
)
from pipecat.services.stt_service import STTService
from pipecat.processors.frame_processor import FrameDirection

from loguru import logger


class VoskSTTService(STTService):
    """
    Pipecat STT service that connects to a Vosk WebSocket server.
    """

    def __init__(
        self,
        *,
        server_url: str = "ws://localhost:8766",
        sample_rate: int = 16000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._server_url = server_url
        self._sample_rate = sample_rate
        self._websocket = None
        self._connection_lock = asyncio.Lock()
        self._audio_buffer = bytearray()
        self._is_speaking = False  # Track if user is currently speaking

    async def start(self, frame: Frame):
        """Initialize WebSocket connection when service starts."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: Frame):
        """Close WebSocket connection when service stops."""
        await self._disconnect()
        await super().stop(frame)

    async def _connect(self):
        """Establish WebSocket connection to Vosk server."""
        async with self._connection_lock:
            if self._websocket is None:
                try:
                    logger.info(f"Connecting to Vosk server at {self._server_url}")
                    self._websocket = await websockets.connect(self._server_url)
                    logger.info("Connected to Vosk server")
                    
                    # Send ping to verify connection
                    await self._websocket.send(json.dumps({"type": "ping"}))
                    response = await self._websocket.recv()
                    logger.debug(f"Ping response: {response}")
                    
                except Exception as e:
                    logger.error(f"Failed to connect to Vosk server: {e}")
                    self._websocket = None
                    raise

    async def _disconnect(self):
        """Close WebSocket connection."""
        async with self._connection_lock:
            if self._websocket:
                try:
                    await self._websocket.close()
                    logger.info("Disconnected from Vosk server")
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
                finally:
                    self._websocket = None

    async def _ensure_connected(self):
        """Ensure WebSocket is connected, reconnect if necessary."""
        if self._websocket is None:
            await self._connect()
        else:
            # Check if connection is still alive by trying to ping
            try:
                # Try to send a ping to verify connection
                if hasattr(self._websocket, 'closed'):
                    if self._websocket.closed:
                        await self._connect()
                # For older websockets versions, just check if it's None
            except Exception:
                # If any error, reconnect
                await self._connect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Required abstract method from STTService.
        Note: This is not used in our implementation as we use event-driven processing
        based on UserStartedSpeaking/UserStoppedSpeaking frames.
        """
        # This method is required by the abstract base class but not used
        # in our VAD-based implementation
        yield
        return
    
    async def _process_audio_buffer(self) -> AsyncGenerator[Frame, None]:
        """
        Process the accumulated audio buffer and send to Vosk for transcription.
        Only called when user stops speaking.
        """
        if len(self._audio_buffer) == 0:
            logger.debug("Audio buffer empty, skipping transcription")
            return
            
        try:
            await self._ensure_connected()
            
            # Get all buffered audio
            chunk = bytes(self._audio_buffer)
            
            # Clear buffer for next utterance
            self._audio_buffer = bytearray()
            
            logger.debug(f"Processing {len(chunk)} bytes of audio for transcription")
            
            # Convert raw PCM to WAV format
            audio_io = io.BytesIO()
            
            # Convert bytes to int16 array
            import numpy as np
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            
            # Write as WAV file to BytesIO
            sf.write(
                audio_io,
                audio_array,
                self._sample_rate,
                format='WAV',
                subtype='PCM_16'
            )
            
            # Get bytes and send to server
            audio_bytes = audio_io.getvalue()
            
            try:
                await self._websocket.send(audio_bytes)
                
                # Receive response
                response = await self._websocket.recv()
                result = json.loads(response)
                
                # Process result
                if result.get("status") == "final":
                    text = result.get("result", {}).get("text", "")
                    if text.strip():
                        logger.info(f"Final transcription: {text}")
                        yield TranscriptionFrame(text=text, user_id="user", timestamp=0)
                    else:
                        logger.debug("Empty transcription result")
                        
                elif result.get("status") == "partial":
                    text = result.get("result", {}).get("partial", "")
                    if text.strip():
                        logger.debug(f"Partial transcription: {text}")
                        # For partial results on stop speaking, treat as final
                        yield TranscriptionFrame(text=text, user_id="user", timestamp=0)
                        
                elif result.get("status") == "error":
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Vosk server error: {error_msg}")
                    yield ErrorFrame(error=error_msg)
                    
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.ConnectionClosedError,
                    websockets.exceptions.ConnectionClosedOK) as e:
                logger.error(f"WebSocket connection closed: {e}")
                self._websocket = None
                yield ErrorFrame(error="Connection to STT server lost")
                
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}")
            yield ErrorFrame(error=str(e))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames."""
        await super().process_frame(frame, direction)

        # Handle UserStartedSpeaking - start buffering
        if isinstance(frame, UserStartedSpeakingFrame):
            logger.debug("User started speaking - starting audio buffer")
            self._is_speaking = True
            self._audio_buffer = bytearray()  # Clear any old buffer
            await self.push_frame(frame, direction)
            
        # Handle UserStoppedSpeaking - process buffered audio
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("User stopped speaking - processing buffered audio")
            self._is_speaking = False
            
            # Process the buffered audio
            async for result_frame in self._process_audio_buffer():
                await self.push_frame(result_frame)
            
            # Pass through the stopped speaking frame
            await self.push_frame(frame, direction)
            
        # Handle audio frames - buffer while speaking
        elif isinstance(frame, AudioRawFrame):
            if self._is_speaking:
                # Ensure audio is in correct format (16-bit mono)
                if frame.sample_rate != self._sample_rate:
                    logger.warning(
                        f"Audio sample rate mismatch: expected {self._sample_rate}, "
                        f"got {frame.sample_rate}"
                    )
                    # You might want to resample here
                
                # Buffer the audio while user is speaking
                self._audio_buffer.extend(frame.audio)
                logger.debug(f"Buffering audio: {len(frame.audio)} bytes (total: {len(self._audio_buffer)} bytes)")
            # Don't pass through audio frames to avoid duplicate processing
            
        # Pass through other frames
        else:
            await self.push_frame(frame, direction)

