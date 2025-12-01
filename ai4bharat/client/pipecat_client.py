"""Custom WebSocket speech-to-text service implementation for Pipecat."""

import asyncio
import base64
import json
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np
from loguru import logger
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

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

try:
    import websockets
    # Import State for connection state checking (websockets >= 14.0)
    try:
        from websockets.protocol import State as WebSocketState
    except ImportError:
        try:
            from websockets.legacy.protocol import State as WebSocketState
        except ImportError:
            WebSocketState = None
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("You need to `pip install websockets` to use this service.")
    raise Exception(f"Missing module: {e}")


class _AudioBuffer:
    """Buffer to accumulate audio frames while user is speaking."""
    
    def __init__(self):
        self.frames: List[AudioRawFrame] = []
        self.started_at: Optional[float] = None
        self.processing: bool = False


class Ai4BharatSTTService(STTService):
    """Custom WebSocket speech-to-text service for Pipecat.
    
    Connects to your ASR WebSocket server and provides real-time
    speech recognition compatible with the Pipecat framework.
    
    This implementation accumulates audio frames while the user is speaking
    and only sends to the server when the user stops speaking.
    """

    def __init__(
        self,
        *,
        ws_url: str = "ws://localhost:8765",
        language: str = "hi",
        sample_rate: int = 16000,
        persistent_connection: bool = True,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        **kwargs,
    ):
        """Initialize the Custom WebSocket STT service.

        Args:
            ws_url: WebSocket server URL.
            language: Language code for transcription (e.g., 'hi' for Hindi, 'en' for English).
            sample_rate: Audio sample rate in Hz.
            persistent_connection: Keep WebSocket connection alive between requests.
            ping_interval: Interval for WebSocket ping messages.
            ping_timeout: Timeout for WebSocket ping responses.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._ws_url = ws_url
        self._language = language
        self._persist = persistent_connection
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._lock = asyncio.Lock()
        self._buf = _AudioBuffer()
        self._inflight: Dict[str, asyncio.Event] = {}

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics generation.
        """
        return True

    def _is_websocket_open(self) -> bool:
        """Check if websocket connection is open.
        
        Compatible with different websockets library versions.
        
        Returns:
            True if websocket is connected and open, False otherwise.
        """
        if self._websocket is None:
            return False
        
        # For websockets >= 14.0, use state attribute
        if hasattr(self._websocket, 'state'):
            if WebSocketState is not None:
                return self._websocket.state == WebSocketState.OPEN
            # Fallback: state value 1 typically means OPEN
            return self._websocket.state.value == 1 if hasattr(self._websocket.state, 'value') else False
        
        # For older versions (websockets < 14.0), check closed attribute
        if hasattr(self._websocket, 'closed'):
            return not self._websocket.closed
        
        # For websockets >= 13.0, check open attribute
        if hasattr(self._websocket, 'open'):
            return self._websocket.open
        
        # Last resort fallback
        return self._websocket is not None

    async def set_language(self, language: str):
        """Set the recognition language and reconnect.

        Args:
            language: The language code to use for speech recognition.
        """
        logger.info(f"Switching STT language to: [{language}]")
        self._language = language
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        """Start the Custom WebSocket STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self._persist:
            await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Custom WebSocket STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._flush()
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Custom WebSocket STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._cancel_all()
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """This method is not used in this implementation.
        
        Audio processing is handled in process_frame instead.
        """
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with custom handling for speech detection.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            # User started speaking - initialize buffer
            logger.info("User started speaking - beginning audio accumulation")
            self._buf = _AudioBuffer()
            self._buf.started_at = time.time()

        elif isinstance(frame, AudioRawFrame) and self._buf.started_at is not None:
            # Accumulate audio frames while user is speaking
            self._buf.frames.append(frame)
            logger.trace(f"Accumulated audio frame, total frames: {len(self._buf.frames)}")

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # User stopped speaking - process accumulated audio
            if self._buf.frames and not self._buf.processing:
                logger.info(f"User stopped speaking - processing {len(self._buf.frames)} audio frames")
                await self.process_generator(self._process_audio_buffer())
                return

        # Pass through other frames
        if frame is not None:
            await self.push_frame(frame, direction)

    # ------------------ Internal Methods ------------------

    async def _connect(self):
        """Establish WebSocket connection to the ASR server."""
        if self._is_websocket_open():
            return
        
        async with self._lock:
            if self._is_websocket_open():
                return
            
            try:
                logger.debug(f"Connecting to WebSocket server at {self._ws_url}")
                self._websocket = await websockets.connect(
                    self._ws_url,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    max_size=None
                )
                logger.info(f"Connected to WebSocket server at {self._ws_url}")
                
                # Send initial config
                await self._send_config()
                
            except Exception as e:
                logger.error(f"Failed to connect to WebSocket server: {e}")
                self._websocket = None

    async def _disconnect(self):
        """Close WebSocket connection."""
        async with self._lock:
            if self._websocket is not None:
                try:
                    await self._websocket.close()
                except Exception as e:
                    logger.warning(f"Error closing WebSocket: {e}")
            self._websocket = None
            logger.info("Disconnected from WebSocket server")

    async def _ensure_connection(self):
        """Ensure WebSocket connection is established."""
        if not self._persist:
            await self._connect()
            return
        if not self._is_websocket_open():
            await self._connect()

    async def _send_json(self, payload: dict):
        """Send JSON payload over WebSocket."""
        await self._ensure_connection()
        if self._websocket is not None:
            await self._websocket.send(json.dumps(payload))

    async def _send_config(self):
        """Send configuration to the server."""
        try:
            config_message = {
                "type": "config",
                "config": {
                    "language": self._language,
                    "sample_rate": self.sample_rate
                }
            }
            if self._websocket:
                await self._websocket.send(json.dumps(config_message))
                logger.debug("Configuration sent to server")
        except Exception as e:
            logger.warning(f"Failed to send config: {e}")

    async def _flush(self):
        """Flush any remaining audio in the buffer."""
        if self._buf.frames and not self._buf.processing:
            await self.process_generator(self._process_audio_buffer())

    async def _cancel_all(self):
        """Cancel all in-flight requests."""
        for rid, evt in list(self._inflight.items()):
            try:
                await self._send_json({"type": "cancel", "request_id": rid})
            except Exception:
                pass
            finally:
                evt.set()
                self._inflight.pop(rid, None)

    async def _process_audio_buffer(self) -> AsyncGenerator[Frame, None]:
        """Process accumulated audio buffer and yield transcription frames.
        
        Yields:
            Frame: TranscriptionFrame with the transcribed text, or ErrorFrame on failure.
        """
        rid = None
        try:
            self._buf.processing = True

            # Gather audio as int16 PCM
            chunks = []
            for f in self._buf.frames:
                if isinstance(f.audio, bytes):
                    arr = np.frombuffer(f.audio, dtype=np.int16)
                elif isinstance(f.audio, np.ndarray):
                    arr = f.audio.astype(np.int16) if f.audio.dtype != np.int16 else f.audio
                else:
                    arr = None
                if arr is not None and arr.size:
                    chunks.append(arr)

            if not chunks:
                logger.warning("No valid audio frames to process")
                yield ErrorFrame(error="No valid audio")
                return

            # Concatenate all audio chunks
            audio_int16 = np.concatenate(chunks)
            audio_bytes = audio_int16.tobytes()
            
            logger.info(f"Processing {len(audio_bytes)} bytes of audio ({len(chunks)} chunks)")

            # Start metrics
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()

            # Generate request ID
            rid = f"stt-{uuid.uuid4()}"
            done = asyncio.Event()
            self._inflight[rid] = done

            # Ensure connection and send audio
            await self._ensure_connection()
            
            if self._websocket is None:
                yield ErrorFrame(error="WebSocket not connected")
                return

            # Send audio data directly as bytes
            await self._websocket.send(audio_bytes)

            # Wait for response
            try:
                raw = await asyncio.wait_for(
                    self._websocket.recv(),
                    timeout=30.0  # 30 second timeout for transcription
                )
                
                data = json.loads(raw)
                status = data.get("status")

                if status == "success":
                    await self.stop_ttfb_metrics()
                    
                    transcription = data.get("transcription", "").strip()
                    
                    if transcription:
                        # Determine language
                        language = None
                        try:
                            language = Language(self._language)
                        except ValueError:
                            pass
                        
                        # Yield transcription frame
                        yield TranscriptionFrame(
                            text=transcription,
                            user_id=self._user_id,
                            timestamp=time_now_iso8601(),
                            language=language,
                        )
                        logger.info(f"Transcription: {transcription}")
                    else:
                        logger.debug("Empty transcription received")
                    
                    await self.stop_processing_metrics()
                    done.set()

                elif status == "no_speech":
                    await self.stop_ttfb_metrics()
                    await self.stop_processing_metrics()
                    logger.debug("No speech detected in audio")
                    done.set()

                elif status == "error":
                    await self.stop_processing_metrics()
                    error_msg = data.get("error", "Unknown error")
                    logger.error(f"Server error: {error_msg}")
                    yield ErrorFrame(error=f"STT error: {error_msg}")
                    done.set()

                elif status == "config_received":
                    logger.debug("Configuration acknowledged by server")
                    # Wait for actual transcription response
                    raw = await asyncio.wait_for(
                        self._websocket.recv(),
                        timeout=30.0
                    )
                    # Process the transcription response
                    data = json.loads(raw)
                    if data.get("status") == "success":
                        await self.stop_ttfb_metrics()
                        transcription = data.get("transcription", "").strip()
                        if transcription:
                            language = None
                            try:
                                language = Language(self._language)
                            except ValueError:
                                pass
                            yield TranscriptionFrame(
                                text=transcription,
                                user_id=self._user_id,
                                timestamp=time_now_iso8601(),
                                language=language,
                            )
                            logger.info(f"Transcription: {transcription}")
                        await self.stop_processing_metrics()
                    done.set()

                else:
                    logger.warning(f"Unknown response status: {status}")
                    done.set()

            except asyncio.TimeoutError:
                await self.stop_processing_metrics()
                logger.warning("Timeout waiting for transcription response")
                yield ErrorFrame(error="Transcription timeout")
                done.set()

            # Close connection if non-persistent
            if not self._persist:
                await self._disconnect()

        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as e:
            logger.error(f"WebSocket connection closed: {e}")
            yield ErrorFrame(error="Connection closed")
        except Exception as e:
            logger.exception(f"Error processing audio buffer: {e}")
            yield ErrorFrame(error=f"Processing error: {e}")
        finally:
            # Clean up buffer
            self._buf.processing = False
            self._buf.frames = []
            self._buf.started_at = None
            if rid is not None:
                self._inflight.pop(rid, None)

    async def send_ping(self):
        """Send a ping message to keep the connection alive."""
        if self._is_websocket_open():
            try:
                ping_message = {"type": "ping"}
                await self._websocket.send(json.dumps(ping_message))
            except Exception as e:
                logger.warning(f"Failed to send ping: {e}")