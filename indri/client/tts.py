"""
Pipecat TTS service implementation for Indri WebSocket TTS server.

This module provides integration with a custom Indri TTS WebSocket server
for text-to-speech synthesis in Pipecat pipelines.

Usage:
    from indri_tts_service import IndriTTSService
    
    tts = IndriTTSService(
        base_url="ws://localhost:8765",
        voice="[spkr_63]",
        sample_rate=24000
    )
"""

import asyncio
import base64
import json
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

try:
    import websockets
    from websockets.asyncio.client import connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use IndriTTSService, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")


class IndriTTSService(TTSService):
    """Indri WebSocket TTS service for Pipecat.
    
    Provides integration with a custom Indri TTS WebSocket server for
    text-to-speech synthesis. Supports configurable voice/speaker IDs
    and sample rates.
    
    Args:
        base_url: WebSocket URL for the Indri TTS server (e.g., "ws://localhost:8765")
        voice: Speaker ID to use (e.g., "[spkr_63]")
        sample_rate: Audio sample rate in Hz (default: 24000)
        binary_mode: Whether server uses binary mode (default: False, uses JSON/base64)
        **kwargs: Additional arguments passed to parent TTSService
    
    Example:
        ```python
        tts = IndriTTSService(
            base_url="ws://localhost:8765",
            voice="[spkr_63]",
            sample_rate=24000
        )
        ```
    """
    
    def __init__(
        self,
        *,
        base_url: str = "ws://localhost:8765",
        voice: str = "[spkr_63]",
        sample_rate: int = 24000,
        binary_mode: bool = False,
        **kwargs
    ):
        """Initialize the Indri TTS service.
        
        Args:
            base_url: WebSocket URL for the Indri TTS server
            voice: Speaker ID to use for synthesis
            sample_rate: Output audio sample rate in Hz
            binary_mode: Whether server sends binary audio (vs base64 JSON)
            **kwargs: Additional arguments passed to TTSService
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        # Remove trailing slash if present
        if base_url.endswith("/"):
            logger.warning("Base URL ends with a slash, removing it.")
            base_url = base_url[:-1]
            
        self._base_url = base_url
        self._binary_mode = binary_mode
        
        # Store settings for potential updates
        self._settings = {
            "base_url": base_url,
            "voice": voice,
            "sample_rate": sample_rate,
        }
        
    @property
    def voice(self) -> str:
        """Get the current voice/speaker ID."""
        return self._settings["voice"]
    
    @voice.setter
    def voice(self, value: str):
        """Set the voice/speaker ID."""
        self._settings["voice"] = value
        
    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.
        
        Returns:
            True, as this service supports metrics generation.
        """
        return True
    
    def _strip_wav_header(self, audio_data: bytes) -> bytes:
        """Strip WAV header from audio data to get raw PCM.
        
        The WAV header is typically 44 bytes for standard PCM format.
        
        Args:
            audio_data: WAV audio data with header
            
        Returns:
            Raw PCM audio data without header
        """
        # Standard WAV header is 44 bytes
        # Check for RIFF header
        if len(audio_data) > 44 and audio_data[:4] == b'RIFF':
            return audio_data[44:]
        return audio_data
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using the Indri WebSocket server.
        
        Args:
            text: The text to convert to speech
            
        Yields:
            Frame: TTSStartedFrame, TTSAudioRawFrame(s), TTSStoppedFrame,
                   or ErrorFrame on failure
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        
        try:
            await self.start_ttfb_metrics()
            
            # Connect to WebSocket server
            async with connect(self._base_url) as websocket:
                # Send TTS request
                request = {
                    "text": text,
                    "speaker": self._settings["voice"]
                }
                await websocket.send(json.dumps(request))
                
                # Receive response
                response = await websocket.recv()
                
                await self.stop_ttfb_metrics()
                
                if self._binary_mode:
                    # Binary mode: response is raw WAV bytes
                    if isinstance(response, bytes):
                        audio_data = self._strip_wav_header(response)
                    else:
                        # JSON error message
                        data = json.loads(response)
                        error_msg = data.get('error', 'Unknown error from TTS server')
                        logger.error(f"{self} error: {error_msg}")
                        yield ErrorFrame(error=error_msg)
                        return
                else:
                    # JSON mode: response is JSON with base64-encoded audio
                    data = json.loads(response)
                    
                    if data.get('status') != 'success':
                        error_msg = data.get('error', 'Unknown error from TTS server')
                        logger.error(f"{self} error: {error_msg}")
                        yield ErrorFrame(error=error_msg)
                        return
                    
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(data['audio'])
                    audio_data = self._strip_wav_header(audio_bytes)
                
                # Start TTS usage metrics
                await self.start_tts_usage_metrics(text)
                
                yield TTSStartedFrame()
                
                # Yield audio frame with raw PCM data
                # The audio is mono (1 channel) at the configured sample rate
                yield TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=self._settings["sample_rate"],
                    num_channels=1
                )
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"{self} WebSocket connection closed: {e}")
            yield ErrorFrame(error=f"WebSocket connection closed: {e}")
            
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"{self} WebSocket error: {e}")
            yield ErrorFrame(error=f"WebSocket error: {e}")
            
        except json.JSONDecodeError as e:
            logger.error(f"{self} Invalid JSON response: {e}")
            yield ErrorFrame(error=f"Invalid JSON response from TTS server: {e}")
            
        except Exception as e:
            logger.error(f"{self} error generating TTS: {e}")
            yield ErrorFrame(error=str(e))
            
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()


class IndriTTSServicePersistent(TTSService):
    """Indri WebSocket TTS service with persistent connection.
    
    This version maintains a persistent WebSocket connection for better
    performance when making multiple TTS requests. The connection is
    established on start() and closed on stop().
    
    Args:
        base_url: WebSocket URL for the Indri TTS server
        voice: Speaker ID to use (e.g., "[spkr_63]")
        sample_rate: Audio sample rate in Hz (default: 24000)
        binary_mode: Whether server uses binary mode (default: False)
        reconnect_on_error: Whether to reconnect on connection errors (default: True)
        **kwargs: Additional arguments passed to parent TTSService
    """
    
    def __init__(
        self,
        *,
        base_url: str = "ws://localhost:8765",
        voice: str = "[spkr_63]",
        sample_rate: int = 24000,
        binary_mode: bool = False,
        reconnect_on_error: bool = True,
        **kwargs
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        if base_url.endswith("/"):
            base_url = base_url[:-1]
            
        self._base_url = base_url
        self._binary_mode = binary_mode
        self._reconnect_on_error = reconnect_on_error
        self._websocket = None
        self._connected = False
        
        self._settings = {
            "base_url": base_url,
            "voice": voice,
            "sample_rate": sample_rate,
        }
        
    @property
    def voice(self) -> str:
        return self._settings["voice"]
    
    @voice.setter
    def voice(self, value: str):
        self._settings["voice"] = value
        
    def can_generate_metrics(self) -> bool:
        return True
    
    async def start(self, frame: StartFrame):
        """Start the service and establish WebSocket connection."""
        await super().start(frame)
        await self._connect()
        
    async def stop(self, frame: EndFrame):
        """Stop the service and close WebSocket connection."""
        await super().stop(frame)
        await self._disconnect()
        
    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close WebSocket connection."""
        await super().cancel(frame)
        await self._disconnect()
        
    async def _connect(self):
        """Establish WebSocket connection."""
        try:
            self._websocket = await connect(self._base_url)
            self._connected = True
            logger.info(f"{self}: Connected to Indri TTS server at {self._base_url}")
        except Exception as e:
            logger.error(f"{self}: Failed to connect to TTS server: {e}")
            self._connected = False
            raise
            
    async def _disconnect(self):
        """Close WebSocket connection."""
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"{self}: Error closing WebSocket: {e}")
            finally:
                self._websocket = None
                self._connected = False
                
    async def _ensure_connected(self):
        """Ensure WebSocket is connected, reconnect if needed."""
        if not self._connected or self._websocket is None:
            if self._reconnect_on_error:
                await self._connect()
            else:
                raise Exception("WebSocket not connected")
    
    def _strip_wav_header(self, audio_data: bytes) -> bytes:
        if len(audio_data) > 44 and audio_data[:4] == b'RIFF':
            return audio_data[44:]
        return audio_data
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using persistent WebSocket connection."""
        logger.debug(f"{self}: Generating TTS [{text}]")
        
        try:
            await self._ensure_connected()
            await self.start_ttfb_metrics()
            
            # Send TTS request
            request = {
                "text": text,
                "speaker": self._settings["voice"]
            }
            await self._websocket.send(json.dumps(request))
            
            # Receive response
            response = await self._websocket.recv()
            
            await self.stop_ttfb_metrics()
            
            if self._binary_mode:
                if isinstance(response, bytes):
                    audio_data = self._strip_wav_header(response)
                else:
                    data = json.loads(response)
                    error_msg = data.get('error', 'Unknown error')
                    logger.error(f"{self} error: {error_msg}")
                    yield ErrorFrame(error=error_msg)
                    return
            else:
                data = json.loads(response)
                
                if data.get('status') != 'success':
                    error_msg = data.get('error', 'Unknown error')
                    logger.error(f"{self} error: {error_msg}")
                    yield ErrorFrame(error=error_msg)
                    return
                
                audio_bytes = base64.b64decode(data['audio'])
                audio_data = self._strip_wav_header(audio_bytes)
            
            await self.start_tts_usage_metrics(text)
            
            yield TTSStartedFrame()
            
            yield TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self._settings["sample_rate"],
                num_channels=1
            )
            
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"{self} WebSocket connection closed: {e}")
            self._connected = False
            yield ErrorFrame(error=f"WebSocket connection closed: {e}")
            
        except Exception as e:
            logger.error(f"{self} error generating TTS: {e}")
            yield ErrorFrame(error=str(e))
            
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()