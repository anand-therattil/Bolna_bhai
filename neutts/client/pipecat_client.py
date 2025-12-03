"""
NeuTTS Pipecat TTS Service Implementation.

This module provides a Pipecat-compatible TTS service that connects to
the NeuTTS WebSocket server for text-to-speech synthesis.
"""

import asyncio
import base64
import io
import json
from typing import AsyncGenerator, Optional

import websockets
from loguru import logger
from pydantic import BaseModel

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


class NeuTTSService(TTSService):
    """NeuTTS WebSocket-based text-to-speech service.
    
    Provides text-to-speech synthesis by connecting to a NeuTTS WebSocket
    server. The server handles voice cloning using pre-configured reference
    audio and text.
    
    Example::
    
        tts = NeuTTSService(
            websocket_url="ws://localhost:8764",
            sample_rate=24000
        )
    """
    
    class InputParams(BaseModel):
        """Input parameters for NeuTTS configuration.
        
        Parameters:
            output_format: Audio output format (wav, flac, etc.)
        """
        output_format: str = "wav"
    
    def __init__(
        self,
        *,
        websocket_url: str = "ws://localhost:8764",
        sample_rate: int = 24000,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the NeuTTS service.
        
        Args:
            websocket_url: WebSocket URL for the NeuTTS server.
            sample_rate: Audio sample rate. Defaults to 24000.
            params: Additional input parameters.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._websocket_url = websocket_url
        self._params = params or self.InputParams()
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._connection_lock = asyncio.Lock()
    
    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.
        
        Returns:
            True, as NeuTTS service supports metrics generation.
        """
        return True
    
    def _is_connected(self) -> bool:
        """Check if WebSocket is connected.
        
        Returns:
            True if connected, False otherwise.
        """
        if self._websocket is None:
            return False
        try:
            # websockets >= 14.0 uses state
            from websockets.protocol import State
            return self._websocket.state == State.OPEN
        except (ImportError, AttributeError):
            # Fallback for older versions or different API
            try:
                return not self._websocket.closed
            except AttributeError:
                # If neither works, check if connection exists
                return self._websocket is not None
    
    async def _ensure_connection(self) -> websockets.WebSocketClientProtocol:
        """Ensure WebSocket connection is established.
        
        Returns:
            Active WebSocket connection.
        """
        async with self._connection_lock:
            if not self._is_connected():
                logger.debug(f"{self}: Connecting to {self._websocket_url}")
                self._websocket = await websockets.connect(self._websocket_url)
                logger.info(f"{self}: Connected to NeuTTS server")
            return self._websocket
    
    async def _disconnect(self):
        """Close WebSocket connection."""
        async with self._connection_lock:
            if self._websocket is not None:
                try:
                    await self._websocket.close()
                except Exception:
                    pass
                self._websocket = None
                logger.debug(f"{self}: Disconnected from NeuTTS server")
    
    async def start(self, frame: StartFrame):
        """Start the TTS service.
        
        Args:
            frame: The start frame.
        """
        await super().start(frame)
        await self._ensure_connection()
    
    async def stop(self, frame: EndFrame):
        """Stop the TTS service.
        
        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()
    
    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service.
        
        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using NeuTTS WebSocket server.
        
        Args:
            text: The text to synthesize into speech.
            
        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        
        try:
            await self.start_ttfb_metrics()
            
            websocket = await self._ensure_connection()
            
            # Send synthesis request
            request = {
                "action": "synthesize",
                "text": text,
                "format": self._params.output_format,
                "sample_rate": self.sample_rate
            }
            
            await websocket.send(json.dumps(request))
            
            # Wait for response
            response_text = await websocket.recv()
            response = json.loads(response_text)
            
            await self.stop_ttfb_metrics()
            
            if response.get("status") != "success":
                error_msg = response.get("message", "Unknown error")
                logger.error(f"{self}: TTS error: {error_msg}")
                yield ErrorFrame(error=error_msg)
                return
            
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()
            
            # Decode audio from base64
            audio_base64 = response.get("audio", "")
            audio_bytes = base64.b64decode(audio_base64)
            
            # Skip WAV header (44 bytes) to get raw PCM data
            if audio_bytes[:4] == b'RIFF':
                audio_bytes = audio_bytes[44:]
            
            # Stream audio in chunks
            chunk_size = self.chunk_size
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                yield TTSAudioRawFrame(
                    audio=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=1
                )
            
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"{self}: WebSocket connection closed: {e}")
            self._websocket = None
            yield ErrorFrame(error=f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"{self}: Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            yield TTSStoppedFrame()


class NeuTTSStreamingService(TTSService):
    """NeuTTS WebSocket-based TTS service with per-request connections.
    
    Creates a new WebSocket connection for each TTS request. This is simpler
    but may have higher latency for rapid requests.
    
    Example::
    
        tts = NeuTTSStreamingService(
            websocket_url="ws://localhost:8764",
            sample_rate=24000
        )
    """
    
    def __init__(
        self,
        *,
        websocket_url: str = "ws://localhost:8764",
        sample_rate: int = 24000,
        **kwargs,
    ):
        """Initialize the NeuTTS streaming service.
        
        Args:
            websocket_url: WebSocket URL for the NeuTTS server.
            sample_rate: Audio sample rate. Defaults to 24000.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._websocket_url = websocket_url
    
    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using NeuTTS WebSocket server.
        
        Args:
            text: The text to synthesize into speech.
            
        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        
        try:
            await self.start_ttfb_metrics()
            
            async with websockets.connect(self._websocket_url) as websocket:
                # Send synthesis request
                request = {
                    "action": "synthesize",
                    "text": text,
                    "sample_rate": self.sample_rate
                }
                
                await websocket.send(json.dumps(request))
                
                # Wait for response
                response_text = await websocket.recv()
                response = json.loads(response_text)
                
                await self.stop_ttfb_metrics()
                
                if response.get("status") != "success":
                    error_msg = response.get("message", "Unknown error")
                    logger.error(f"{self}: TTS error: {error_msg}")
                    yield ErrorFrame(error=error_msg)
                    return
                
                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()
                
                # Decode audio from base64
                audio_base64 = response.get("audio", "")
                audio_bytes = base64.b64decode(audio_base64)
                
                # Skip WAV header (44 bytes) to get raw PCM data
                if audio_bytes[:4] == b'RIFF':
                    audio_bytes = audio_bytes[44:]
                
                # Stream audio in chunks
                chunk_size = self.chunk_size
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1
                    )
                    
        except Exception as e:
            logger.error(f"{self}: Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            yield TTSStoppedFrame()