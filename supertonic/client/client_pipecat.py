"""
Custom WebSocket TTS Service for Pipecat

This module provides a Pipecat-compatible TTS service that connects to your
custom WebSocket TTS server. It supports both single synthesis and batch
synthesis modes, with proper handling for interruptions and reconnection.

Usage:
    from custom_tts_service import CustomWebSocketTTSService
    
    tts = CustomWebSocketTTSService(
        url="ws://localhost:8764",
        voice_style="assets/voice_styles/M1.json",
        speed=1.05,
        total_step=5,
    )
"""

import asyncio
import base64
import json
from typing import AsyncGenerator, Optional

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

try:
    import websockets
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("Please install websockets: pip install websockets")
    raise Exception(f"Missing module: {e}")

try:
    import numpy as np
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("Please install numpy: pip install numpy")
    raise Exception(f"Missing module: {e}")


class SupertonicTTSService(TTSService):
    """
    Custom WebSocket-based TTS service for Pipecat.
    
    Connects to your TTS WebSocket server and synthesizes speech from text.
    Supports voice style customization, speed control, and denoising steps.
    
    Attributes:
        url: WebSocket URL of your TTS server (e.g., "ws://localhost:8764")
        voice_style: Path to voice style JSON file on the server
        speed: Speech speed multiplier (default: 1.05)
        total_step: Number of denoising steps (default: 5)
    """

    class InputParams(BaseModel):
        """Input parameters for TTS configuration."""
        voice_style: Optional[str] = None
        speed: float = 1.05
        total_step: int = 5

    def __init__(
        self,
        *,
        url: str = "ws://localhost:8764",
        voice_style: Optional[str] = None,
        speed: float = 1.05,
        total_step: int = 5,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """
        Initialize the Custom WebSocket TTS service.
        
        Args:
            url: WebSocket URL of the TTS server
            voice_style: Path to voice style JSON file (server-side path)
            speed: Speech speed multiplier (0.5-2.0 recommended)
            total_step: Number of denoising steps (higher = better quality, slower)
            sample_rate: Audio sample rate. If None, will be fetched from server.
            params: Additional input parameters
            **kwargs: Additional arguments passed to parent TTSService
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        # Use params if provided, otherwise use individual arguments
        if params:
            voice_style = params.voice_style or voice_style
            speed = params.speed
            total_step = params.total_step
        
        self._url = url
        self._voice_style = voice_style
        self._speed = speed
        self._total_step = total_step
        self._websocket = None
        self._connected = False
        self._server_sample_rate: Optional[int] = None
        
    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True
    
    @property
    def sample_rate(self) -> int:
        """Get the sample rate, preferring server-reported rate."""
        if self._server_sample_rate:
            return self._server_sample_rate
        return super().sample_rate or 24000  # Default fallback
    
    async def start(self, frame: StartFrame):
        """Start the TTS service and connect to WebSocket server."""
        await super().start(frame)
        await self._connect()
        
    async def stop(self, frame: EndFrame):
        """Stop the TTS service and disconnect from WebSocket server."""
        await super().stop(frame)
        await self._disconnect()
        
    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service."""
        await super().cancel(frame)
        await self._disconnect()
    
    async def _connect(self):
        """Establish WebSocket connection to the TTS server."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
                
            logger.debug(f"Connecting to TTS server at {self._url}")
            self._websocket = await websocket_connect(
                self._url,
                max_size=16 * 1024 * 1024,  # 16MB for large audio
            )
            self._connected = True
            
            # Fetch server info to get sample rate
            await self._fetch_server_info()
            logger.info(f"Connected to TTS server (sample_rate: {self._server_sample_rate})")
            
        except Exception as e:
            logger.error(f"{self} connection error: {e}")
            self._websocket = None
            self._connected = False
            raise
    
    async def _disconnect(self):
        """Close WebSocket connection."""
        try:
            if self._websocket:
                logger.debug("Disconnecting from TTS server")
                await self._websocket.close()
                self._websocket = None
            self._connected = False
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
    
    async def _reconnect(self):
        """Reconnect to the TTS server."""
        await self._disconnect()
        await self._connect()
    
    async def _fetch_server_info(self):
        """Fetch server information including sample rate."""
        try:
            request = {"action": "info"}
            await self._websocket.send(json.dumps(request))
            response = await self._websocket.recv()
            data = json.loads(response)
            
            if data.get("status") == "ok":
                self._server_sample_rate = data.get("sample_rate", 24000)
                logger.debug(f"Server info: sample_rate={self._server_sample_rate}")
            else:
                logger.warning(f"Failed to get server info: {data}")
                
        except Exception as e:
            logger.warning(f"Could not fetch server info: {e}")
    
    async def _send_request(self, request: dict) -> dict:
        """Send a request to the TTS server and get response."""
        if not self._websocket or self._websocket.state is not State.OPEN:
            await self._connect()
        
        await self._websocket.send(json.dumps(request))
        response = await self._websocket.recv()
        return json.loads(response)
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """
        Run text-to-speech synthesis on the provided text.
        
        Args:
            text: The text to synthesize into speech
            
        Yields:
            Frame: Audio frames containing the synthesized speech
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        
        try:
            if not self._websocket or self._websocket.state is not State.OPEN:
                await self._connect()
            
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            
            yield TTSStartedFrame()
            
            # Build synthesis request
            request = {
                "action": "synthesize",
                "text": text,
                "speed": self._speed,
                "total_step": self._total_step,
            }
            
            if self._voice_style:
                request["voice_style"] = self._voice_style
            
            # Send request and get response
            response = await self._send_request(request)
            
            await self.stop_ttfb_metrics()
            
            if "error" in response:
                error_msg = response["error"]
                logger.error(f"{self} synthesis error: {error_msg}")
                yield ErrorFrame(error=f"TTS synthesis failed: {error_msg}")
            elif response.get("status") == "success":
                # Decode base64 audio
                audio_base64 = response.get("audio", "")
                audio_bytes = base64.b64decode(audio_base64)
                
                # Convert WAV bytes to raw PCM
                # Skip WAV header (44 bytes) to get raw PCM data
                pcm_data = audio_bytes[44:] if len(audio_bytes) > 44 else audio_bytes
                
                # Convert to numpy array (assuming 16-bit PCM)
                audio_array = np.frombuffer(pcm_data, dtype=np.int16)
                
                # Get sample rate from response or use server default
                response_sample_rate = response.get("sample_rate", self.sample_rate)
                
                # Push audio frame
                frame = TTSAudioRawFrame(
                    audio=audio_array.tobytes(),
                    sample_rate=response_sample_rate,
                    num_channels=1,
                )
                yield frame
                
                logger.debug(
                    f"{self}: Generated {response.get('duration', 0):.2f}s of audio"
                )
            else:
                logger.warning(f"{self} unexpected response: {response}")
                yield ErrorFrame(error="Unexpected TTS response")
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"{self} connection closed: {e}")
            yield ErrorFrame(error=f"TTS connection closed: {e}")
            await self._reconnect()
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
    
    async def set_voice_style(self, voice_style: str):
        """Update the voice style for synthesis."""
        self._voice_style = voice_style
        logger.debug(f"{self}: Voice style set to {voice_style}")
    
    async def set_speed(self, speed: float):
        """Update the speech speed."""
        self._speed = speed
        logger.debug(f"{self}: Speed set to {speed}")
    
    async def set_total_step(self, total_step: int):
        """Update the number of denoising steps."""
        self._total_step = total_step
        logger.debug(f"{self}: Total steps set to {total_step}")


class InterruptibleCustomTTSService(SupertonicTTSService):
    """
    Custom TTS service with interruption handling.
    
    This version handles interruptions by reconnecting to the WebSocket
    when the bot is interrupted while speaking.
    """
    
    def __init__(self, **kwargs):
        """Initialize the interruptible TTS service."""
        super().__init__(**kwargs)
        self._is_speaking = False
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Run TTS with interruption awareness."""
        self._is_speaking = True
        try:
            async for frame in super().run_tts(text):
                yield frame
        finally:
            self._is_speaking = False
    
    async def process_frame(self, frame: Frame, direction):
        """Process frames with interruption handling."""
        from pipecat.frames.frames import StartInterruptionFrame
        from pipecat.processors.frame_processor import FrameDirection
        
        await super().process_frame(frame, direction)
        
        # Handle interruption by reconnecting
        if isinstance(frame, StartInterruptionFrame) and self._is_speaking:
            logger.debug(f"{self}: Handling interruption, reconnecting...")
            await self._reconnect()
            self._is_speaking = False


# Example usage and testing
async def test_service():
    """Test the custom TTS service."""
    import sys
    
    # Create service instance
    tts = SupertonicTTSService(
        url="ws://localhost:8764",
        speed=1.05,
        total_step=5,
    )
    
    # Simulate start
    from pipecat.frames.frames import StartFrame
    start_frame = StartFrame()
    await tts.start(start_frame)
    
    # Test synthesis
    text = "Hello, this is a test of the custom TTS service."
    print(f"Synthesizing: {text}")
    
    async for frame in tts.run_tts(text):
        if isinstance(frame, TTSStartedFrame):
            print("TTS Started")
        elif isinstance(frame, TTSAudioRawFrame):
            print(f"Got audio frame: {len(frame.audio)} bytes")
        elif isinstance(frame, TTSStoppedFrame):
            print("TTS Stopped")
        elif isinstance(frame, ErrorFrame):
            print(f"Error: {frame.error}")
    
    # Cleanup
    from pipecat.frames.frames import EndFrame
    end_frame = EndFrame()
    await tts.stop(end_frame)


if __name__ == "__main__":
    asyncio.run(test_service())