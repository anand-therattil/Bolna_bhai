"""
Indic Parler TTS service implementation for Pipecat.

This module provides a production-ready integration with the Indic Parler TTS 
WebSocket server for text-to-speech synthesis in Indian languages.

Usage:
    ```python
    from indic_parler_pipecat_tts import IndicParlerTTSService
    
    tts = IndicParlerTTSService(
        websocket_url="ws://localhost:8765",
        sample_rate=22050,
        params=IndicParlerTTSService.InputParams(
            description="A female speaker with clear voice..."
        )
    )
    ```
"""

import base64
import io
import json
import struct
import asyncio
from typing import AsyncGenerator, Optional

import aiohttp
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


def parse_wav_header(wav_bytes: bytes) -> tuple[int, int, int]:
    """Parse WAV header to get audio parameters.
    
    Args:
        wav_bytes: WAV file bytes
        
    Returns:
        Tuple of (sample_rate, num_channels, bits_per_sample)
    """
    if len(wav_bytes) < 44:
        return 22050, 1, 16  # Default values
    
    try:
        # WAV header format
        num_channels = struct.unpack('<H', wav_bytes[22:24])[0]
        sample_rate = struct.unpack('<I', wav_bytes[24:28])[0]
        bits_per_sample = struct.unpack('<H', wav_bytes[34:36])[0]
        return sample_rate, num_channels, bits_per_sample
    except Exception:
        return 22050, 1, 16  # Default values


class IndicParlerTTSService(TTSService):
    """Indic Parler TTS service using WebSocket connection.
    
    Provides text-to-speech synthesis using the Indic Parler TTS model
    through a WebSocket server. Supports multiple Indian languages and
    customizable voice descriptions.
    
    Supported languages include Hindi, Tamil, Telugu, Kannada, Malayalam,
    Bengali, Marathi, Gujarati, Punjabi, and more.
    
    Example:
        ```python
        tts = IndicParlerTTSService(
            websocket_url="ws://localhost:8765",
            params=IndicParlerTTSService.InputParams(
                description="A female speaker with clear, expressive voice."
            )
        )
        
        # Use in a Pipecat pipeline
        pipeline = Pipeline([
            llm,
            tts,
            transport.output(),
        ])
        ```
    """
    
    class InputParams(BaseModel):
        """Input parameters for Indic Parler TTS configuration.
        
        Parameters:
            description: Voice description for the TTS model. Controls
                speaker characteristics, emotion, speed, pitch, and 
                audio quality. The model interprets natural language
                descriptions to generate appropriate voice output.
            response_format: Format for the audio response from the server.
                Either "base64" (default) or "binary".
        """
        description: str = (
            "A female speaker delivers a slightly expressive and animated speech "
            "with a moderate speed and pitch. The recording is of very high quality, "
            "with the speaker's voice sounding clear and very close up."
        )
        response_format: str = "base64"
    
    # Voice description presets for common use cases
    VOICE_PRESETS = {
        "female_expressive": (
            "A female speaker delivers a slightly expressive and animated speech "
            "with a moderate speed and pitch. The recording is of very high quality, "
            "with the speaker's voice sounding clear and very close up."
        ),
        "female_calm": (
            "A female speaker delivers calm and soothing speech with a slow pace "
            "and low pitch. The recording is of very high quality with minimal "
            "background noise."
        ),
        "male_professional": (
            "A male speaker delivers clear and professional speech with moderate "
            "speed and pitch. The recording is of high quality, suitable for "
            "business communication."
        ),
        "male_energetic": (
            "A male speaker delivers energetic and enthusiastic speech with "
            "fast pace and varied pitch. The recording is crisp and clear."
        ),
    }
    
    def __init__(
        self,
        *,
        websocket_url: str = "ws://localhost:8765",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        voice_preset: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Indic Parler TTS service.
        
        Args:
            websocket_url: WebSocket URL for the TTS server.
            sample_rate: Audio sample rate in Hz. If None, auto-detected
                from server response.
            params: Voice description and other parameters.
            voice_preset: Name of a voice preset to use. One of:
                "female_expressive", "female_calm", "male_professional",
                "male_energetic". Overridden by params.description if provided.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        # Set default sample rate if not provided
        super().__init__(sample_rate=sample_rate or 22050, **kwargs)
        
        self._websocket_url = websocket_url
        self._params = params or IndicParlerTTSService.InputParams()
        self._session: Optional[aiohttp.ClientSession] = None
        self._auto_sample_rate = sample_rate is None
        
        # Apply voice preset if specified and no custom description provided
        if voice_preset and voice_preset in self.VOICE_PRESETS:
            if params is None or params.description == IndicParlerTTSService.InputParams().description:
                self._params.description = self.VOICE_PRESETS[voice_preset]
        
        self.set_model_name("indic-parler-tts")
    
    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.
        
        Returns:
            True, as this service supports metrics generation.
        """
        return True
    
    async def start(self, frame: StartFrame):
        """Start the TTS service and create HTTP session.
        
        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if not self._session:
            self._session = aiohttp.ClientSession()
        logger.info(f"IndicParlerTTSService started, server: {self._websocket_url}")
    
    async def stop(self, frame: EndFrame):
        """Stop the TTS service and cleanup resources.
        
        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("IndicParlerTTSService stopped")
    
    async def cancel(self, frame: CancelFrame):
        """Cancel ongoing TTS operations.
        
        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        logger.debug("IndicParlerTTSService cancelled")
    
    def set_voice(self, voice: str):
        """Set the voice using a preset name or custom description.
        
        Args:
            voice: Either a preset name ("female_expressive", etc.)
                or a custom voice description string.
        """
        if voice in self.VOICE_PRESETS:
            self._params.description = self.VOICE_PRESETS[voice]
        else:
            self._params.description = voice
    
    async def set_voice_description(self, description: str):
        """Set the voice description for synthesis.
        
        Args:
            description: The voice description string.
        """
        self._params.description = description
    
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Indic Parler TTS.
        
        This method connects to the WebSocket server, sends the text
        for synthesis, and yields audio frames as they become available.
        
        Args:
            text: The text to synthesize into speech. Supports Hindi,
                Tamil, Telugu, and other Indian languages.
            
        Yields:
            Frame: Audio frames containing the synthesized speech.
            
        Raises:
            ErrorFrame: If connection fails or synthesis errors occur.
        """
        logger.debug(f"IndicParlerTTSService: Generating TTS for [{text}]")
        
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        try:
            await self.start_ttfb_metrics()
            
            async with self._session.ws_connect(
                self._websocket_url,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as ws:
                # Send synthesis request
                request = {
                    "prompt": text,
                    "description": self._params.description,
                    "format": self._params.response_format,
                }
                await ws.send_str(json.dumps(request))
                
                # Receive response
                response = await ws.receive()
                
                if response.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(response.data)
                    
                    if data.get("status") == "success":
                        await self.start_tts_usage_metrics(text)
                        
                        # Decode audio
                        audio_base64 = data.get("audio", "")
                        audio_bytes = base64.b64decode(audio_base64)
                        
                        # Parse WAV header for audio parameters
                        detected_sample_rate, num_channels, _ = parse_wav_header(audio_bytes)
                        
                        # Update sample rate if auto-detecting
                        if self._auto_sample_rate:
                            self._sample_rate = detected_sample_rate
                        
                        # Use server-provided sample rate if available
                        server_sample_rate = data.get("sample_rate", detected_sample_rate)
                        
                        # Skip WAV header (44 bytes) to get raw PCM
                        raw_audio = audio_bytes[44:] if len(audio_bytes) > 44 else audio_bytes
                        
                        await self.stop_ttfb_metrics()
                        
                        yield TTSStartedFrame()
                        
                        # Yield audio in chunks for smoother streaming playback
                        # Chunk size: ~100ms of audio
                        bytes_per_sample = 2  # 16-bit audio
                        chunk_duration_ms = 100
                        chunk_size = int(
                            server_sample_rate * bytes_per_sample * num_channels * 
                            chunk_duration_ms / 1000
                        )
                        
                        for i in range(0, len(raw_audio), chunk_size):
                            chunk = raw_audio[i:i + chunk_size]
                            yield TTSAudioRawFrame(
                                audio=chunk,
                                sample_rate=server_sample_rate,
                                num_channels=num_channels,
                            )
                        
                        yield TTSStoppedFrame()
                        
                    else:
                        error_msg = data.get("message", "Unknown error from TTS server")
                        logger.error(f"TTS synthesis error: {error_msg}")
                        yield ErrorFrame(f"TTS error: {error_msg}")
                        
                elif response.type == aiohttp.WSMsgType.BINARY:
                    # Handle binary response (raw WAV data)
                    audio_bytes = response.data
                    detected_sample_rate, num_channels, _ = parse_wav_header(audio_bytes)
                    raw_audio = audio_bytes[44:] if len(audio_bytes) > 44 else audio_bytes
                    
                    await self.stop_ttfb_metrics()
                    
                    yield TTSStartedFrame()
                    
                    chunk_size = int(detected_sample_rate * 2 * num_channels * 0.1)
                    for i in range(0, len(raw_audio), chunk_size):
                        chunk = raw_audio[i:i + chunk_size]
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=detected_sample_rate,
                            num_channels=num_channels,
                        )
                    
                    yield TTSStoppedFrame()
                    
                elif response.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    yield ErrorFrame(f"WebSocket error: {ws.exception()}")
                    
                else:
                    logger.error(f"Unexpected response type: {response.type}")
                    yield ErrorFrame(f"Unexpected response type: {response.type}")
                    
        except aiohttp.ClientConnectorError as e:
            error_msg = f"Cannot connect to TTS server at {self._websocket_url}: {e}"
            logger.error(error_msg)
            yield ErrorFrame(error_msg)
            
        except aiohttp.WSServerHandshakeError as e:
            error_msg = f"WebSocket handshake failed: {e}"
            logger.error(error_msg)
            yield ErrorFrame(error_msg)
            
        except asyncio.TimeoutError:
            error_msg = "TTS request timed out"
            logger.error(error_msg)
            yield ErrorFrame(error_msg)
            
        except Exception as e:
            error_msg = f"TTS generation error: {e}"
            logger.error(error_msg)
            yield ErrorFrame(error_msg)


# Convenience function for creating the service
def create_indic_tts(
    websocket_url: str = "ws://localhost:8765",
    voice_preset: str = "female_expressive",
    **kwargs,
) -> IndicParlerTTSService:
    """Create an Indic Parler TTS service with common defaults.
    
    Args:
        websocket_url: WebSocket URL for the TTS server.
        voice_preset: Voice preset name.
        **kwargs: Additional arguments for IndicParlerTTSService.
        
    Returns:
        Configured IndicParlerTTSService instance.
    """
    return IndicParlerTTSService(
        websocket_url=websocket_url,
        voice_preset=voice_preset,
        **kwargs,
    )