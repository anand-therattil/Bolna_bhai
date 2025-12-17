"""
Pipecat client for Vosk WebSocket ASR server.
Sends complete transcriptions (not streaming) to LLM.
"""

import asyncio
import json
import struct
from typing import AsyncGenerator

import websockets
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
    ErrorFrame,
)
from pipecat.services.ai_services import STTService
from pipecat.transcriptions.language import Language
from loguru import logger


class VoskWebSocketSTT(STTService):
    """
    Pipecat STT service that connects to a Vosk WebSocket server.
    
    Sends audio frames to Vosk and emits TranscriptionFrame for final results.
    Only emits complete transcriptions, not streaming partials to LLM.
    """

    def __init__(
        self,
        *,
        url: str = "ws://localhost:8762",
        sample_rate: int = 16000,
        emit_interim: bool = False,  # Set to True if you want interim frames too
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._url = url
        self._sample_rate = sample_rate
        self._emit_interim = emit_interim
        self._websocket = None
        self._receive_task = None
        self._connected = False

    async def start(self, frame: StartFrame):
        """Connect to Vosk WebSocket server."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Disconnect from Vosk WebSocket server."""
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Handle cancellation."""
        await self._disconnect()
        await super().cancel(frame)

    async def _connect(self):
        """Establish WebSocket connection to Vosk server."""
        try:
            logger.info(f"Connecting to Vosk server at {self._url}")
            self._websocket = await websockets.connect(self._url)
            self._connected = True
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.info("Connected to Vosk server")
        except Exception as e:
            logger.error(f"Failed to connect to Vosk server: {e}")
            self._connected = False
            await self.push_error(ErrorFrame(error=f"Vosk connection failed: {e}"))

    async def _disconnect(self):
        """Close WebSocket connection."""
        self._connected = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            logger.info("Disconnected from Vosk server")

    async def _receive_loop(self):
        """Background task to receive transcriptions from Vosk."""
        try:
            while self._connected and self._websocket:
                try:
                    message = await self._websocket.recv()
                    data = json.loads(message)
                    
                    msg_type = data.get("type")
                    text = data.get("text", "").strip()
                    
                    if msg_type == "final" and text:
                        # Emit final transcription - this goes to LLM
                        logger.debug(f"Final transcription: {text}")
                        await self.push_frame(
                            TranscriptionFrame(
                                text=text,
                                user_id="",
                                timestamp="",
                            )
                        )
                    
                    elif msg_type == "partial" and text and self._emit_interim:
                        # Optionally emit interim transcription
                        logger.debug(f"Interim transcription: {text}")
                        await self.push_frame(
                            InterimTranscriptionFrame(
                                text=text,
                                user_id="",
                                timestamp="",
                            )
                        )
                    
                    elif msg_type == "error":
                        logger.error(f"Vosk error: {data.get('error')}")

                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Vosk WebSocket connection closed")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Vosk response: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in Vosk receive loop: {e}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """
        Process audio through Vosk.
        
        Note: Actual transcription frames are pushed via _receive_loop.
        This method sends audio to Vosk.
        """
        if not self._connected or not self._websocket:
            logger.warning("Not connected to Vosk server")
            return

        try:
            # Send raw PCM audio bytes to Vosk
            await self._websocket.send(audio)
        except Exception as e:
            logger.error(f"Failed to send audio to Vosk: {e}")
            yield ErrorFrame(error=f"Vosk send failed: {e}")

        # Frames are yielded via _receive_loop, not here
        return
        yield  # Make this a generator


# # ------------------------------------------------------------
# # Example Pipeline Usage
# # ------------------------------------------------------------
# async def create_vosk_pipeline():
#     """
#     Example of creating a Pipecat pipeline with Vosk STT.
#     """
#     from pipecat.pipeline.pipeline import Pipeline
#     from pipecat.pipeline.runner import PipelineRunner
#     from pipecat.pipeline.task import PipelineTask, PipelineParams
#     from pipecat.processors.aggregators.sentence import SentenceAggregator
#     from pipecat.processors.frame_processor import FrameProcessor
#     from pipecat.services.openai import OpenAILLMService
#     from pipecat.transports.services.daily import DailyParams, DailyTransport
    
#     # Create Vosk STT service
#     vosk_stt = VoskWebSocketSTT(
#         url="ws://localhost:8764",
#         sample_rate=16000,
#         emit_interim=False,  # Only send complete transcriptions to LLM
#     )
    
#     # Create LLM service (example with OpenAI)
#     llm = OpenAILLMService(
#         api_key="your-api-key",
#         model="gpt-4",
#     )
    
#     # Optional: Aggregate sentences before sending to LLM
#     sentence_aggregator = SentenceAggregator()
    
#     # Build pipeline
#     pipeline = Pipeline([
#         vosk_stt,
#         sentence_aggregator,  # Collects complete sentences
#         llm,
#     ])
    
#     return pipeline

