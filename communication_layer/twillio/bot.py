import datetime
import io
import os
import wave
from typing import Optional

import aiofiles
import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.gpt2.llm import GPT2WebSocketLLMService
from pipecat.services.supertonic.tts import InterruptibleCustomTTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv(override=True)

SYSTEM_INSTRUCTION = """
You are VOICE BOT, a friendly, helpful robot.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""


async def get_call_info(call_sid: str) -> dict:
    """Fetch call information from Twilio REST API using aiohttp.

    Args:
        call_sid: The Twilio call SID

    Returns:
        Dictionary containing call information including from_number, to_number, status, etc.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if not account_sid or not auth_token:
        logger.warning("Missing Twilio credentials, cannot fetch call info")
        return {}

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Calls/{call_sid}.json"

    try:
        # Use HTTP Basic Auth with aiohttp
        auth = aiohttp.BasicAuth(account_sid, auth_token)

        async with aiohttp.ClientSession() as session:
            async with session.get(url, auth=auth) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Twilio API error ({response.status}): {error_text}")
                    return {}

                data = await response.json()

                call_info = {
                    "from_number": data.get("from"),
                    "to_number": data.get("to"),
                }

                return call_info

    except Exception as e:
        logger.error(f"Error fetching call info from Twilio: {e}")
        return {}


async def save_audio(audio: bytes, sample_rate: int, num_channels: int):
    """Save audio buffer to a WAV file."""
    if len(audio) > 0:
        filename = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")


async def run_bot(transport: BaseTransport, handle_sigint: bool, testing: Optional[bool] = False):
    """Run the voice bot pipeline with Twilio transport."""
    
    # Initialize STT service
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        language="en-US"
    )

    # Initialize LLM service - choose one of the following:
    # Option 1: OpenAI LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Option 2: GPT2 WebSocket LLM (uncomment to use)
    # llm = GPT2WebSocketLLMService(
    #     ws_url="ws://localhost:8764",
    #     caller_id=12345,
    # )

    # Initialize TTS service - choose one of the following:
    # Option 1: Deepgram TTS
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-2-andromeda-en"
    )
    
    # Option 2: Custom Interruptible TTS (uncomment to use)
    # tts = InterruptibleCustomTTSService(
    #     url="ws://0.0.0.0:8764",
    # )

    # Initialize context
    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # Audio buffer for recording (optional)
    audiobuffer = AudioBufferProcessor()

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from Twilio
            stt,  # Speech-To-Text
            context_aggregator.user(),  # User context aggregator
            llm,  # LLM
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to Twilio
            audiobuffer,  # Audio buffer for recording
            context_aggregator.assistant(),  # Assistant context aggregator
        ]
    )

    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,  # Twilio uses 8kHz
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Twilio client connected")
        # Start recording
        await audiobuffer.start_recording()
        # Kick off the conversation
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Twilio client disconnected")
        await task.cancel()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        await save_audio(audio, sample_rate, num_channels)

    # Run the pipeline
    runner = PipelineRunner(handle_sigint=handle_sigint, force_gc=True)
    await runner.run(task)


async def bot(runner_args: RunnerArguments, testing: Optional[bool] = False):
    """Main bot entry point compatible with Pipecat Cloud and Twilio."""

    # Parse Twilio websocket data
    _, call_data = await parse_telephony_websocket(runner_args.websocket)

    # Fetch call information from Twilio REST API
    call_info = await get_call_info(call_data["call_id"])
    if call_info:
        logger.info(f"Call from: {call_info.get('from_number')} to: {call_info.get('to_number')}")

    # Create Twilio serializer
    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    # Create FastAPI websocket transport with Twilio configuration
    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,  # Twilio doesn't need WAV headers
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    await run_bot(transport, runner_args.handle_sigint, testing)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()