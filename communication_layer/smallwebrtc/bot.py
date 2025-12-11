import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator,
    LLMAssistantContextAggregator,
)
# from pipecat.services.ai4bharat.stt import Ai4BharatSTTService
# from pipecat.services.indic_parler.tts import IndicParlerTTSService
# from pipecat.services.indri.tts import IndriTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.gpt2.llm import GPT2WebSocketLLMService
from pipecat.services.supertonic.tts import InterruptibleCustomTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv(override=True)

SYSTEM_INSTRUCTION = f"""
"You are  VOICE BOT , a friendly, helpful robot.

Respond to what the user said in a creative and helpful way. Keep your responses brief. One or two sentences at most.
"""


async def run_bot(webrtc_connection):
    pipecat_transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=2,
        ),
    )
    # stt = Ai4BharatSTTService(ws_url = "ws://localhost:8761",language= "hi",)
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), language="en-US")
    llm = GPT2WebSocketLLMService(
        ws_url="ws://localhost:8764",
        caller_id=12345,
    )

    tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-2-andromeda-en")
    # tts = IndriTTSService(
    #     base_url="ws://localhost:8760",
    #     voice="[spkr_63]",
    #     sample_rate=24000
    # )

    # tts = IndicParlerTTSService(
    #         websocket_url="ws://localhost:8763",
    #         voice_preset="female_expressive",  # or custom description
    #     )     
    tts = InterruptibleCustomTTSService(
        url="ws://0.0.0.0:8764",
    )                  
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))


    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
    ]

    context = OpenAILLMContext(messages)
    
    # Create context aggregator pair manually
    class ContextAggregatorPair:
        def __init__(self, user_agg, assistant_agg):
            self._user = user_agg
            self._assistant = assistant_agg
        
        def user(self):
            return self._user
        
        def assistant(self):
            return self._assistant
    
    user_aggregator = LLMUserContextAggregator(context)
    assistant_aggregator = LLMAssistantContextAggregator(context)
    context_aggregator = ContextAggregatorPair(user_aggregator, assistant_aggregator)


    pipeline = Pipeline(
        [
            pipecat_transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            pipecat_transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )


    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @pipecat_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        # Kick off the conversation.
        await task.queue_frames([LLMRunFrame()])

    @pipecat_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)