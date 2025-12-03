# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """Pipecat client service for Ultravox LLM over WebSocket (audio → streamed text)."""

# import asyncio
# import base64
# import json
# import time
# import uuid
# import sys
# from typing import AsyncGenerator, Dict, List, Optional

# import numpy as np
# import websockets
# from loguru import logger
# from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

# from piopiy.frames.frames import (
#     ErrorFrame,
#     Frame,
#     LLMFullResponseEndFrame,
#     LLMFullResponseStartFrame,
#     LLMTextFrame,
#     TranscriptionFrame
# )
# from piopiy.processors.frame_processor import FrameDirection

# from piopiy.services.llm_service import LLMService
# from piopiy.processors.aggregators.llm_context import LLMContext
# from piopiy.transcriptions.language import Language
# from piopiy.utils.time import time_now_iso8601
# from piopiy.processors.aggregators.llm_response import (
#     LLMUserContextAggregator,
#     LLMAssistantContextAggregator,
# )
# from piopiy.processors.aggregators.openai_llm_context import OpenAILLMContextFrame

# class _Buf:
#     def __init__(self):
#         self.frames: List[TranscriptionFrame] = []
#         self.started_at: Optional[float] = None
#         self.processing: bool = False

# class QwenService(LLMService):
#     """
#     Client that sends buffered audio to the uvx LLM server and streams text back.
#     Downstream TTS will speak the streamed LLMTextFrame.
#     """

#     def __init__(
#         self,
#         *,
#         server_url: str = "ws://localhost:8766",
#         language: Language = Language.EN,
#         temperature: float = 0.7,
#         max_tokens: int = 200,
#         system_prompt: Optional[str] = None,
#         persistent_connection: bool = True,
#         ping_interval: float = 20.0,
#         ping_timeout: float = 20.0,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self._url = server_url
#         self._lang = language
#         self._temp = temperature
#         self._max_tokens = max_tokens
#         self._sys = system_prompt
#         self._persist = persistent_connection
#         self._ping_i = ping_interval
#         self._ping_t = ping_timeout

#         self._ws: Optional[websockets.WebSocketClientProtocol] = None
#         self._lock = asyncio.Lock()
#         self._buf = _Buf()
#         self._inflight: Dict[str, asyncio.Event] = {}
    
#     def create_context_aggregator(self, context):
#         """Create context aggregators for conversation management"""
#         from dataclasses import dataclass
        
#         @dataclass
#         class ContextAggregatorPair:
#             _user: LLMUserContextAggregator
#             _assistant: LLMAssistantContextAggregator
            
#             def user(self):
#                 return self._user
            
#             def assistant(self):
#                 return self._assistant
        
#         user_agg = LLMUserContextAggregator(context)
#         assistant_agg = LLMAssistantContextAggregator(context)
        
#         return ContextAggregatorPair(user_agg, assistant_agg)

#     def can_generate_metrics(self) -> bool:
#         return True

#     async def disconnet_client(self, caller_id: Optional[str] = 123456789):
#         if caller_id is None:
#             return
#         try:
#             logger.info(f"Disconnecting client {caller_id} from Ultravox server")
#         except Exception as e:
#             logger.warning(f"Failed to disconnect client {caller_id}: {e}")

#     async def transfer_client(self, caller_id: Optional[str] = 123456789):
#         if caller_id is None:
#             return
#         try:
#             logger.info(f"Tranfering client {caller_id} to real agent from Ultravox server")
#         except Exception as e:
#             logger.warning(f"Failed to disconnect client {caller_id}: {e}")

#     async def start(self, frame: TranscriptionFrame):
#         logger.info("############# STARTED WITH THE LLM")
#         await super().start(frame)
#         if self._persist:
#             await self._connect()

#     async def process_frame(self, frame: Frame, direction: FrameDirection):
#         await super().process_frame(frame, direction)
#         if isinstance(frame, OpenAILLMContextFrame):
#             messages = frame.context.messages
#             if messages and messages[-1]["role"] == "user":
#                 user_message = messages[-1]["content"]
#                 await self.process_generator(self._process_audio_buffer(user_message,messages))
#             # await self.process_generator(self._process_audio_buffer(frame))
#                 return
#         await self.push_frame(frame, direction)

#     # ------------------ internals ------------------

#     async def _connect(self):
#         if self._ws and not self._ws.closed:
#             return
#         async with self._lock:
#             if self._ws and not self._ws.closed:
#                 return
#             self._ws = await websockets.connect(
#                 self._url, ping_interval=self._ping_i, ping_timeout=self._ping_t, max_size=None
#             )

#     async def _disconnect(self):
#         async with self._lock:
#             if self._ws and not self._ws.closed:
#                 try:
#                     await self._ws.close()
#                 except Exception:
#                     pass
#             self._ws = None

#     async def _ensure(self):
#         if not self._persist:
#             await self._connect()
#             return
#         if not self._ws or self._ws.closed:
#             await self._connect()

#     async def _send_json(self, payload: dict):
#         await self._ensure()
#         assert self._ws is not None
#         await self._ws.send(json.dumps(payload))

#     async def _flush(self):
#         if self._buf.frames and not self._buf.processing:
#             await self.process_generator(self._process_audio_buffer())

#     async def _cancel_all(self):
#         for rid, evt in list(self._inflight.items()):
#             try:
#                 await self._send_json({"type":"cancel","request_id": rid})
#             except Exception:
#                 pass
#             finally:
#                 evt.set()
#                 self._inflight.pop(rid, None)

#     async def _process_audio_buffer(self, user_text: str, conversation_history:List) -> AsyncGenerator[Frame, None]:
#         try:
#             text= user_text
#             # build messages
#             messages = []
#             if self._sys:
#                 messages.append({"role":"system","content": self._sys})
#             messages.append({"role":"user","content": user_text})

#             rid = f"uvx-{uuid.uuid4()}"
#             req = {
#                 "type": "generate",
#                 "request_id": rid,
#                 "language": self._lang.value,
#                 "temperature": self._temp,
#                 "max_tokens": self._max_tokens,
#                 "messages": messages,
#                 "conversation_history":conversation_history
#             }

#             # metrics
#             await self.start_ttfb_metrics()
#             await self.start_processing_metrics()

#             # tell downstream TTS a response is starting
#             yield LLMFullResponseStartFrame()

#             done = asyncio.Event()
#             self._inflight[rid] = done

#             # send
#             await self._send_json(req)

#             # ---------- FIX for cumulative text issue ----------
#             previous_text = ""
#             # ---------- FIX for cumulative text issue ----------

#             # receive loop (non-persistent is also supported)
#             while True:
#                 if not self._persist:
#                     await self._ensure()
#                 assert self._ws is not None
#                 raw = await self._ws.recv()
#                 data = json.loads(raw)
#                 mtype = data.get("type")
#                 if mtype == "started":
#                     # wait for first token to stop TTFB
#                     pass
#                 elif mtype == "partial":
#                     # first arrival => stop TTFB
#                     await self.stop_ttfb_metrics()
#                     text = (data.get("text") or "").strip()
#                     text = text.replace("function_call","")  # to avoid function call text
#                     # cumulative_text = (data.get("text") or "").strip()
#                     cumulative_text = text

#                     # Compute the delta (new text only)
#                     if cumulative_text.startswith(previous_text):
#                         delta = cumulative_text[len(previous_text):]
#                     else:
#                         # Fallback if text doesn't build cumulatively (shouldn't happen)
#                         delta = cumulative_text
                    
#                     previous_text = cumulative_text
#                     if  "function_call" not in delta and delta:
#                         yield LLMTextFrame(text=delta)

#                 elif mtype == "completed":
#                     await self.stop_processing_metrics()

#                     final_text = (data.get("text") or "").strip()
                    
#                     # Check if there's any remaining text not yet sent
#                     if final_text.startswith(previous_text):
#                         delta = final_text[len(previous_text):]
#                         if delta:
#                             yield LLMTextFrame(text=delta)
                    
#                     yield LLMFullResponseEndFrame()
#                     done.set()
#                     break

#                 elif mtype == "error":
#                     await self.stop_processing_metrics()
#                     yield ErrorFrame(f"Ultravox LLM error: {data.get('error')}")
#                     yield LLMFullResponseEndFrame()
#                     done.set()
#                     break
#                 elif mtype == "cancelled":
#                     await self.stop_processing_metrics()
#                     yield LLMFullResponseEndFrame()
#                     done.set()
#                     break
#                 elif mtype == "disconnect":
#                     logger.info(f"###########Received disconnect command from server: {data}")
#                     await self.stop_processing_metrics()
#                     text = data["text"]
#                     yield LLMTextFrame(text=text)
#                     yield LLMFullResponseEndFrame()
#                     done.set()
#                     await self._disconnect()
#                     await self.disconnet_client()
#                     break

#                 elif mtype == "transfer":
#                     logger.info(f"###########Received tranfer command from server: {data}")
#                     await self.stop_processing_metrics()
#                     text = data["text"]
#                     yield LLMTextFrame(text=text)
#                     yield LLMFullResponseEndFrame()
#                     done.set()
#                     await self._disconnect()
#                     await self.transfer_client()
#                     break
                
#             # if non-persistent, close after each call
#             if not self._persist:
#                 await self._disconnect()

#         except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
#             yield ErrorFrame("Connection closed")
#             yield LLMFullResponseEndFrame()
#         except Exception as e:
#             logger.exception(e)
#             yield ErrorFrame(f"Client processing error: {e}")
#             yield LLMFullResponseEndFrame()
#         finally:
#             self._buf.processing = False
#             self._buf.frames = []
#             self._buf.started_at = None









#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pipecat client service for Ultravox LLM over WebSocket (audio → streamed text)."""
import sys
from pathlib import Path

# go up three levels: llm.py -> qwen_model (1) -> Bolan_bhai (2)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import base64
import json
import time
import uuid
import sys
from config.loader import load_config
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np
import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    TextFrame,
    TranscriptionFrame
)
from pipecat.processors.frame_processor import FrameDirection

from pipecat.services.ai_services import LLMService
from pipecat.processors.aggregators.llm_response import (
    LLMUserResponseAggregator,
    LLMAssistantResponseAggregator,
)
from pipecat.processors.frameworks.openai import OpenAILLMContext

config = load_config()
qwen_cfg = config["qwen"]

class _Buf:
    def __init__(self):
        self.frames: List[TranscriptionFrame] = []
        self.started_at: Optional[float] = None
        self.processing: bool = False

class QwenService(LLMService):
    """
    Client that sends buffered audio to the uvx LLM server and streams text back.
    Downstream TTS will speak the streamed TextFrame.
    """

    def __init__(
        self,
        *,
        server_url: str = "ws://localhost:8766",
        language: str = "en",
        temperature: float = 0.7,
        max_tokens: int = 200,
        system_prompt: Optional[str] = None,
        persistent_connection: bool = True,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._url = server_url
        self._lang = language
        self._temp = temperature
        self._max_tokens = max_tokens
        self._sys = system_prompt
        self._persist = persistent_connection
        self._ping_i = ping_interval
        self._ping_t = ping_timeout

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._lock = asyncio.Lock()
        self._buf = _Buf()
        self._inflight: Dict[str, asyncio.Event] = {}
    
    def create_context_aggregator(self, context):
        """Create context aggregators for conversation management"""
        from dataclasses import dataclass
        
        @dataclass
        class ContextAggregatorPair:
            _user: LLMUserResponseAggregator
            _assistant: LLMAssistantResponseAggregator
            
            def user(self):
                return self._user
            
            def assistant(self):
                return self._assistant
        
        user_agg = LLMUserResponseAggregator(context)
        assistant_agg = LLMAssistantResponseAggregator(context)
        
        return ContextAggregatorPair(user_agg, assistant_agg)

    def can_generate_metrics(self) -> bool:
        return True

    async def disconnect_client(self, caller_id: Optional[str] = "123456789"):
        if caller_id is None:
            return
        try:
            logger.info(f"Disconnecting client {caller_id} from Ultravox server")
        except Exception as e:
            logger.warning(f"Failed to disconnect client {caller_id}: {e}")

    async def transfer_client(self, caller_id: Optional[str] = "123456789"):
        if caller_id is None:
            return
        try:
            logger.info(f"Transferring client {caller_id} to real agent from Ultravox server")
        except Exception as e:
            logger.warning(f"Failed to transfer client {caller_id}: {e}")

    async def start(self, frame: Frame):
        logger.info("############# STARTED WITH THE LLM")
        await super().start(frame)
        if self._persist:
            await self._connect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Handle LLMMessagesFrame in pipecat
        if isinstance(frame, LLMMessagesFrame):
            messages = frame.messages
            if messages and messages[-1]["role"] == "user":
                user_message = messages[-1]["content"]
                await self.process_generator(self._process_audio_buffer(user_message, messages))
                return
        
        await self.push_frame(frame, direction)

    # ------------------ internals ------------------

    async def _connect(self):
        if self._ws and not self._ws.closed:
            return
        async with self._lock:
            if self._ws and not self._ws.closed:
                return
            self._ws = await websockets.connect(
                self._url, ping_interval=self._ping_i, ping_timeout=self._ping_t, max_size=None
            )

    async def _disconnect(self):
        async with self._lock:
            if self._ws and not self._ws.closed:
                try:
                    await self._ws.close()
                except Exception:
                    pass
            self._ws = None

    async def _ensure(self):
        if not self._persist:
            await self._connect()
            return
        if not self._ws or self._ws.closed:
            await self._connect()

    async def _send_json(self, payload: dict):
        await self._ensure()
        assert self._ws is not None
        await self._ws.send(json.dumps(payload))

    async def _flush(self):
        if self._buf.frames and not self._buf.processing:
            await self.process_generator(self._process_audio_buffer())

    async def _cancel_all(self):
        for rid, evt in list(self._inflight.items()):
            try:
                await self._send_json({"type":"cancel","request_id": rid})
            except Exception:
                pass
            finally:
                evt.set()
                self._inflight.pop(rid, None)

    async def _process_audio_buffer(self, user_text: str, conversation_history: List) -> AsyncGenerator[Frame, None]:
        try:
            text = user_text
            # build messages
            messages = []
            if self._sys:
                messages.append({"role":"system","content": self._sys})
            messages.append({"role":"user","content": user_text})

            rid = f"uvx-{uuid.uuid4()}"
            req = {
                "type": "generate",
                "request_id": rid,
                "language": self._lang,
                "temperature": self._temp,
                "max_tokens": self._max_tokens,
                "messages": messages,
                "conversation_history": conversation_history
            }

            # metrics
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()

            # tell downstream TTS a response is starting
            yield LLMFullResponseStartFrame()

            done = asyncio.Event()
            self._inflight[rid] = done

            # send
            await self._send_json(req)

            # ---------- FIX for cumulative text issue ----------
            previous_text = ""
            # ---------- FIX for cumulative text issue ----------

            # receive loop (non-persistent is also supported)
            while True:
                if not self._persist:
                    await self._ensure()
                assert self._ws is not None
                raw = await self._ws.recv()
                data = json.loads(raw)
                mtype = data.get("type")
                if mtype == "started":
                    # wait for first token to stop TTFB
                    pass
                elif mtype == "partial":
                    # first arrival => stop TTFB
                    await self.stop_ttfb_metrics()
                    text = (data.get("text") or "").strip()
                    text = text.replace("function_call","")  # to avoid function call text
                    cumulative_text = text

                    # Compute the delta (new text only)
                    if cumulative_text.startswith(previous_text):
                        delta = cumulative_text[len(previous_text):]
                    else:
                        # Fallback if text doesn't build cumulatively (shouldn't happen)
                        delta = cumulative_text
                    
                    previous_text = cumulative_text
                    if "function_call" not in delta and delta:
                        yield TextFrame(text=delta)

                elif mtype == "completed":
                    await self.stop_processing_metrics()

                    final_text = (data.get("text") or "").strip()
                    
                    # Check if there's any remaining text not yet sent
                    if final_text.startswith(previous_text):
                        delta = final_text[len(previous_text):]
                        if delta:
                            yield TextFrame(text=delta)
                    
                    yield LLMFullResponseEndFrame()
                    done.set()
                    break

                elif mtype == "error":
                    await self.stop_processing_metrics()
                    yield ErrorFrame(f"Ultravox LLM error: {data.get('error')}")
                    yield LLMFullResponseEndFrame()
                    done.set()
                    break
                elif mtype == "cancelled":
                    await self.stop_processing_metrics()
                    yield LLMFullResponseEndFrame()
                    done.set()
                    break
                elif mtype == "disconnect":
                    logger.info(f"###########Received disconnect command from server: {data}")
                    await self.stop_processing_metrics()
                    text = data["text"]
                    yield TextFrame(text=text)
                    yield LLMFullResponseEndFrame()
                    done.set()
                    await self._disconnect()
                    await self.disconnect_client()
                    break

                elif mtype == "transfer":
                    logger.info(f"###########Received transfer command from server: {data}")
                    await self.stop_processing_metrics()
                    text = data["text"]
                    yield TextFrame(text=text)
                    yield LLMFullResponseEndFrame()
                    done.set()
                    await self._disconnect()
                    await self.transfer_client()
                    break
                
            # if non-persistent, close after each call
            if not self._persist:
                await self._disconnect()

        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            yield ErrorFrame("Connection closed")
            yield LLMFullResponseEndFrame()
        except Exception as e:
            logger.exception(e)
            yield ErrorFrame(f"Client processing error: {e}")
            yield LLMFullResponseEndFrame()
        finally:
            self._buf.processing = False
            self._buf.frames = []
            self._buf.started_at = None








if __name__ == "__main__":
    qwen_service = QwenService(
        server_url=f"ws://{qwen_cfg['host']}:{qwen_cfg['port']}",
        language=qwen_cfg.get("language", "en"),  
        temperature=qwen_cfg.get("default_temperature", 0.7),
        max_tokens=qwen_cfg.get("default_max_tokens", 1024),
        system_prompt=qwen_cfg.get("system_prompt"),
        persistent_connection=True,
        ping_interval=qwen_cfg["timeouts"]["client_ping_interval"],
        ping_timeout=qwen_cfg["timeouts"]["client_ping_timeout"],
    )

    print("QwenService initialized with config.yaml settings")
