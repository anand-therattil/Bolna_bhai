#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-2 Pipecat LLM Service

A custom Pipecat LLMService implementation that connects to your GPT-2 vLLM 
WebSocket server. This service can be used as a drop-in replacement for other 
LLM services in Pipecat pipelines.

Usage:
    from gpt2_pipecat_service import GPT2WebSocketLLMService
    
    llm = GPT2WebSocketLLMService(
        ws_url="ws://localhost:8764",
        caller_id="my-bot"
    )
    
    # Use in pipeline like any other LLM service
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMMessagesFrame,
    TextFrame,
    ErrorFrame,
    CancelFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use GPT2WebSocketLLMService, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")


class GPT2WebSocketLLMService(LLMService):
    """
    LLM service that connects to a GPT-2 vLLM WebSocket server.
    
    This service implements the Pipecat LLMService interface and communicates
    with GPT-2 WebSocket server using the protocol defined in your
    server code.
    
    Protocol (JSON messages):
    - Client -> Server:
        {"type": "connect", "caller_id": "..."}
        {"type": "generate", "request_id": "...", "caller_id": "...", "text": "...",
         "temperature": 0.7, "top_p": 0.9, "max_tokens": 200}
        {"type": "cancel", "request_id": "..."}
    - Server -> Client:
        {"type": "connected"}
        {"type": "started", "request_id": "..."}
        {"type": "partial", "request_id": "...", "text": "..."}
        {"type": "completed", "request_id": "...", "text": "..."}
        {"type": "error", "request_id": "...", "error": "..."}
    """
    
    class InputParams(BaseModel):
        """Input parameters for GPT-2 model configuration."""
        temperature: float = 0.7
        top_p: float = 0.9
        max_tokens: int = 200
    
    def __init__(
        self,
        *,
        ws_url: str = "ws://localhost:8764",
        caller_id: str = "pipecat-client",
        params: Optional[InputParams] = None,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        **kwargs
    ):
        """
        Initialize the GPT-2 WebSocket LLM service.
        
        Args:
            ws_url: WebSocket URL of the GPT-2 server (e.g., "ws://localhost:8764")
            caller_id: Identifier for this client connection
            params: Model parameters (temperature, top_p, max_tokens)
            reconnect_attempts: Number of reconnection attempts on disconnect
            reconnect_delay: Delay between reconnection attempts (seconds)
            **kwargs: Additional arguments passed to parent LLMService
        """
        super().__init__(**kwargs)
        
        self._ws_url = ws_url
        self._caller_id = caller_id
        self._params = params or self.InputParams()
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._current_request_id: Optional[str] = None
        self._lock = asyncio.Lock()
    
    async def start(self, frame: Frame):
        """Start the service and establish WebSocket connection."""
        await super().start(frame)
        await self._connect()
    
    async def stop(self, frame: Frame):
        """Stop the service and close WebSocket connection."""
        await self._disconnect()
        await super().stop(frame)
    
    async def cancel(self, frame: Frame):
        """Cancel any ongoing generation and close connection."""
        if self._current_request_id and self._ws:
            try:
                await self._ws.send(json.dumps({
                    "type": "cancel",
                    "request_id": self._current_request_id
                }))
            except Exception as e:
                logger.warning(f"Failed to send cancel: {e}")
        await self._disconnect()
        await super().cancel(frame)
    
    async def _connect(self) -> bool:
        """Establish WebSocket connection to the GPT-2 server."""
        for attempt in range(self._reconnect_attempts):
            try:
                logger.info(f"Connecting to GPT-2 server at {self._ws_url} (attempt {attempt + 1})")
                self._ws = await websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=20
                )
                
                # Send connect message
                await self._ws.send(json.dumps({
                    "type": "connect",
                    "caller_id": self._caller_id
                }))
                
                # Wait for connected response
                response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
                msg = json.loads(response)
                
                if msg.get("type") == "connected":
                    self._connected = True
                    logger.info(f"Connected to GPT-2 server as {self._caller_id}")
                    return True
                else:
                    logger.warning(f"Unexpected response from server: {msg}")
                    await self._ws.close()
                    
            except asyncio.TimeoutError:
                logger.warning(f"Connection timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.warning(f"Connection failed (attempt {attempt + 1}): {e}")
            
            if attempt < self._reconnect_attempts - 1:
                await asyncio.sleep(self._reconnect_delay)
        
        logger.error(f"Failed to connect to GPT-2 server after {self._reconnect_attempts} attempts")
        self._connected = False
        return False
    
    async def _disconnect(self):
        """Close WebSocket connection."""
        self._connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
            finally:
                self._ws = None
    
    async def _ensure_connected(self) -> bool:
        """Ensure we have an active connection, reconnecting if necessary."""
        if self._connected and self._ws:
            return True
        return await self._connect()
    
    def _extract_text_from_context(self, context: OpenAILLMContext | LLMContext) -> str:
        """
        Extract the user's text prompt from the LLM context.
        
        The context contains the conversation history in OpenAI format.
        We extract the last user message as the prompt for GPT-2.
        """
        messages = context.messages if hasattr(context, 'messages') else []
        
        # Find the last user message
        user_text = ""
        for msg in reversed(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "user":
                # Handle both string content and list content (multimodal)
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    # Extract text from content list
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_text = item.get("text", "")
                            break
                        elif isinstance(item, str):
                            user_text = item
                            break
                break
        
        # Optionally include system prompt context
        system_text = ""
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_text = content
                break
        
        # Combine system context with user text if available
        if system_text and user_text:
            return f"{system_text}\n\nUser: {user_text}"
        
        return user_text
    
    async def _generate_stream(
        self,
        text: str,
        request_id: str
    ) -> AsyncGenerator[str, None]:
        """
        Stream generation from the GPT-2 server.
        
        Args:
            text: The input text/prompt
            request_id: Unique ID for this generation request
            
        Yields:
            Partial text chunks as they arrive from the server
        """
        if not await self._ensure_connected():
            raise ConnectionError("Failed to connect to GPT-2 server")
        
        self._current_request_id = request_id
        
        try:
            # Send generate request
            generate_msg = {
                "type": "generate",
                "request_id": request_id,
                "caller_id": self._caller_id,
                "text": text,
                "temperature": self._params.temperature,
                "top_p": self._params.top_p,
                "max_tokens": self._params.max_tokens
            }
            
            await self._ws.send(json.dumps(generate_msg))
            logger.debug(f"Sent generate request: {request_id}")
            
            # Track previous text for incremental updates
            previous_text = ""
            
            # Receive streaming responses
            while True:
                try:
                    response = await asyncio.wait_for(self._ws.recv(), timeout=60.0)
                    msg = json.loads(response)
                    
                    msg_type = msg.get("type")
                    msg_request_id = msg.get("request_id")
                    
                    # Ignore messages for other requests
                    if msg_request_id and msg_request_id != request_id:
                        continue
                    
                    if msg_type == "started":
                        logger.debug(f"Generation started: {request_id}")
                        continue
                    
                    elif msg_type == "partial":
                        # vLLM typically sends cumulative text, so we extract the delta
                        current_text = msg.get("text", "")
                        if len(current_text) > len(previous_text):
                            delta = current_text[len(previous_text):]
                            previous_text = current_text
                            yield delta
                    
                    elif msg_type == "completed":
                        # Final text - send any remaining delta
                        final_text = msg.get("text", "")
                        if len(final_text) > len(previous_text):
                            delta = final_text[len(previous_text):]
                            yield delta
                        logger.debug(f"Generation completed: {request_id}")
                        break
                    
                    elif msg_type == "error":
                        error = msg.get("error", "Unknown error")
                        logger.error(f"Generation error: {error}")
                        raise RuntimeError(f"GPT-2 server error: {error}")
                    
                    elif msg_type == "cancelled":
                        logger.info(f"Generation cancelled: {request_id}")
                        break
                    
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")
                
                except asyncio.TimeoutError:
                    logger.error(f"Timeout waiting for response: {request_id}")
                    raise
        
        finally:
            self._current_request_id = None
    
    async def _process_text_directly(self, text: str):
        """
        Process text directly (for LLMMessagesFrame compatibility).
        """
        if not text:
            logger.warning("No text provided to process")
            return
        
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Processing text directly: {text[:100]}...")
        
        # Push start frame
        await self.push_frame(LLMFullResponseStartFrame())
        
        try:
            # Stream the response
            async for chunk in self._generate_stream(text, request_id):
                if chunk:
                    # Push each text chunk as a TextFrame
                    await self.push_frame(TextFrame(text=chunk))
        
        except asyncio.CancelledError:
            logger.info(f"Generation cancelled for request {request_id}")
            raise
        
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            await self.push_frame(ErrorFrame(error=str(e)))
        
        finally:
            # Push end frame
            await self.push_frame(LLMFullResponseEndFrame())
    
    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        """
        Process an LLM context and generate a response.
        
        This is the main method called by the Pipecat pipeline when an
        LLMContextFrame is received.
        """
        # Extract text from context
        text = self._extract_text_from_context(context)
        
        if not text:
            logger.warning("No text found in context to process")
            return
        
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        
        logger.info(f"Processing context, text: {text[:100]}...")
        
        # Push start frame
        await self.push_frame(LLMFullResponseStartFrame())
        
        try:
            # Stream the response
            async for chunk in self._generate_stream(text, request_id):
                if chunk:
                    # Push each text chunk as a TextFrame
                    await self.push_frame(TextFrame(text=chunk))
        
        except asyncio.CancelledError:
            logger.info(f"Generation cancelled for request {request_id}")
            raise
        
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            await self.push_frame(ErrorFrame(error=str(e)))
        
        finally:
            # Push end frame
            await self.push_frame(LLMFullResponseEndFrame())
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process incoming frames from the pipeline.
        
        Handles LLMContextFrame, OpenAILLMContextFrame, and LLMMessagesFrame to trigger generation.
        """
        await super().process_frame(frame, direction)
        
        if isinstance(frame, (OpenAILLMContextFrame)):
            context = frame.context
            await self._process_context(context)
            return  # Don't pass through, we handle it
        
        elif hasattr(frame, 'context') and hasattr(frame.context, 'messages'):
            # Handle generic LLMContextFrame (from LLMContext)
            context = frame.context
            await self._process_context(context)
            return  # Don't pass through, we handle it
        
        elif hasattr(frame, 'messages'):  # Handle LLMMessagesFrame for testing
            # Extract user message for direct testing
            messages = frame.messages
            if messages and messages[-1].get("role") == "user":
                user_text = messages[-1].get("content", "")
                if user_text:
                    # Create a mock context for testing
                    class MockContext:
                        def __init__(self, messages):
                            self.messages = messages
                    
                    mock_context = MockContext(messages)
                    await self._process_context(mock_context)
            return  # Don't pass through, we handle it
        
        elif isinstance(frame, LLMMessagesFrame):
            # Handle legacy LLMMessagesFrame format
            messages = frame.messages
            if messages and messages[-1]["role"] == "user":
                user_text = messages[-1]["content"]
                await self._process_text_directly(user_text)
            return  # Don't pass through, we handle it
        
        elif isinstance(frame, CancelFrame):
            # Cancel any ongoing generation
            if self._current_request_id and self._ws:
                try:
                    await self._ws.send(json.dumps({
                        "type": "cancel",
                        "request_id": self._current_request_id
                    }))
                except Exception as e:
                    logger.warning(f"Failed to send cancel: {e}")
            await self.push_frame(frame, direction)
        
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)
    
    # For compatibility with Pipecat's context aggregator system
    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        **kwargs
    ):
        """
        Create a context aggregator pair for this LLM service.
        
        Uses the standard OpenAI context aggregator since our service
        accepts OpenAI-formatted contexts.
        """
        from pipecat.processors.aggregators.openai_llm_context import (
            OpenAIUserContextAggregator,
            OpenAIAssistantContextAggregator,
        )
        
        user_aggregator = OpenAIUserContextAggregator(context)
        assistant_aggregator = OpenAIAssistantContextAggregator(context)
        
        class ContextAggregatorPair:
            def __init__(self, user, assistant, context):
                self._user = user
                self._assistant = assistant
                self._context = context
            
            def user(self):
                return self._user
            
            def assistant(self):
                return self._assistant
            
            @property
            def context(self):
                return self._context
        
        return ContextAggregatorPair(user_aggregator, assistant_aggregator, context)