"""
GPT-2 WebSocket LLM Service for Pipecat

A Pipecat-compatible LLM service that connects to a GPT-2 WebSocket server
for text generation. Integrates seamlessly with Pipecat pipelines.

Usage in bot.py:
    from gpt2_websocket_llm_service import GPT2WebSocketLLMService
    
    llm = GPT2WebSocketLLMService(
        ws_url="ws://localhost:8764",
        caller_id="my_bot",
        temperature=0.7,
        max_tokens=200,
    )
"""

import asyncio
import json
import uuid
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass

import websockets
from websockets.client import WebSocketClientProtocol
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    StartFrame,
    EndFrame,
    CancelFrame,
    ErrorFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    LLMMessagesFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService


@dataclass
class GPT2InputParams:
    """Input parameters for GPT-2 model configuration."""
    temperature: float = 0.7
    max_tokens: int = 200
    top_p: float = 1.0
    top_k: int = 50


class GPT2WebSocketLLMService(LLMService):
    """
    Pipecat LLM service that connects to a GPT-2 WebSocket server.
    
    This service:
    - Connects to a GPT-2 WebSocket server on startup
    - Processes OpenAILLMContextFrame and LLMMessagesFrame frames
    - Streams text responses via LLMTextFrame
    - Handles connection lifecycle (connect/disconnect)
    
    Args:
        ws_url: WebSocket URL of the GPT-2 server (e.g., "ws://localhost:8764")
        caller_id: Unique identifier for this client connection
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Top-p (nucleus) sampling parameter
        top_k: Top-k sampling parameter
        reconnect_attempts: Number of reconnection attempts on failure
        reconnect_delay: Delay between reconnection attempts (seconds)
    """

    def __init__(
        self,
        *,
        ws_url: str = "ws://localhost:8764",
        caller_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        top_p: float = 1.0,
        top_k: int = 50,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self._ws_url = ws_url
        self._caller_id = caller_id or f"gpt2_pipecat_{uuid.uuid4().hex[:8]}"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._top_k = top_k
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        
        # Connection state
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._connecting = False
        self._connection_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        """Check if the service is connected to the GPT-2 server."""
        return self._connected and self._websocket is not None

    async def start(self, frame: StartFrame):
        """Initialize the service and connect to GPT-2 server."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and disconnect from GPT-2 server."""
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Handle cancellation - disconnect from server."""
        await self._disconnect()
        await super().cancel(frame)

    async def _connect(self) -> bool:
        """
        Establish WebSocket connection to GPT-2 server.
        
        Returns:
            True if connection successful, False otherwise.
        """
        async with self._connection_lock:
            if self._connected:
                return True
            
            if self._connecting:
                return False
            
            self._connecting = True
            
            for attempt in range(self._reconnect_attempts):
                try:
                    logger.info(f"Connecting to GPT-2 server at {self._ws_url} (attempt {attempt + 1}/{self._reconnect_attempts})")
                    
                    self._websocket = await websockets.connect(
                        self._ws_url,
                        ping_interval=20,
                        ping_timeout=20,
                    )
                    
                    # Send connection handshake
                    await self._websocket.send(json.dumps({
                        "type": "connect",
                        "caller_id": self._caller_id,
                    }))
                    
                    # Wait for connection acknowledgment
                    response = await asyncio.wait_for(
                        self._websocket.recv(),
                        timeout=10.0
                    )
                    data = json.loads(response)
                    
                    if data.get("type") == "connected":
                        self._connected = True
                        self._connecting = False
                        logger.info(f"Connected to GPT-2 server as {self._caller_id}")
                        return True
                    else:
                        logger.warning(f"Unexpected connection response: {data}")
                        
                except asyncio.TimeoutError:
                    logger.warning(f"Connection timeout (attempt {attempt + 1})")
                except Exception as e:
                    logger.error(f"Connection failed (attempt {attempt + 1}): {e}")
                
                if attempt < self._reconnect_attempts - 1:
                    await asyncio.sleep(self._reconnect_delay)
            
            self._connecting = False
            logger.error(f"Failed to connect to GPT-2 server after {self._reconnect_attempts} attempts")
            return False

    async def _disconnect(self):
        """Close WebSocket connection."""
        async with self._connection_lock:
            self._connected = False
            
            if self._websocket:
                try:
                    await self._websocket.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
                finally:
                    self._websocket = None
                    logger.info("Disconnected from GPT-2 server")

    async def _ensure_connected(self) -> bool:
        """Ensure connection is established, reconnecting if necessary."""
        if self.connected:
            return True
        return await self._connect()

    def _format_messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert OpenAI-style messages to a single prompt string for GPT-2.
        
        GPT-2 doesn't have native chat support, so we format messages
        into a conversation-style prompt.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            
        Returns:
            Formatted prompt string.
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)
        
        # Add prompt for assistant response
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)

    async def _generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Send generation request and yield response tokens.
        
        Args:
            prompt: The formatted prompt to send to GPT-2.
            
        Yields:
            Text tokens as they are received.
        """
        if not await self._ensure_connected():
            logger.error("Cannot generate: not connected to GPT-2 server")
            return
        
        request_id = str(uuid.uuid4())
        
        request = {
            "type": "generate",
            "request_id": request_id,
            "caller_id": self._caller_id,
            "text": prompt,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
            "top_k": self._top_k,
        }
        
        try:
            await self._websocket.send(json.dumps(request))
            
            previous_text = ""
            
            while True:
                try:
                    raw = await asyncio.wait_for(
                        self._websocket.recv(),
                        timeout=60.0  # Generation timeout
                    )
                    data = json.loads(raw)
                    event_type = data.get("type")
                    
                    if event_type == "started":
                        logger.debug(f"Generation started for request {request_id}")
                        
                    elif event_type == "partial":
                        # Get the new text since last partial
                        full_text = data.get("text", "")
                        new_text = full_text[len(previous_text):]
                        previous_text = full_text
                        
                        if new_text:
                            yield new_text
                            
                    elif event_type == "completed":
                        # Get any remaining text
                        final_text = data.get("text", "")
                        new_text = final_text[len(previous_text):]
                        
                        if new_text:
                            yield new_text
                        
                        logger.debug(f"Generation completed for request {request_id}")
                        break
                        
                    elif event_type == "error":
                        error_msg = data.get("error", "Unknown error")
                        logger.error(f"GPT-2 generation error: {error_msg}")
                        break
                        
                except asyncio.TimeoutError:
                    logger.error("Generation timeout")
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            logger.error("WebSocket connection closed during generation")
            self._connected = False
        except Exception as e:
            logger.error(f"Error during generation: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """
        Process incoming frames and generate LLM responses.
        
        Handles:
        - OpenAILLMContextFrame: Process context and generate response
        - LLMMessagesFrame: Process messages and generate response
        
        Note: StartFrame, EndFrame, CancelFrame are handled by the base class.
        We just need to override start(), stop(), cancel() methods.
        """
        await super().process_frame(frame, direction)
        
        if isinstance(frame, OpenAILLMContextFrame):
            # Extract messages from context
            context: OpenAILLMContext = frame.context
            messages = context.get_messages()
            await self._process_messages(messages)
            
        elif isinstance(frame, LLMMessagesFrame):
            # Process messages directly
            messages = frame.messages
            await self._process_messages(messages)
            
        else:
            # Pass through all other frames (including StartFrame, EndFrame, etc.)
            await self.push_frame(frame, direction)

    async def _process_messages(self, messages: List[Dict[str, Any]]):
        """
        Process messages and generate LLM response.
        
        Args:
            messages: List of message dicts to process.
        """
        if not messages:
            logger.warning("No messages to process")
            return
        
        # Format messages into a prompt
        prompt = self._format_messages_to_prompt(messages)
        logger.debug(f"Generated prompt: {prompt[:100]}...")
        
        # Signal start of response
        await self.push_frame(LLMFullResponseStartFrame())
        
        try:
            # Stream tokens
            async for token in self._generate(prompt):
                await self.push_frame(LLMTextFrame(text=token))
                
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
            await self.push_frame(ErrorFrame(error=str(e)))
            
        finally:
            # Signal end of response
            await self.push_frame(LLMFullResponseEndFrame())

    async def generate_text(
        self,
        text: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Convenience method for direct text generation (non-pipeline use).
        
        Args:
            text: Input text/prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Generated text response.
        """
        # Temporarily override settings if provided
        orig_temp = self._temperature
        orig_max = self._max_tokens
        
        if temperature is not None:
            self._temperature = temperature
        if max_tokens is not None:
            self._max_tokens = max_tokens
        
        try:
            result = []
            async for token in self._generate(text):
                result.append(token)
            return "".join(result)
        finally:
            # Restore original settings
            self._temperature = orig_temp
            self._max_tokens = orig_max


# For backwards compatibility with the import in bot.py
__all__ = ["GPT2WebSocketLLMService", "GPT2InputParams"]