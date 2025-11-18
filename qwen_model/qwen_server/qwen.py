import asyncio
import json
import os
import uuid
import re
import signal
import sys
from typing import Dict, Optional, List

import websockets
from loguru import logger
from websockets.server import WebSocketServerProtocol

# vLLM + HF
from huggingface_hub import login
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# --------------------- Simple system prompt for testing ---------------------
SIMPLE_SYSTEM_PROMPT = """You are a helpful assistant for Customer Support. Be helpful and answer the user's questions directly."""

# --------------------- Server state ---------------------
ACTIVE: Dict[str, asyncio.Task] = {}
OWNER: Dict[str, WebSocketServerProtocol] = {}
CALLER_IDS: Dict[WebSocketServerProtocol, str] = {}
MODEL: Optional['QwenModel'] = None


class QwenModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Initializing model: {model_name}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Tokenizer loaded. EOS token ID: {self.tokenizer.eos_token_id}")
        
        # Initialize vLLM engine with conservative settings
        args = AsyncEngineArgs(
            model=model_name,
            gpu_memory_utilization=0.5,  # Conservative GPU usage
            max_model_len=4096,  # Reduced context length
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)
        logger.info("vLLM engine initialized")

    def format_prompt(self, messages: List[dict]) -> str:
        """Format prompt for generation."""
        # If no messages, create a simple one
        if not messages:
            messages = [{"role": "user", "content": "Hello"}]
        
        # Add simple system prompt if not present
        has_system = any(msg.get("role") == "system" for msg in messages)
        if not has_system:
            messages = [
                {"role": "system", "content": SIMPLE_SYSTEM_PROMPT}
            ] + messages
        
        # Use tokenizer's chat template WITHOUT thinking mode
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug(f"Formatted prompt length: {len(prompt)} chars")
            logger.debug(f"Prompt preview: {prompt[:200]}...")
            return prompt
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            # Fallback to simple format
            return "User: " + messages[-1].get("content", "Hello") + "\nAssistant:"

    async def stream_generate(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
    ):
        """Stream generation from model."""
        try:
            # Format prompt
            prompt = self.format_prompt(messages)
            
            # Conservative sampling parameters
            sampling = SamplingParams(
                temperature=max(0.1, temperature),  # Ensure some randomness
                max_tokens=max_tokens,
                stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []
            )
            
            logger.info(f"Starting generation with temp={sampling.temperature}, max_tokens={max_tokens}")
            
            # Generate
            gen_id = "qwen-" + os.urandom(6).hex()
            prev = ""
            token_count = 0
            
            async for out in self.engine.generate({"prompt": prompt}, sampling, gen_id):
                t = out.outputs[0].text
                delta = t[len(prev):]
                prev = t
                
                if delta:
                    token_count += 1
                    logger.debug(f"Token {token_count}: {repr(delta[:50])}")
                    yield delta
            
            logger.info(f"Generation completed. Total tokens: {token_count}")
            
            # If nothing was generated, yield a default response
            if token_count == 0:
                logger.warning("No tokens generated, sending default response")
                yield "I apologize, but I'm having trouble generating a response. Could you please try again?"
                
        except Exception as e:
            logger.error(f"Error in stream_generate: {e}")
            yield f"Error: {str(e)}"


# --------------------- Safe send helper ---------------------
async def safe_send(ws: WebSocketServerProtocol, payload: dict) -> None:
    """Send message to WebSocket client."""
    try:
        logger.debug(f"Sending: {payload.get('type')} - {str(payload.get('text', ''))[:100]}")
        await ws.send(json.dumps(payload))
    except Exception as e:
        logger.error(f"Failed to send message: {e}")


# --------------------- Handlers ---------------------
async def handle_connect(ws: WebSocketServerProtocol, msg: dict):
    """Handle initial connection."""
    caller_id = msg.get("caller_id", f"anonymous_{uuid.uuid4().hex[:8]}")
    CALLER_IDS[ws] = caller_id
    logger.info(f"✓ Client connected: {caller_id}")
    await safe_send(ws, {"type": "connected", "caller_id": caller_id})


async def handle_generate(ws: WebSocketServerProtocol, msg: dict):
    """Handle text generation request."""
    rid = msg.get("request_id", str(uuid.uuid4()))
    temp = float(msg.get("temperature", 0.7))
    max_tok = int(msg.get("max_tokens", 200))
    messages = msg.get("messages", [])
    text_input = msg.get("text", "")
    caller_id = msg.get("caller_id", CALLER_IDS.get(ws, "unknown"))
    
    logger.info(f"Generation request from {caller_id}: {text_input[:100] if text_input else 'messages'}")
    
    # Convert text input to messages if needed
    if text_input and not messages:
        messages = [{"role": "user", "content": text_input}]
    elif text_input and messages:
        messages.append({"role": "user", "content": text_input})
    
    if not messages:
        await safe_send(ws, {
            "type": "error",
            "request_id": rid,
            "error": "No input provided"
        })
        return
    
    OWNER[rid] = ws
    await safe_send(ws, {"type": "started", "request_id": rid})
    
    async def _generate():
        global MODEL
        
        try:
            full_response = ""
            partial_count = 0
            
            # Stream generation
            async for piece in MODEL.stream_generate(messages, temp, max_tok):
                if piece:
                    full_response += piece
                    partial_count += 1
                    
                    # Send partial every few tokens
                    if partial_count % 3 == 0:  # Send every 3 tokens
                        await safe_send(ws, {
                            "type": "partial",
                            "request_id": rid,
                            "text": full_response
                        })
            
            # Send final response
            logger.info(f"Sending final response: {full_response[:100]}...")
            await safe_send(ws, {
                "type": "completed",
                "request_id": rid,
                "text": full_response.strip(),
                "caller_id": caller_id
            })
            
        except asyncio.CancelledError:
            await safe_send(ws, {"type": "cancelled", "request_id": rid})
            raise
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            await safe_send(ws, {
                "type": "error",
                "request_id": rid,
                "error": str(e)
            })
        finally:
            ACTIVE.pop(rid, None)
            OWNER.pop(rid, None)
    
    task = asyncio.create_task(_generate())
    ACTIVE[rid] = task


async def handle_cancel(ws: WebSocketServerProtocol, msg: dict):
    """Handle cancellation request."""
    rid = msg.get("request_id", "")
    task = ACTIVE.get(rid)
    
    if task and not task.done():
        task.cancel()
        logger.info(f"Cancelled request: {rid}")
    
    ACTIVE.pop(rid, None)
    OWNER.pop(rid, None)


async def ws_handler(ws: WebSocketServerProtocol):
    """Main WebSocket handler."""
    logger.info("New client connection")
    
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
                mtype = msg.get("type")
                
                logger.debug(f"Received message type: {mtype}")
                
                if mtype == "connect":
                    await handle_connect(ws, msg)
                elif mtype == "generate":
                    await handle_generate(ws, msg)
                elif mtype == "cancel":
                    await handle_cancel(ws, msg)
                else:
                    await safe_send(ws, {"type": "error", "error": f"Unknown type: {mtype}"})
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                await safe_send(ws, {"type": "error", "error": "Invalid JSON"})
            except Exception as e:
                logger.error(f"Handler error: {e}")
                await safe_send(ws, {"type": "error", "error": str(e)})
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    finally:
        # Cleanup
        CALLER_IDS.pop(ws, None)
        to_cancel = [rid for rid, owner in OWNER.items() if owner is ws]
        for rid in to_cancel:
            task = ACTIVE.pop(rid, None)
            OWNER.pop(rid, None)
            if task and not task.done():
                task.cancel()


async def test_model():
    """Test model generation directly."""
    global MODEL
    
    logger.info("="*50)
    logger.info("TESTING MODEL DIRECTLY")
    logger.info("="*50)
    
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    
    response = ""
    async for chunk in MODEL.stream_generate(messages, 0.7, 50):
        response += chunk
        print(chunk, end="", flush=True)
    
    print()
    
    if response.strip():
        logger.success(f"✓ Model test passed! Response: {response[:100]}")
        return True
    else:
        logger.error("✗ Model test failed - empty response")
        return False


async def main():
    """Main entry point."""
    # Login to HuggingFace if token provided
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    # Initialize model
    model_name = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    
    global MODEL
    MODEL = QwenModel(model_name)
    
    # Test the model first
    logger.info("Testing model generation...")
    if not await test_model():
        logger.error("Model test failed! Check your setup.")
        return
    
    logger.info(f"✓ Model loaded and tested: {model_name}")
    
    # Server configuration
    host = os.environ.get("QWEN_HOST", "0.0.0.0")
    port = int(os.environ.get("QWEN_PORT", "8766"))
    
    logger.info(f"🚀 Starting server on ws://{host}:{port}")
    
    async with websockets.serve(
        ws_handler,
        host,
        port,
        ping_interval=20,
        ping_timeout=20,
        max_size=None
    ):
        stop = asyncio.Future()
        loop = asyncio.get_running_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: stop.set_result(True))
            except NotImplementedError:
                pass
        
        await stop


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)