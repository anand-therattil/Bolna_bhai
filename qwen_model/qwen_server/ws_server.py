#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen LLM server with intelligent function calling for text-based interactions.
The LLM decides when to disconnect or transfer based on conversation context.
"""

import sys
from pathlib import Path

# go up three levels: qwen.py -> qwen_server (1) -> qwen_model (2) -> Bolan_bhai (3)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import json
import os, uuid
import re
import signal
import sys
from typing import Dict, Optional, List

import websockets
from loguru import logger
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

# vLLM + HF
from huggingface_hub import login
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs


from config.loader import load_config

cfg = load_config()
QWEN_CFG = cfg.get("qwen", {})

# ---- BASIC SETTINGS ----
QWEN_SYSTEM_PROMPT = QWEN_CFG.get("system_prompt", "")
QWEN_HOST = QWEN_CFG.get("host", "0.0.0.0")
QWEN_PORT = QWEN_CFG.get("port", 8766)
QWEN_MODEL_NAME = QWEN_CFG.get("model_name", "Qwen/Qwen2.5-7B-Instruct")

# ---- MODEL / vLLM SETTINGS ----
GPU_UTIL = QWEN_CFG.get("gpu_memory_utilization", 0.6)
MAX_MODEL_LEN = QWEN_CFG.get("max_model_len", 4096)

DEFAULT_TEMP = QWEN_CFG.get("default_temperature", 0.7)
MIN_TEMP = QWEN_CFG.get("temperature_min", 0.1)
DEFAULT_MAX_TOKENS = QWEN_CFG.get("default_max_tokens", 512)
PARTIAL_N = QWEN_CFG.get("partial_every_n_tokens", 3)

# ---- TIMEOUT SETTINGS ----
timeouts_cfg = QWEN_CFG.get("timeouts", {})
GEN_TIMEOUT = timeouts_cfg.get("generation_timeout_sec", 60)
PING_INTERVAL = timeouts_cfg.get("client_ping_interval", 20)
PING_TIMEOUT = timeouts_cfg.get("client_ping_timeout", 20)

# ---- ADVANCED vLLM ----
vllm_cfg = QWEN_CFG.get("vllm", {})
TENSOR_PARALLEL = vllm_cfg.get("tensor_parallel_size", 1)
MAX_NUM_SEQS = vllm_cfg.get("max_num_seqs", 2048)

# ---- LOGGING SETTINGS ----
log_cfg = QWEN_CFG.get("logging", {})
LOG_LEVEL = log_cfg.get("level", "INFO")
LOG_FILE = log_cfg.get("logfile", "qwen_server.log")

if LOG_FILE:
    logger.add(LOG_FILE, level=LOG_LEVEL, rotation="10 MB")

# --------------------- Function calling config ---------------------
FUNCTION_DEFINITIONS = """
Available Functions:

1. function_call
   Description: To be used when additional information about **BOLNA** is needed to response or to decide to disconnect or transfer the call.
   Parameters:
   - transcription: string (required) - The user's message text
   
   When to use:
   ✓ Not able to answer user's question directly
   ✓ Need to gather more context from external service or database
   ✓ User says they need to go, have to leave, or are busy
   
   When NOT to use:
   ✗ User is asking simple questions you can answer directly
   ✗ User is abusing or being rude
   ✗ User explicitly says "hmm" or "let me think" or "hold on"

CRITICAL: You MUST use this EXACT format - no variations allowed:

 <function_call> 
 <parameters> 
 {"transcribe_text": "<USER MESSAGE TEXT>"} 
 </parameters> 
 </function_call> 

IMPORTANT: 
- The transcribe_text in the function_call should contain the user's message
- Use ONLY the tags shown above: <function_call>, <parameters>
- DO NOT create tags like <disconnect_call> or other variations
- Parameters MUST be valid JSON on a single line
"""

SYSTEM_PROMPT_TEMPLATE = '''
You are a pre sales agent team. Your job is respond ask questions about the requirement for the Calling Solution
***Ask Question one by one not all together.***
### QUESTION NEEDED TO BE ASKED ####:
1. How many **user** are required?
2. Current CRM Integration present ? if yes which one ? 
3. Which type of calling inbound or outboud or both? 
4. Which Solution are you looking for PSTN or Webrtc?
5. Which Calling solution they have presently ? 
6. What is the Call Volume they expect ? 


***You must use **function call** to get the necessary information about products and pricing and to make the descision for the disconnect or transfer***

{function_definitions}

{rules}
🟡 FUNCTION CALL POLICY (CRITICAL):
- Use a function call *anytime* you're unsure, missing details, or need accurate, real-time information.
- Never fabricate answers.
- if information regarding **BOLNA*** is requried use **function_call**
- Function calls should follow the exact format:  
  `<function_call>,<parameters>`

🔴 CRITICAL RULES (DO NOT BREAK):
1. If you're unsure, or need verification — **always** trigger a function call.
2. Use *only* the exact function call tag format: `<function_call>,<parameters>`
3. Do not guess or make assumptions — rely on function calls instead.

✅ Remember: Your role is to support the user thoughtfully and clearly.

'''
RULES = '''

***You must use **function call** to get the necessary information about products and pricing and to make the descision for the disconnect or transfer***

🟡 FUNCTION CALL POLICY (CRITICAL):
- Use a function call *anytime* you're unsure, missing details, or need accurate, real-time information.
- Never fabricate answers.
- if information regarding **BOLNA** or **about company** is requried use **function_call**
- Function calls should follow the exact format:  
  `<function_call>,<parameters>`

🔴 CRITICAL RULES (DO NOT BREAK):
1. If you're unsure, or need verification — **always** trigger a function call.
2. Use *only* the exact function call tag format: `<function_call>,<parameters>`
3. Do not guess or make assumptions — rely on function calls instead.

✅ Remember: Your role is to support the user thoughtfully and clearly.
'''
# --------------------- Model wrapper ---------------------

class QwenFunctionCallClient:
    """
    WebSocket client for connecting to Qwen Function Calling Server.
    Sends user messages and receives action decisions (disconnect/transfer/respond).
    """
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        """
        Initialize the client.
        
        Args:
            server_url: WebSocket server URL
        """
        self.server_url = server_url
        self.websocket = None
        self.connected = False
        self.pending_requests = {}
        
    async def connect(self):
        """Connect to the Qwen Function Calling Server."""
        try:
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=20,    # Wait 20 seconds for pong
                close_timeout=10    # Wait 10 seconds for close
            )
            
            self.connected = True
            logger.info(f"Connected to Qwen server at {self.server_url}")
            
            # Receive connection confirmation
            response = await self.websocket.recv()
            connection_msg = json.loads(response)
            logger.info(f"Server message: {connection_msg.get('message')}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from server")
    
    async def analyze_conversation(
        self,
        transcription: str,
        caller_id: str = "unknown",
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Send user message to server for analysis.
        
        Args:
            transcription: User's message text
            caller_id: Identifier for the caller
            conversation_history: Optional conversation context
            
        Returns:
            Dictionary with action decision
        """
        if not self.connected:
            raise ConnectionError("Not connected to server. Call connect() first.")
        
        request_id = str(uuid.uuid4())
        
        # Build request
        request = {
            "type": "analyze_conversation",
            "request_id": request_id,
            "caller_id": caller_id,
            "transcription": transcription,
            "conversation_history": conversation_history
        }
        
        try:
            # Send request
            await self.websocket.send(json.dumps(request))
            logger.info(f"Sent analysis request for caller {caller_id}")
            
            # Wait for response
            response = await self.websocket.recv()
            result = json.loads(response)
            
            if result.get("type") == "error":
                logger.error(f"Server error: {result.get('message')}")
                return {
                    "disconnect": "false",
                    "transfer": "false",
                    "response": "Sorry, there was an error processing your request.",
                    "error": result.get('message')
                }
            
            if result.get("type") == "action_decision":
                decision = {
                    "disconnect": result.get("disconnect"),
                    "transfer": result.get("transfer"),
                    "response": result.get("response"),
                    "caller_id": result.get("caller_id"),
                    "request_id": result.get("request_id")
                }
                
                logger.info(f"Received decision - Disconnect: {decision['disconnect']}, "
                          f"Transfer: {decision['transfer']}")
                logger.info(f"Response: {decision['response'][:100]}...")
                
                return decision
            
            return result
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return {
                "disconnect": "false",
                "transfer": "false",
                "response": "Sorry, I'm having trouble processing that right now.",
                "error": str(e)
            }
    
    async def handle_conversation_turn(
        self,
        user_message: str,
        caller_id: str = "test_caller",
        conversation_history: Optional[List[Dict]] = None
    ):
        """
        Handle a single conversation turn with full logging.
        
        Args:
            user_message: The user's message
            caller_id: Caller identifier
            conversation_history: Previous conversation context
        """
       
        decision = await self.analyze_conversation(
            transcription=user_message,
            caller_id=caller_id,
            conversation_history=conversation_history
        )
        
        return decision


class QwenTextModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        args = AsyncEngineArgs(
            model=model_name,
            gpu_memory_utilization=GPU_UTIL,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=TENSOR_PARALLEL,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(args)

    def format_prompt(self, messages: List[dict]) -> str:
        """Format prompt with function calling instructions."""
        # Check if system message exists 
        has_system = any(msg.get("role") == "system" for msg in messages)
        
        if not has_system:
            # Add system message with function definitions
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_TEMPLATE.format(
                        function_definitions=FUNCTION_DEFINITIONS,
                        rules =RULES
                    )
                }
            ] + messages
        else:
            # Inject function definitions into existing system message
            messages = messages.copy()
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    content = msg["content"]
                    if "function_call" not in content:
                        messages[i] = {
                            "role": "system",
                            "content": content + "\n\n" + FUNCTION_DEFINITIONS
                        }
                    break
        logger.info(f"####### Current Prompt:{messages}")
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    async def stream_generate(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
    ):
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        inputs = {"prompt": self.format_prompt(messages)}
        gen_id = "qwen-" + os.urandom(6).hex()
        prev = ""
        async for out in self.engine.generate(inputs, sampling, gen_id):
            t = out.outputs[0].text
            delta = t[len(prev):]
            prev = t
            yield delta

# --------------------- Server state ---------------------

ACTIVE: Dict[str, asyncio.Task] = {}
OWNER: Dict[str, WebSocketServerProtocol] = {}
CALLER_IDS: Dict[WebSocketServerProtocol, str] = {}
MODEL: Optional[QwenTextModel] = None
QWEN_CLIENT: Optional[QwenFunctionCallClient] = None


# --------------------- Function call parsing ---------------------

def parse_function_call(text: str) -> Optional[dict]:
    """
    Parse function call from model output with strict format enforcement.
    ONLY accepts: <function_call><parameters>...</parameters></function_call>
    """
    # Strict pattern - only accept proper format
    pattern = r'<function_call>\s*<parameters>(.*?)</parameters>\s*</function_call>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if not match:
        # Check for common hallucinations and log them
        if '<disconnect_call>' in text.lower():
            logger.warning("⚠️ Model hallucinated <disconnect_call> tag instead of proper format!")
            logger.warning(f"   Generated text: {text[-200:]}")
        elif '<function>' in text.lower():
            logger.warning("⚠️ Model hallucinated <function> tag instead of <function_call>!")
            logger.warning(f"   Generated text: {text[-200:]}")
        return None
    
    params_str = match.group(1).strip()
    
    try:
        parameters = json.loads(params_str)
        return {
            "parameters": parameters,
            "raw_text": match.group(0)
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse function parameters: {params_str}")
        logger.warning(f"JSON decode error: {e}")
        return None

def extract_user_message(text: str) -> str:
    """Extract the user-facing message, removing all function call tags."""
    # Remove properly formatted function calls
    cleaned = re.sub(
        r'<function_call>.*?</function_call>',
        '',
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # Also remove any hallucinated tags as fallback
    cleaned = re.sub(r'<disconnect_call>.*?</disconnect_call>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<function>.*?</function>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    return cleaned.strip()

# --------------------- Safe send helper ---------------------

async def safe_send(ws: WebSocketServerProtocol, payload: dict) -> None:
    """Best-effort send that never raises if the socket is closed."""
    if hasattr(ws, 'state') and ws.state.name != 'OPEN':
        return
    
    try:
        logger.debug(f"Sending message: {payload}")
        await ws.send(json.dumps(payload))
    except (ConnectionClosed, ConnectionClosedOK, ConnectionClosedError):
        pass
    except Exception as e:
        logger.debug(f"Failed to send message: {e}")
        pass

# --------------------- Handlers ---------------------

async def handle_connect(ws: WebSocketServerProtocol, msg: dict):
    """Handle initial connection with caller ID."""
    caller_id = msg.get("caller_id")
    if not caller_id:
        await safe_send(ws, {"type": "error", "error": "caller_id required"})
        return
    
    CALLER_IDS[ws] = caller_id
    logger.info(f"✓ Client connected with caller_id: {caller_id}")
    await safe_send(ws, {"type": "connected", "caller_id": caller_id})

async def handle_generate(ws: WebSocketServerProtocol, msg: dict):
    """Handle text generation request."""
    rid = msg.get("request_id") or ""
    lang = (msg.get("language") or "en").lower()
    temp = float(msg.get("temperature", DEFAULT_TEMP))
    temp = max(temp, MIN_TEMP)  # never let it go below min
    max_tok = int(msg.get("max_tokens", DEFAULT_MAX_TOKENS))

    messages = msg.get("messages", [])
    text_input = msg.get("text", "")  # Direct text input
    conversation = msg.get("conversation_history","")
    
    # If text_input is provided, add it as a user message
    if text_input and not messages:
        messages = [{"role": "user", "content": text_input}]
    
    if not conversation:
        await safe_send(ws, {"type": "error", "request_id": rid, "error": "No input provided"})
        return 
    
    caller_id = msg.get("caller_id") or CALLER_IDS.get(ws, "unknown")

    OWNER[rid] = ws
    await safe_send(ws, {"type": "started", "request_id": rid})

    async def _job():
        global MODEL

        try:
            full_response = ""
            function_call_detected = None
            suppress_partials = False
            
            # Stream generation from model
            async for piece in MODEL.stream_generate(conversation, temp, max_tok):
                if piece:
                    full_response += piece
                    
                    # Check if function call is starting
                    if full_response.count(" ") == 1 and "<function_call>" in full_response:
                        suppress_partials = True
                    
                    # Only send partial updates if we haven't detected function call start
                    if full_response.count(" ") > 1 and not suppress_partials:
                        user_text = extract_user_message(full_response)
                        await safe_send(ws, {
                            "type": "partial",
                            "request_id": rid,
                            "text": user_text,
                            "language": lang
                        })
            
            # Process function call if detected
            if suppress_partials:
                function_call_detected = parse_function_call(full_response)
                if function_call_detected:
                    params = function_call_detected["parameters"]
                    transcription = params.get("transcribe_text", "")
                    logger.info(f"Function call detected with parameters: {params}")
                    
                    qwen_client = QwenFunctionCallClient(server_url="ws://localhost:8765")
                    try:
                        # Connect, use, and disconnect
                        connected = await qwen_client.connect()
                        if connected:
                            decision = await qwen_client.analyze_conversation(
                                transcription=params.get("transcription", transcription),
                                caller_id=caller_id
                            )
                            reason = decision.get("response", "Could you repeat that?")
                            disconnect = decision.get("disconnect", "false").lower() == "true"
                            transfer = decision.get("transfer", "false").lower() == "true"
                        else:
                            logger.error("Failed to connect to Qwen server")
                            reason = "I'm having some technical difficulties. Could you please repeat that?"
                            disconnect = False
                            transfer = False
                    
                    except Exception as e:
                        logger.error(f"Error calling Qwen server: {e}")
                        reason = "I'm having some technical difficulties. Could you please repeat that?"
                        disconnect = False
                        transfer = False
                    
                    finally:
                        # Always disconnect after use
                        await qwen_client.disconnect()
                    
                    # Send appropriate response based on decision
                    if disconnect:
                        await safe_send(ws, {
                            "type": "disconnect",
                            "request_id": rid,
                            "text": reason,
                            "caller_id": caller_id,
                            "language": lang
                        })
                    elif transfer:
                        await safe_send(ws, {
                            "type": "transfer",
                            "request_id": rid,
                            "text": reason,
                            "caller_id": caller_id,
                            "language": lang
                        })
                    else:
                        await safe_send(ws, {
                            "type": "completed",
                            "request_id": rid,
                            "text": reason,
                            "caller_id": caller_id,
                            "language": lang
                        })
                else:
                    # Function call tag was detected but parsing failed
                    logger.error("Failed to parse function call")
                    final_text = extract_user_message(full_response)
                    await safe_send(ws, {
                        "type": "completed",
                        "request_id": rid,
                        "text": final_text.strip() if final_text else "I'm having trouble understanding. Could you rephrase that?",
                        "disconnect": False,
                        "transfer": False,
                        "language": lang
                    })
                    
            else:
                # No function call - send normal completed message
                final_user_text = extract_user_message(full_response)
                
                # Check if model might have hallucinated a function call
                if any(tag in full_response.lower() for tag in ['<disconnect_call>', '<function>']):
                    logger.error("⚠️ Model generated malformed function call!")
                    logger.error(f"Full response: {full_response}")
                    logger.error("Consider using few-shot examples or fine-tuning")
                
                await safe_send(ws, {
                    "type": "completed",
                    "request_id": rid,
                    "text": final_user_text.strip(),
                    "disconnect": False,
                    "transfer": False,
                    "language": lang
                })
                    
        except asyncio.CancelledError:
            await safe_send(ws, {"type": "cancelled", "request_id": rid})
            raise
        except Exception as e:
            logger.exception(f"Error in generation: {e}")
            await safe_send(ws, {"type": "error", "request_id": rid, "error": str(e)})
        finally:
            ACTIVE.pop(rid, None)
            OWNER.pop(rid, None)

    task = asyncio.create_task(_job(), name=f"qwen-llm-{rid}")
    ACTIVE[rid] = task

async def handle_cancel(ws: WebSocketServerProtocol, msg: dict):
    """Handle request cancellation."""
    rid = msg.get("request_id") or ""
    t = ACTIVE.get(rid)

    if t and not t.done():
        t.cancel()
    else:
        owner = OWNER.get(rid)
        if owner is ws:
            await safe_send(ws, {"type": "cancelled", "request_id": rid})
        ACTIVE.pop(rid, None)
        OWNER.pop(rid, None)

async def ws_handler(ws: WebSocketServerProtocol):
    """Main WebSocket handler."""
    logger.info("Client connected")
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                await safe_send(ws, {"type": "error", "error": "invalid JSON"})
                continue

            mtype = msg.get("type")
            if mtype == "connect":
                await handle_connect(ws, msg)
            elif mtype == "generate":
                await handle_generate(ws, msg)
            elif mtype == "cancel":
                await handle_cancel(ws, msg)
            else:
                await safe_send(ws, {"type": "error", "error": f"unknown type: {mtype}"})
    finally:
        logger.info("Client disconnected")
        CALLER_IDS.pop(ws, None)
        
        to_cancel = [rid for rid, owner in OWNER.items() if owner is ws]
        for rid in to_cancel:
            task = ACTIVE.pop(rid, None)
            OWNER.pop(rid, None)
            if task and not task.done():
                task.cancel()


# --------------------- Main ---------------------

async def warmup_model(model: QwenTextModel, num_warmup_runs: int = 2) -> None:
    """
    Warm up the model by running dummy inference to load it into GPU memory.
    
    Args:
        model: The QwenTextModel instance to warm up
        num_warmup_runs: Number of warmup iterations (default: 2)
    """
    logger.info(f"Starting model warmup for {model.model_name}...")
    
    # Simple warmup messages
    warmup_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello"
        }
    ]
    
    for i in range(num_warmup_runs):
        try:
            # Run a short generation
            text_generated = ""
            async for delta in model.stream_generate(
                messages=warmup_messages,
                temperature=MIN_TEMP,
                max_tokens=10  # Short response for warmup
            ):
                text_generated += delta
            
            logger.info(f"Warmup run {i+1}/{num_warmup_runs} completed")
        except Exception as e:
            logger.error(f"Warmup run {i+1} failed: {e}")
    
    logger.info("Model warmup completed!")

async def initialize_model():
    """Initialize the Qwen model."""
    model_name = QWEN_MODEL_NAME
    
    global MODEL
    MODEL = QwenTextModel(model_name)
    
    # Warm up the model
    await warmup_model(MODEL)
    
    return MODEL

async def main():
    """Main entry point."""
    # Login to HuggingFace if token is provided
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    # Initialize model
    model_name = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    await initialize_model()
    logger.info(f"✓ Loaded model: {model_name}")
    logger.info(f"✓ Function calling enabled: function_call")

    # Start WebSocket server
    host = QWEN_HOST
    port = QWEN_PORT

    logger.info(f"🚀 Starting WS server on ws://{host}:{port}")
    async with websockets.serve(
        ws_handler,
        host,
        port,
        ping_interval=PING_INTERVAL,
        ping_timeout=PING_TIMEOUT,
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
        sys.exit(0)