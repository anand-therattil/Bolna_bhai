#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-2 WebSocket server wrapping vLLM AsyncLLMEngine.
...
"""

import asyncio
import json
import logging
import signal
import time
from typing import Dict, Any
import multiprocessing

import websockets
from websockets.server import WebSocketServerProtocol

from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

# ---------------------------
WS_HOST = "localhost"
WS_PORT = 8763
MODEL_PATH = "/Users/cmi_10128/Desktop/documents/projects/Bolna_bhai/gpt_model/gpt2-finetuned-polymarket-final"

ENGINE_ARGS = AsyncEngineArgs(
    model=MODEL_PATH,
    dtype="float16",
    gpu_memory_utilization=0.025,
)

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 200
# ---------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gpt2-ws-server")

# Global LLM engine reference — initialize later inside main()
llm = None

_running_generations: Dict[str, Dict[str, asyncio.Task]] = {}


async def handle_generate_message(ws: WebSocketServerProtocol, msg: Dict[str, Any]) -> None:
    """
    Handle a 'generate' message from the client and stream partial/completed events.
    Expected msg fields: request_id, caller_id, text, temperature, top_p, max_tokens
    """
    final_text = ""
    request_id = msg.get("request_id") or f"request-{time.time()}"
    caller_id = msg.get("caller_id", "unknown")
    text = msg.get("text", "")
    temperature = float(msg.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(msg.get("top_p", DEFAULT_TOP_P))
    max_tokens = int(msg.get("max_tokens", DEFAULT_MAX_TOKENS))

    client_key = f"{id(ws)}"
    if client_key not in _running_generations:
        _running_generations[client_key] = {}

    logger.info(f"[{caller_id}] Received generate request {request_id}")

    # Prepare prompt same as your inference code
    prompt = f"<|instruction|>{text}<|response|>"

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    async def _run_generation():
        nonlocal final_text
        try:
            # Inform client generation started
            await ws.send(json.dumps({"type": "started", "request_id": request_id}))
            logger.debug(f"[{caller_id}] started {request_id}")

            results_generator = llm.generate(prompt, params, request_id)

            final_text = ""
            async for request_output in results_generator:
                # request_output may contain multiple outputs — take the first as in your script
                outputs = getattr(request_output, "outputs", None)
                if not outputs:
                    continue

                raw = outputs[0].text or ""

                # Always extract ONLY new tokens (delta)
                delta = raw[len(final_text):]

                # Update accumulated
                final_text = raw

                # Clean delta only (not entire chunk)
                delta = delta.replace("<|endoftext|>", "")
                delta = delta.replace("<|response|>", "")
                delta = delta



                # Send partial update with cumulative text (not delta)
                if delta:  # Only send if there's new content
                    payload = {
                        "type": "partial",
                        "request_id": request_id,
                        "caller_id": caller_id,
                        "text": final_text  # Send cumulative text, not delta
                    }
                    await ws.send(json.dumps(payload))

            # After the generator completes, send completed with final text
            completed_payload = {
                "type": "completed",
                "request_id": request_id,
                "caller_id": caller_id,
                "text": final_text
            }
            await ws.send(json.dumps(completed_payload))
            logger.info(f"[{caller_id}] completed {request_id}, complete payload: {completed_payload}")

        except asyncio.CancelledError:
            logger.info(f"[{caller_id}] Generation cancelled: {request_id}")
            # Optionally inform client of cancellation as an error or special event
            try:
                await ws.send(json.dumps({
                    "type": "error",
                    "request_id": request_id,
                    "caller_id": caller_id,
                    "error": "generation_cancelled"
                }))
            except Exception:
                pass
        except Exception as e:
            logger.exception(f"Error during generation {request_id}: {e}")
            try:
                await ws.send(json.dumps({
                    "type": "error",
                    "request_id": request_id,
                    "caller_id": caller_id,
                    "error": str(e)
                }))
            except Exception:
                pass
        finally:
            # cleanup — remove from running map
            _running_generations.get(client_key, {}).pop(request_id, None)

    # Start the generation task (so the handler can still accept other messages e.g. cancel)
    task = asyncio.create_task(_run_generation())
    _running_generations[client_key][request_id] = task


async def handler(ws: WebSocketServerProtocol):
    """
    Per-connection WebSocket handler.
    Waits for a connect message first, then processes generate/cancel messages.
    """
    client_key = f"{id(ws)}"
    _running_generations[client_key] = {}

    caller_id = "unknown"
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"type": "error", "error": "invalid_json"}))
                continue

            mtype = msg.get("type")
            if mtype == "connect":
                caller_id = msg.get("caller_id", caller_id)
                await ws.send(json.dumps({"type": "connected"}))
                logger.info(f"[{caller_id}] connected (ws_id={client_key})")

            elif mtype == "generate":
                # Start generation (streams partial updates)
                await handle_generate_message(ws, msg)

            elif mtype == "cancel":
                req_id = msg.get("request_id")
                if not req_id:
                    await ws.send(json.dumps({"type": "error", "error": "missing_request_id"}))
                    continue
                task = _running_generations.get(client_key, {}).get(req_id)
                if task and not task.done():
                    task.cancel()
                    await ws.send(json.dumps({"type": "cancelled", "request_id": req_id}))
                else:
                    await ws.send(json.dumps({"type": "error", "request_id": req_id, "error": "no_active_request"}))

            else:
                await ws.send(json.dumps({"type": "error", "error": f"unknown_message_type:{mtype}"}))

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[{caller_id}] connection closed")
    finally:
        # Cancel any running tasks for this websocket
        tasks = list(_running_generations.get(client_key, {}).values())
        for t in tasks:
            if not t.done():
                t.cancel()
        _running_generations.pop(client_key, None)


async def main():
    global llm
    
    # On certain platforms it's harmless to call this; it helps when freezing to exe.
    multiprocessing.freeze_support()

    stop = asyncio.Event()

    async def _stop_signal():
        logger.info("SIGTERM/SIGINT received. Stopping server...")
        stop.set()

    # Initialize vLLM engine here (inside the guarded main)
    logger.info("Initializing vLLM Async engine...")
    try:
        llm = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
    except Exception as e:
        logger.exception("Failed to initialize vLLM engine: %s", e)
        raise

    logger.info("vLLM engine initialized.")

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(_stop_signal()))
    loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(_stop_signal()))

    logger.info(f"Starting GPT-2 WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(handler, WS_HOST, WS_PORT, max_size=None, ping_interval=20, ping_timeout=20):
        await stop.wait()

    logger.info("Server shutting down...")



if __name__ == "__main__":
    asyncio.run(main())
