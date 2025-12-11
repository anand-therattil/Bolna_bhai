#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple GPT-2 LLM WebSocket Client (for vLLM streaming)
"""

import sys
from pathlib import Path

# Same pattern as Qwen client (adjust if needed)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import json
import uuid
import logging
import websockets
# from config.loader import load_config
from typing import Optional, Dict, Any, List
from websockets.client import WebSocketClientProtocol

# cfg = load_config()
# GPT2_URL = cfg["urls"]["gpt2_server"]  # ADD gpt2_server entry in your config
GPT2_URL = "ws://localhost:8764"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleGPT2Client:
    """Simple async client for GPT-2 WebSocket server."""

    def __init__(self, server_url="ws://localhost:9999", caller_id=None):
        self.server_url = server_url
        self.caller_id = caller_id or f"gpt2_client_{uuid.uuid4().hex[:8]}"
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connected = False

    async def connect(self) -> bool:
        """Connect to GPT-2 server."""
        try:
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=20
            )

            await self.websocket.send(json.dumps({
                "type": "connect",
                "caller_id": self.caller_id
            }))

            response = await self.websocket.recv()
            data = json.loads(response)

            if data.get("type") == "connected":
                self.connected = True
                logger.info(f"Connected to GPT-2 server as {self.caller_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from GPT-2 server")

    async def generate_text(
        self,
        text: str,
        temperature: float = 0.7,
        max_tokens: int = 200,
        stream: bool = False
    ) -> Dict[str, Any]:

        if not self.connected:
            raise ConnectionError("Not connected — call connect() first.")

        request_id = str(uuid.uuid4())

        # Build message to GPT-2 server
        request = {
            "type": "generate",
            "request_id": request_id,
            "caller_id": self.caller_id,
            "text": text,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        await self.websocket.send(json.dumps(request))

        full_text = ""
        final_response = None

        while True:
            raw = await self.websocket.recv()
            data = json.loads(raw)
            event = data.get("type")

            if event == "started":
                if stream:
                    logger.info("Generation started...")

            elif event == "partial":
                partial = data.get("text", "")
                full_text = partial
                if stream:
                    print(partial, end="", flush=True)

            elif event == "completed":
                final_text = data.get("text", full_text)
                data["text"] = final_text
                final_response = data
                break

            elif event == "error":
                logger.error(f"Error: {data.get('error')}")
                final_response = data
                break

        return final_response

    async def send_and_receive(self, text: str, **kwargs) -> str:
        response = await self.generate_text(text=text, **kwargs)
        return response.get("text", "")


# --------------------- EXAMPLE USAGE -----------------------------

async def main():
    client = SimpleGPT2Client(server_url=GPT2_URL, caller_id="test_user")

    if not await client.connect():
        print("Failed to connect to GPT-2 server")
        return

    logger.info("\n--- GPT-2 Simple Text Generation ---")
    resp = await client.generate_text(
        text="Explain the significance of blockchain.",
        temperature=0.7,
        max_tokens=150
    )
    logger.info("Response:")
    logger.info(resp["text"])

    logger.info("\n--- GPT-2 Streaming Generation ---")
    await client.generate_text(
        text="Write a short poem about AI.",
        temperature=0.7,
        max_tokens=150,
        stream=True
    )

    await client.disconnect()


async def minimal_example():
    client = SimpleGPT2Client("ws://localhost:9999")
    if await client.connect():
        reply = await client.send_and_receive("Hello GPT-2!")
        print("GPT-2:", reply)
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
