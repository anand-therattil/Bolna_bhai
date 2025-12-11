#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Qwen LLM Server Client
A minimal client for communicating with the Qwen LLM WebSocket server.
"""


import sys
from pathlib import Path

# go up three levels: qwen.py -> qwen_server (1) -> qwen_model (2) -> Bolan_bhai (3)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import json
import uuid
import logging
import websockets
from config.loader import load_config
from typing import Optional, List, Dict, Any
from websockets.client import WebSocketClientProtocol


cfg = load_config()
QWEN_URL = cfg["urls"]["qwen_server"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)



class SimpleQwenClient:
    """Simple async client for Qwen LLM WebSocket server."""
    
    def __init__(self, server_url: str = "ws://localhost:8766", caller_id: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            server_url: WebSocket server URL
            caller_id: Unique identifier for this caller
        """
        self.server_url = server_url
        self.caller_id = caller_id or f"client_{uuid.uuid4().hex[:8]}"
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connected = False
    
    async def connect(self) -> bool:
        """Connect to the Qwen LLM server."""
        try:
            # Connect to WebSocket server
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=20
            )
            
            # Send initial connect message
            connect_msg = {
                "type": "connect",
                "caller_id": self.caller_id
            }
            await self.websocket.send(json.dumps(connect_msg))
            
            # Wait for connection confirmation
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "connected":
                self.connected = True
                logger.info(f"Connected with caller_id: {self.caller_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from server")
    
    async def generate_text(
        self,
        text: str = None,
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text response from the server.
        
        Args:
            text: Input text (simple mode)
            messages: Conversation messages (advanced mode)
            temperature: Generation temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to print streaming responses
            
        Returns:
            Final response dictionary
        """
        if not self.connected:
            raise ConnectionError("Not connected to server. Call connect() first.")
        
        request_id = str(uuid.uuid4())
        
        # Build request
        request = {
            "type": "generate",
            "request_id": request_id,
            "caller_id": self.caller_id,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if text:
            request["text"] = text
        elif messages:
            request["messages"] = messages
        else:
            raise ValueError("Either 'text' or 'messages' must be provided")
        
        # Send request
        await self.websocket.send(json.dumps(request))
        
        # Collect responses
        full_text = ""
        final_response = None
        
        while True:
            response = await self.websocket.recv()
            data = json.loads(response)
            
            response_type = data.get("type")
            
            if response_type == "started":
                if stream:
                    logger.info("Generating...")
            
            elif response_type == "partial":
                partial_text = data.get("text", "")
                if stream:
                    logger.info(partial_text)
                full_text = partial_text  # Keep updating with latest partial
            
            elif response_type in ["completed", "disconnect", "transfer"]:
                if stream and response_type == "partial":
                    logger.info("")  # New line after streaming
                
                # Get final text
                final_text = data.get("text", full_text)
                data["text"] = final_text
                final_response = data
                break
            
            elif response_type == "error":
                logger.error(f"Error: {data.get('error')}")
                final_response = data
                break
        
        return final_response
    
    async def send_and_receive(self, text: str, **kwargs) -> str:
        """
        Simple helper to send text and get response.
        
        Args:
            text: Input text
            **kwargs: Additional arguments for generate_text
            
        Returns:
            Response text string
        """
        response = await self.generate_text(text=text, **kwargs)
        return response.get("text", "")


async def main():
    """Example usage of the simple client."""
    
    # Create client
    client = SimpleQwenClient(
        server_url=QWEN_URL,
        caller_id="test_user"
    )
    
    try:
        # Connect to server
        if not await client.connect():
            print("Failed to connect to server")
            return
        
        # Example 1: Simple text generation
        logger.info("--- Example 1: Simple Text ---")
        response = await client.generate_text(
            text="Hello, I'm interested in know the sum of 100 + 100.",
            temperature=0.7,
            max_tokens=200
        )
        logger.info(f"Response: {response.get('text')}")
        logger.info(f"Type: {response.get('type')}")
        
        # # Example 2: With streaming
        # logger.info("\n--- Example 2: Streaming ---")
        # logger.info("Response: ", end="")
        # response = await client.generate_text(
        #     text="Tell me about your key features.",
        #     temperature=0.7,
        #     max_tokens=200,
        #     stream=True
        # )
        logger.info(f"\nType: {response.get('type')}")

        # # Example 3: Using messages for conversation
        # logger.info("\n--- Example 3: Conversation ---")
        # messages = [
        #     {"role": "system", "content": "You are Tara, a helpful sales agent."},
        #     {"role": "user", "content": "I need a solution for 50 agents."}
        # ]
        # response = await client.generate_text(
        #     messages=messages,
        #     temperature=0.7,
        #     max_tokens=200
        # )
        # logger.info(f"Response: {response.get('text')}")
        
        # Example 4: Trigger function call
        logger.info("\n--- Example 4: Function Call ---")
        response = await client.generate_text(
            text="What products do you sell at Myntra ?",
            temperature=0.7,
            max_tokens=300
        )
        logger.info(f"Response: {response.get('text')}")
        logger.info(f"Type: {response.get('type')}")

        # Check for special responses
        if response.get("type") == "disconnect":
            logger.info("The agent has ended the conversation.")
        elif response.get("type") == "transfer":
            logger.info("You're being transferred to another agent.")

        # Example 5: Simple helper method
        logger.info("\n--- Example 5: Simple Helper ---")
        text = await client.send_and_receive(
            "Do you integrate with Salesforce?",
            temperature=0.5,
            max_tokens=150
        )
        logger.info(f"Response: {text}")
        
    except Exception as e:
        logger.error(f"Error: {e}")

    finally:
        # Disconnect
        await client.disconnect()


async def minimal_example():
    """Minimal usage example."""
    
    # Create and connect
    client = SimpleQwenClient("ws://localhost:8766")
    
    if await client.connect():
        # Send message and get response
        response = await client.send_and_receive("Hello, how can you help me?")
        logger.info(f"Agent: {response}")
        
        # Disconnect
        await client.disconnect()


if __name__ == "__main__":
    logger.info("Starting Qwen Client Example...")
    logger.info("-" * 50)
    asyncio.run(main())  