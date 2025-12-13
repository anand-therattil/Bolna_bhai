import argparse
import sys

import uvicorn
from bot import bot
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from loguru import logger
from pipecat.runner.types import RunnerArguments

# Load environment variables
load_dotenv(override=True)

app = FastAPI()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Twilio Voice Bot Server"}


@app.post("/twiml")
async def twiml():
    """TwiML endpoint that Twilio calls when a phone call is initiated.
    
    This endpoint should be configured in your Twilio phone number settings
    as the Voice webhook URL.
    """
    # Get the host and port from environment or use defaults
    host = "your-server-domain.com"  # Replace with your actual domain or ngrok URL
    
    twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/ws" />
    </Connect>
</Response>"""
    
    return Response(content=twiml_response, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for Twilio Media Streams.
    
    This is where Twilio sends the audio stream during a phone call.
    """
    await websocket.accept()
    logger.info("Twilio WebSocket connection established")
    
    # Create runner arguments
    runner_args = RunnerArguments(
        websocket=websocket,
        handle_sigint=False  # FastAPI/Uvicorn handles this
    )
    
    try:
        # Run the bot
        await bot(runner_args)
    except Exception as e:
        logger.error(f"Error running bot: {e}")
    finally:
        logger.info("Twilio WebSocket connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twilio Voice Bot Server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE")
    else:
        logger.add(sys.stderr, level="DEBUG")

    uvicorn.run(app, host=args.host, port=args.port)