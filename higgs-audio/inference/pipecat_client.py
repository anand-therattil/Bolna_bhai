import asyncio
import json
import base64
import wave
import numpy as np
from pipecat.frames.frames import Frame, AudioRawFrame, TextFrame, EndFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from websockets.client import connect as websocket_connect


class HiggsWebSocketClient(FrameProcessor):
    """WebSocket client that sends text and receives audio from Higgs server."""
    
    def __init__(self, uri: str):
        super().__init__()
        self.uri = uri
        self.websocket = None
        self._connected = False
        
    async def start(self):
        """Connect to the WebSocket server."""
        if self._connected:
            return
        print(f"Connecting to {self.uri}...")
        self.websocket = await websocket_connect(self.uri)
        self._connected = True
        print("✓ Connected to server")
    
    async def stop(self):
        """Disconnect from the WebSocket server."""
        if self.websocket and self._connected:
            await self.websocket.close()
            self._connected = False
            print("✓ Disconnected from server")
    
    async def _receive_audio(self):
        """Receive a single audio response from the server."""
        try:
            print("Waiting for audio response...")
            message = await self.websocket.recv()
            print("✓ Received response from server")
            
            # Parse the response
            data = json.loads(message)
            audio_base64 = data["audio"]
            sample_rate = data["sampling_rate"]
            
            # Decode the audio (it's a complete WAV file)
            audio_wav_bytes = base64.b64decode(audio_base64)
            
            print(f"  Audio WAV: {len(audio_wav_bytes)} bytes at {sample_rate}Hz")
            
            # Extract PCM data from WAV (skip 44-byte header)
            pcm_data = audio_wav_bytes[44:]
            
            # IMPORTANT: Don't create AudioRawFrame, just store the data
            # We'll pass it to AudioSaver directly
            return pcm_data, sample_rate
            
        except Exception as e:
            print(f"Error receiving audio: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames from the pipeline."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, StartFrame):
            await self.start()
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, TextFrame):
            # Send text to server
            print(f"Sending text to server: '{frame.text}'")
            message = json.dumps({"text": frame.text})
            await self.websocket.send(message)
            print("✓ Text sent to server")
            
            # Receive the audio response
            audio_data, sample_rate = await self._receive_audio()
            
            if audio_data:
                # Store in the frame for AudioSaver to pick up
                # Create a custom frame that won't trigger observers
                class AudioDataFrame(Frame):
                    def __init__(self, audio_data, sample_rate):
                        super().__init__()
                        self.audio_data = audio_data
                        self.sample_rate = sample_rate
                
                audio_frame = AudioDataFrame(audio_data, sample_rate)
                await self.push_frame(audio_frame, direction)
                print("✓ Audio data passed to pipeline")
            
        elif isinstance(frame, EndFrame):
            await self.stop()
            await self.push_frame(frame, direction)
            
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)


class AudioSaver(FrameProcessor):
    """Save audio frames to a WAV file."""
    
    def __init__(self, output_file: str = "output.wav"):
        super().__init__()
        self.output_file = output_file
        self.audio_chunks = []
        self.sample_rate = None
        print(f"AudioSaver initialized - will save to {output_file}")
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Handle our custom AudioDataFrame
        if hasattr(frame, 'audio_data'):
            print(f"✓ AudioSaver received chunk: {len(frame.audio_data)} bytes")
            self.audio_chunks.append(frame.audio_data)
            if self.sample_rate is None:
                self.sample_rate = frame.sample_rate
                print(f"  Sample rate: {self.sample_rate}Hz")
            # Don't push this frame further
            
        elif isinstance(frame, EndFrame):
            if self.audio_chunks:
                print("\nSaving audio to file...")
                self._save_audio()
            else:
                print("⚠ No audio chunks received")
            await self.push_frame(frame, direction)
            
        else:
            await self.push_frame(frame, direction)
    
    def _save_audio(self):
        """Combine chunks and save to WAV file."""
        # Concatenate all audio chunks
        audio_data = b''.join(self.audio_chunks)
        
        # The audio is already in int16 PCM format from the WAV
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Save as WAV
        with wave.open(self.output_file, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate or 24000)
            wav_file.writeframes(audio_data)
        
        file_size = len(audio_data)
        duration = len(audio_array) / (self.sample_rate or 24000)
        print(f"✓ Audio saved to {self.output_file}!")
        print(f"  Sample rate: {self.sample_rate}Hz")
        print(f"  Duration: {duration:.2f}s")
        print(f"  File size: {file_size} bytes")


class TextInjector(FrameProcessor):
    """Inject a text frame into the pipeline after startup."""
    
    def __init__(self, text: str, delay: float = 0.5):
        super().__init__()
        self.text = text
        self.delay = delay
        self._injected = False
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # After StartFrame, inject our text
        if isinstance(frame, StartFrame) and not self._injected:
            self._injected = True
            await self.push_frame(frame, direction)
            
            # Wait a bit for connection
            await asyncio.sleep(self.delay)
            
            # Inject text frame
            print(f"\n>>> Injecting text: '{self.text}'\n")
            text_frame = TextFrame(text=self.text)
            await self.push_frame(text_frame, direction)
        else:
            await self.push_frame(frame, direction)


async def generate_audio(text: str, server_uri: str = "ws://localhost:8000", output_file: str = "output.wav"):
    """Generate audio using Pipecat pipeline."""
    
    print("=" * 60)
    print("Pipecat Client - Higgs Audio Generation")
    print("=" * 60)
    print()
    
    # Create processors
    text_injector = TextInjector(text, delay=0.5)
    websocket_client = HiggsWebSocketClient(server_uri)
    audio_saver = AudioSaver(output_file)
    
    # Create pipeline
    pipeline = Pipeline([
        text_injector,      # Inject text after startup
        websocket_client,   # Send to server and receive audio
        audio_saver,        # Save audio to file
    ])
    
    # Create task
    task = PipelineTask(pipeline)
    
    # Create runner
    runner = PipelineRunner()
    
    # Queue start and end frames
    async def control_flow():
        # Start the pipeline
        await task.queue_frame(StartFrame())
        
        # Wait for processing (enough time for audio generation)
        await asyncio.sleep(20)
        
        # End the pipeline
        print("\n>>> Sending EndFrame to close pipeline")
        await task.queue_frame(EndFrame())
    
    # Run everything
    try:
        await asyncio.gather(
            runner.run(task),
            control_flow()
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)


if __name__ == "__main__":
    # Your text prompt
    text = "what is this? How can something like this happen today? I am not angry with this i am just sad"
    
    # Run the pipeline
    asyncio.run(generate_audio(text, output_file="output.wav"))