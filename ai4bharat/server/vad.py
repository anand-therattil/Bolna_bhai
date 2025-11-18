import torch
import numpy as np

class VAD:
    def __init__(self):
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils
    
    def has_speech(self, audio_array):
        
        if audio_array is None:
            return False

        try:
            # Load audio (Silero VAD expects 16kHz)
            
            # Get speech timestamps
            speech_timestamps = []
            speech_timestamps = self.get_speech_timestamps(
                audio_array, 
                self.model,
                sampling_rate=16000,
                threshold=0.3,  # Adjust sensitivity (0.0-1.0)
                min_speech_duration_ms=200,  # Minimum speech chunk duration
                min_silence_duration_ms=100  # Minimum silence between chunks
            )
            return speech_timestamps
        except Exception as e:
            print(f"Silero VAD error: {e}")
            return False