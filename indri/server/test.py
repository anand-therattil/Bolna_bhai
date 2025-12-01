import torch
import torchaudio
from transformers import pipeline

model_id = '11mlabs/indri-0.1-124m-tts'
task = 'indri-tts'

pipe = pipeline(
    task,
    model=model_id,
    device=torch.device('cpu'), # Update this based on your hardware,
    trust_remote_code=True
)

output = pipe(['मेरा नाम आनंद है'], speaker = '[spkr_63]')

torchaudio.save('output.wav', output[0]['audio'][0], sample_rate=24000)
