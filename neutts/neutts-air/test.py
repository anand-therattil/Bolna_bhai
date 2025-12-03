from neuttsair.neutts import NeuTTSAir
import soundfile as sf

tts = NeuTTSAir( backbone_repo="neuphonic/neutts-air", backbone_device="cpu", codec_repo="neuphonic/neucodec", codec_device="cpu")
input_text = "My name is Dave, and um, I'm from London."

ref_text = "samples/dave.txt"
ref_audio_path = "samples/dave.wav"

ref_text = open(ref_text, "r").read().strip()
ref_codes = tts.encode_reference(ref_audio_path)

wav = tts.infer(input_text, ref_codes, ref_text)
sf.write("test.wav", wav, 24000)
