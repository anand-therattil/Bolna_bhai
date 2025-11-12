from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torchaudio
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Suppress *all* warnings and onnxruntime chatter
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
os.environ["ORT_LOGGING_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def device_present() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# ASR class (silent)
# ---------------------------------------------------------------------------
class ASR:  # noqa: D101
    TARGET_SR = 16_000

    def __init__(
        self,
        model_name: str = "ai4bharat/indic-conformer-600m-multilingual",
    ) -> None:
        self.device = device_present()
        self.asr = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map={"": self.device},
            trust_remote_code=True,
        ).eval()

    def cleanup(self):
        """Clean up model resources"""
        try:
            if hasattr(self, 'asr'):
                del self.asr
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.cleanup()

    # -------------------------------------------------------------------
    # Public entry
    # -------------------------------------------------------------------
    def __call__(
        self,
        audio: Union[str, np.ndarray, torch.Tensor, Path],
        *,
        lang: str = "hi",
        sr: int | None = None,
        timestamps: List | None=None
    ) -> Dict:
        wav_t, wav_np, sr = self._load_audio(audio, sr)
        segments = timestamps
        chunks: List[Dict] = []
        full_text_parts: List[str] = []
        for seg in segments:
            start_time = seg['start']
            end_time = seg['end']

            clip = self._crop(wav_t, start_time, end_time, sr)
            text = self._decode(clip, lang)
            if text:
                chunks.append({"text": text, "timestamp": [start_time/self.TARGET_SR, end_time/self.TARGET_SR]})
                full_text_parts.append(text)
        return {"text": " ".join(full_text_parts).strip(), "chunks": chunks}


    # -------------------------------------------------------------------
    # Audio I/O
    # -------------------------------------------------------------------
    def _load_audio(
        self, audio: Union[str, np.ndarray, torch.Tensor, Path], sr: int | None
    ) -> Tuple[torch.Tensor, np.ndarray, int]:
        if isinstance(audio, (np.ndarray, torch.Tensor)):
            wav_t = torch.as_tensor(audio) if isinstance(audio, np.ndarray) else audio.detach()
            if wav_t.ndim == 1:
                wav_t = wav_t.unsqueeze(0)
            if sr is None:
                raise ValueError("Provide sr for ndarray/tensor input")
            cur_sr = sr
        else:
            wav_t, cur_sr = torchaudio.load(str(audio))
        if wav_t.shape[0] > 1:
            wav_t = wav_t.mean(dim=0, keepdim=True)
        if cur_sr != self.TARGET_SR:
            wav_t = torchaudio.transforms.Resample(cur_sr, self.TARGET_SR)(wav_t)
            cur_sr = self.TARGET_SR
        return wav_t.to(self.device), wav_t.squeeze().cpu().numpy(), cur_sr

    @staticmethod
    def _crop(wav: torch.Tensor, start: float, end: float, sr: int) -> torch.Tensor:
        # s = int(max(0, start) * sr)
        # e = int(min(end, wav.shape[1] / sr) * sr)
        s, e = int(max(0, start)), int(min(end, wav.shape[1]))
        return wav[:, s:e]

    # -------------------------------------------------------------------
    # ASR decode
    # -------------------------------------------------------------------
    def _decode(self, clip: torch.Tensor, lang: str) -> str:  # noqa: D401
        if clip.shape[1] < int(0.1 * self.TARGET_SR):
            return ""
        try:
            text = self.asr(clip, lang, "rnnt").strip()
            if text:
                return text
        except Exception:
            pass
        try:
            return self.asr(clip, lang, "ctc").strip()
        except Exception:
            return ""
