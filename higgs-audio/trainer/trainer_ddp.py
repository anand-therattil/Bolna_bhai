#!/usr/bin/env python3
"""
Higgs Audio v2 Training Script with LoRA Support and Task-specific Training
Based on the Higgs Audio v2 architecture from Boson AI

FINAL CORRECTED VERSION for Multi-GPU Training with torchrun and bfloat16
"""

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import librosa
import re

# 尝试导入 Higgs Audio 相关模块
try:
    from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator, HiggsAudioBatchInput
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
    from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
    HIGGS_AVAILABLE = True
except ImportError:
    HIGGS_AVAILABLE = False
    logging.warning("Higgs Audio modules not available. Using fallback implementation.")

    # A simple fallback class is sufficient. It does NOT need a .to() method.
    class ChatMLDatasetSample:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加常量定义
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


class ExtendedHiggsAudioBatchInput:
    """
    Extended HiggsAudioBatchInput with __len__ method for Trainer compatibility
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        """Return the batch size based on input_ids"""
        if hasattr(self, 'input_ids') and self.input_ids is not None:
            return self.input_ids.shape[0]
        else:
            return 0

    def __getitem__(self, key):
        """Allow dictionary-style access for compatibility"""
        return getattr(self, key)

    def __contains__(self, key):
        """Check if attribute exists"""
        return hasattr(self, key)

    def keys(self):
        """Return all attribute names for compatibility"""
        return [attr for attr in dir(self) if not attr.startswith('_') and not callable(getattr(self, attr))]


class ExtendedHiggsAudioSampleCollator:
    """
    Extended collator that returns our custom batch input class
    """
    def __init__(self, **kwargs):
        if HIGGS_AVAILABLE:
            # Assuming HiggsAudioSampleCollator can be initialized with these kwargs
            self.base_collator = HiggsAudioSampleCollator(**kwargs)
        else:
            # Fallback collator
            self.pad_token_id = kwargs.get('pad_token_id', 0)

    def __call__(self, batch: List[ChatMLDatasetSample]):
        if HIGGS_AVAILABLE and hasattr(self, 'base_collator'):
            batch_input = self.base_collator(batch)
            label_audio_ids = batch_input.audio_out_ids
            extended_batch = ExtendedHiggsAudioBatchInput(
                input_ids=batch_input.input_ids,
                attention_mask=batch_input.attention_mask,
                audio_features=batch_input.audio_features,
                audio_feature_attention_mask=batch_input.audio_feature_attention_mask,
                audio_out_ids=batch_input.audio_out_ids,
                audio_out_ids_start=batch_input.audio_out_ids_start,
                audio_out_ids_start_group_loc=batch_input.audio_out_ids_start_group_loc,
                audio_in_ids=batch_input.audio_in_ids,
                audio_in_ids_start=batch_input.audio_in_ids_start,
                label_ids=batch_input.label_ids,
                label_audio_ids=label_audio_ids,
                reward=batch_input.reward,
            )
            return extended_batch
        else:
            # Fallback implementation
            input_ids_list, attention_mask_list, label_ids_list = [], [], []
            for sample in batch:
                input_ids_list.append(sample.input_ids)
                attention_mask_list.append(torch.ones_like(sample.input_ids))
                label_ids_list.append(getattr(sample, 'label_ids', sample.input_ids))

            max_len = max(len(ids) for ids in input_ids_list)
            padded_input_ids, padded_attention_mask, padded_label_ids = [], [], []

            for i in range(len(input_ids_list)):
                pad_len = max_len - len(input_ids_list[i])
                padded_input_ids.append(torch.cat([input_ids_list[i], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]))
                padded_attention_mask.append(torch.cat([attention_mask_list[i], torch.zeros(pad_len, dtype=torch.long)]))
                padded_label_ids.append(torch.cat([label_ids_list[i], torch.full((pad_len,), -100, dtype=torch.long)]))

            return ExtendedHiggsAudioBatchInput(
                input_ids=torch.stack(padded_input_ids),
                attention_mask=torch.stack(padded_attention_mask),
                label_ids=torch.stack(padded_label_ids),
                audio_features=None, audio_feature_attention_mask=None,
                audio_out_ids=None, audio_out_ids_start=None, audio_out_ids_start_group_loc=None,
                audio_in_ids=None, audio_in_ids_start=None, label_audio_ids=None, reward=None,
            )

def normalize_chinese_punctuation(text):
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!", "（": "(", "）": ")",
        "【": "[", "】": "]", "《": "<", "》": ">", "“": '"', "”": '"', "‘": "'", "’": "'",
        "、": ",", "——": "-", "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text

def _build_system_message_with_audio_prompt(system_message):
    contents = []
    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN):]
    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    return Message(role="system", content=contents)


class HiggsAudioDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer: AutoTokenizer,
        audio_tokenizer,
        task_type: str = "zero_shot_voice_cloning",
        sample_rate: int = 24000,
        use_metadata: bool = True,
        ref_audio_in_system_message: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.task_type = task_type
        self.sample_rate = sample_rate
        self.use_metadata = use_metadata
        self.ref_audio_in_system_message = ref_audio_in_system_message

        valid_tasks = ["zero_shot_voice_cloning", "single_speaker_smart_voice", "multi_speaker_smart_voice", "multi_speaker_voice_cloning"]
        if self.task_type not in valid_tasks:
            raise ValueError(f"Invalid task_type: {self.task_type}. Must be one of {valid_tasks}")

        self.actual_num_codebooks = self._detect_codebook_size()

        if use_metadata and (self.data_dir / "metadata.json").exists():
            self.samples = self._load_samples_from_metadata()
        else:
            logger.warning(f"metadata.json not found in {data_dir}. Scanning directory instead.")
            self.samples = self._load_samples_from_directory()
            
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {data_dir} for task '{self.task_type}'.")
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir} for task: {self.task_type}")

    def _detect_codebook_size(self) -> int:
        try:
            audio_files = list(self.data_dir.glob("*.wav")) + list(self.data_dir.glob("*.mp3"))
            if audio_files:
                test_tokens = self._encode_audio_tokens(str(audio_files[0]))
                if test_tokens is not None and test_tokens.dim() == 2:
                    logger.info(f"Detected {test_tokens.shape[0]} codebooks from audio tokenizer.")
                    return test_tokens.shape[0]
        except Exception as e:
            logger.warning(f"Could not auto-detect codebook size: {e}. Falling back to default.")
        default_size = getattr(self.audio_tokenizer, 'codebook_size', 8)
        logger.info(f"Using default codebook size: {default_size}")
        return default_size

    def _load_samples_from_metadata(self) -> List[Dict]:
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f).get("samples", [])
        for sample in metadata:
            sample["audio_file"] = str(self.data_dir / sample["audio_file"])
            if "transcript_file" in sample:
                sample["transcript_file"] = str(self.data_dir / sample["transcript_file"])
        return metadata

    def _load_samples_from_directory(self) -> List[Dict]:
        samples = []
        audio_files = list(self.data_dir.glob("*.wav")) + list(self.data_dir.glob("*.mp3"))
        for audio_path in audio_files:
            transcript_path = audio_path.with_suffix('.txt')
            if transcript_path.exists():
                samples.append({"audio_file": str(audio_path), "transcript_file": str(transcript_path), "audio_id": audio_path.stem})
        return samples

    def _load_audio_waveform(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate: waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
            return waveform.squeeze(0), self.sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return torch.zeros(1), self.sample_rate

    def _encode_audio_tokens(self, audio_path: str) -> Optional[torch.Tensor]:
        if not self.audio_tokenizer: return None
        try:
            return self.audio_tokenizer.encode(audio_path)
        except Exception as e:
            logger.error(f"Failed to encode audio {audio_path}: {e}")
            return None

    def _normalize_transcript(self, transcript: str) -> str:
        transcript = normalize_chinese_punctuation(transcript)
        transcript = transcript.replace("(", " ").replace(")", " ")
        transcript = transcript.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")
        for tag, replacement in [("[laugh]", "<SE>[Laughter]</SE>"), ("[cough]", "<SE>[Cough]</SE>")]:
            transcript = transcript.replace(tag, replacement)
        lines = transcript.split("\n")
        transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
        if transcript and not any(transcript.endswith(c) for c in ".!?,\";'</SE_e></SE>"):
            transcript += "."
        return transcript.strip()

    def _get_scene_description(self, sample: Dict) -> str:
        return f"Audio is recorded from {sample.get('scene', 'a quiet room').replace('_', ' ')}."

    def _detect_speaker_tags(self, transcript: str) -> List[str]:
        return sorted(set(re.findall(r"\[(SPEAKER\d+)\]", transcript)))

    def _create_messages_for_task(self, sample: Dict, transcript: str) -> List[Message]:
        messages = []
        speaker_tags = self._detect_speaker_tags(transcript)
        scene_prompt = self._get_scene_description(sample) if self.use_metadata else None

        if self.task_type == "zero_shot_voice_cloning":
            ref_audio = sample.get("ref_audio_file")
            if not ref_audio:
                return [Message(role="system", content="Generate audio following instruction.")]
            ref_audio_path = str(self.data_dir / ref_audio)
            ref_transcript = sample.get("ref_transcript", "This is a voice sample for cloning.")
            if self.ref_audio_in_system_message:
                system_content = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n\nSPEAKER0: {AUDIO_PLACEHOLDER_TOKEN}\n<|scene_desc_end|>"
                messages.append(_build_system_message_with_audio_prompt(system_content))
            else:
                messages.append(Message(role="system", content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"))
                messages.append(Message(role="user", content=ref_transcript))
                messages.append(Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)))
        elif self.task_type == "multi_speaker_voice_cloning":
            ref_speakers = sample.get("ref_speakers")
            if not ref_speakers or not isinstance(ref_speakers, list):
                return [Message(role="system", content=MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE)]
            for i, ref_info in enumerate(ref_speakers):
                messages.append(Message(role="user", content=f"{ref_info.get('speaker_tag', f'[SPEAKER{i}]')} {ref_info.get('ref_transcript', 'This is a voice sample.')}"))
                messages.append(Message(role="assistant", content=AudioContent(audio_url=str(self.data_dir / ref_info['ref_audio_file']))))
            messages.insert(0, Message(role="system", content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"))
        elif self.task_type == "multi_speaker_smart_voice" and len(speaker_tags) > 1:
            speaker_desc_l = [f"{tag}: {'feminine' if i % 2 == 0 else 'masculine'}" for i, tag in enumerate(speaker_tags)]
            scene_desc = f"{scene_prompt}\n\n" + "\n".join(speaker_desc_l) if scene_prompt else "\n".join(speaker_desc_l)
            messages.append(Message(role="system", content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"))
        else:
            content = "Generate audio following instruction."
            if scene_prompt: content += f"\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"
            messages.append(Message(role="system", content=content))
        return messages

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        sample = self.samples[idx]
        try:
            with open(sample["transcript_file"], 'r', encoding='utf-8') as f:
                transcript = self._normalize_transcript(f.read().strip())

            messages = self._create_messages_for_task(sample, transcript)
            messages.append(Message(role="user", content=transcript))
            messages.append(Message(role="assistant", content=AudioContent(audio_url=sample["audio_file"])))

            input_tokens, label_tokens, audio_contents, audio_label_contents, _ = prepare_chatml_sample(ChatMLSample(messages=messages), self.tokenizer)

            context_audio_tokens = [self._encode_audio_tokens(ac.audio_url) for ac in (audio_contents or []) if ac.audio_url]
            context_audio_tokens = [t for t in context_audio_tokens if t is not None]

            label_audio_tokens = [self._encode_audio_tokens(alc.audio_url) for alc in (audio_label_contents or []) if alc.audio_url]
            label_audio_tokens = [t for t in label_audio_tokens if t is not None]

            if context_audio_tokens:
                audio_ids_concat = torch.cat(context_audio_tokens, dim=1)
                audio_ids_start = torch.tensor([0] + [t.shape[1] for t in context_audio_tokens[:-1]], dtype=torch.long).cumsum(0)
            else:
                audio_ids_concat = torch.zeros((self.actual_num_codebooks, 0), dtype=torch.long)
                audio_ids_start = torch.tensor([0], dtype=torch.long)

            label_audio_ids = torch.cat(label_audio_tokens, dim=1) if label_audio_tokens else None
            waveform, sr = self._load_audio_waveform(sample["audio_file"])

            dataset_sample = ChatMLDatasetSample(
                input_ids=torch.tensor(input_tokens, dtype=torch.long),
                label_ids=torch.tensor(label_tokens, dtype=torch.long),
                audio_ids_concat=audio_ids_concat,
                audio_ids_start=audio_ids_start,
                label_audio_ids=label_audio_ids,
                audio_waveforms_concat=waveform,
                audio_waveforms_start=torch.tensor([0], dtype=torch.long),
                audio_sample_rate=torch.tensor([sr], dtype=torch.float32),
                audio_speaker_indices=torch.tensor([0], dtype=torch.long),
            )
            
            return dataset_sample

        except Exception as e:
            logger.error(f"Error processing sample at index {idx} (ID: {sample.get('audio_id', 'N/A')}): {e}", exc_info=True)
            return self.__getitem__((idx + 1) % len(self))


class HiggsAudioModelWrapper(nn.Module):
    """Wrapper for Higgs Audio v2 model to enable training"""
    def __init__(self, model_path: str, device: str = 'cuda:0', args=None):
        super().__init__()
        if HIGGS_AVAILABLE:
            self.model = HiggsAudioModel.from_pretrained(
                config=HiggsAudioConfig.from_pretrained(model_path),
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch.bfloat16,
                device_map={'': device},
            )
            self.config = self.model.config
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.config = self.model.config
        
        self.model = self.model.to(device)
        
        if args:
            if args.freeze_audio_tower: self.model.freeze_audio_tower()
            if args.freeze_audio_encoder_proj: self.model.freeze_audio_encoder_proj()
            if args.freeze_llm: self.model.freeze_llm()

    @property
    def device(self):
        return self.model.device
          
    def forward(self, **kwargs):
        # --- 开始终极修复 ---
        # 在数据进入底层模型前，强制将所有浮点张量的类型与模型权重对齐。
        # 这是解决顽固类型不匹配问题的最可靠方法。
        model_dtype = next(self.model.parameters()).dtype
        
        for key, value in kwargs.items():
            # 仅转换浮点类型的张量，忽略整数类型（如 input_ids）
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                kwargs[key] = value.to(model_dtype)
        # --- 结束终极修复 ---

        if HIGGS_AVAILABLE:
            return self.model(**kwargs)
        else:
            # Fallback 逻辑也受益于上面的类型转换
            outputs = self.model(input_ids=kwargs.get('input_ids'), attention_mask=kwargs.get('attention_mask'))
            loss = None
            if kwargs.get('label_ids') is not None:
                logits = outputs.logits[..., :-1, :].contiguous()
                labels = kwargs.get('label_ids')[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": outputs.logits}

class HiggsAudioTrainer(Trainer):
    """Custom trainer for Higgs Audio v2"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = self.model.config
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation"""
        # 类型转换逻辑已移至 Model Wrapper，此处不再需要
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss
        
def setup_lora_config(model: nn.Module, lora_config: Dict) -> nn.Module:
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("rank", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        auto_mapping=True
    )
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        model.model.text_model = get_peft_model(model.model.text_model, peft_config)
    elif hasattr(model, 'model'):
        model.model = get_peft_model(model.model, peft_config)
    else:
        model = get_peft_model(model, peft_config)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Higgs Audio v2 with LoRA")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--audio_tokenizer_path", type=str, required=True)
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--eval_data_dir", type=str, default="")

    # Task type arguments
    parser.add_argument("--task_type", type=str, default="single_speaker_smart_voice", choices=["zero_shot_voice_cloning", "single_speaker_smart_voice", "multi_speaker_smart_voice", "multi_speaker_voice_cloning"])
    parser.add_argument("--ref_audio_in_system_message", action="store_true", default=False)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=False, 
                       help="Enable bfloat16 mixed precision training.")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="./logs")
    
    # Freeze model components
    parser.add_argument("--freeze_audio_tower", action="store_true", default=False)
    parser.add_argument("--freeze_audio_encoder_proj", action="store_true", default=False)
    parser.add_argument("--freeze_llm", action="store_true", default=False)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    audio_tokenizer = None
    if HIGGS_AVAILABLE:
        try:
            audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path, device=device)
        except Exception as e:
            logger.warning(f"Failed to load audio tokenizer: {e}")

    model = HiggsAudioModelWrapper(args.model_path, device=device, args=args)
    
    if args.bf16:
        model.to(torch.bfloat16)
        logger.info("Model manually cast to bfloat16.")

    if args.use_lora:
        lora_config = {"rank": args.lora_rank, "alpha": args.lora_alpha, "dropout": args.lora_dropout}
        model = setup_lora_config(model, lora_config)
        logger.info("LoRA configuration applied")

    train_dataset = HiggsAudioDataset(args.train_data_dir, tokenizer, audio_tokenizer, task_type=args.task_type, ref_audio_in_system_message=args.ref_audio_in_system_message)
    eval_dataset = HiggsAudioDataset(args.eval_data_dir, tokenizer, audio_tokenizer, task_type=args.task_type, ref_audio_in_system_message=args.ref_audio_in_system_message) if args.eval_data_dir else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        fp16=False,
        bf16=args.bf16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=args.report_to,
        logging_dir=args.logging_dir,
        # --- 开始终极修复 ---
        # 设置为 True 来解决 DDP 挂起问题
        ddp_find_unused_parameters=True,
        # --- 结束终极修复 ---
    )

    data_collator = None
    if HIGGS_AVAILABLE and hasattr(model.config, 'audio_in_token_idx'):
        try:
            from transformers import WhisperProcessor
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            data_collator = ExtendedHiggsAudioSampleCollator(
                whisper_processor=whisper_processor,
                audio_in_token_id=model.config.audio_in_token_idx,
                audio_out_token_id=model.config.audio_out_token_idx,
                audio_stream_bos_id=model.config.audio_stream_bos_id,
                audio_stream_eos_id=model.config.audio_stream_eos_id,
                encode_whisper_embed=True,
                pad_token_id=tokenizer.pad_token_id,
                return_audio_in_tokens=True,
                use_delay_pattern=False,
                round_to=8,
                audio_num_codebooks=getattr(model.config, 'audio_num_codebooks', 8),
            )
        except Exception as e:
            logger.warning(f"Failed to setup Higgs collator: {e}. Using fallback.")
    if data_collator is None:
        data_collator = ExtendedHiggsAudioSampleCollator(pad_token_id=tokenizer.pad_token_id)
        logger.warning("Using fallback collator")
        
    trainer = HiggsAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info(f"Starting training for task: {args.task_type} on device: {device}")
    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model()
        logger.info(f"Model saved to {args.output_dir}")
        if args.use_lora:
            lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
            model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            model_to_save.save_pretrained(lora_output_dir)
            logger.info(f"LoRA adapters saved to {lora_output_dir}")

if __name__ == "__main__":
    main()