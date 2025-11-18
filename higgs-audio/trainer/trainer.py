#!/usr/bin/env python3
"""
Higgs Audio v2 Training Script with LoRA Support and Task-specific Training
Based on the Higgs Audio v2 architecture from Boson AI
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
    
    # 添加fallback类定义
    class ChatMLDatasetSample:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def to(self, device):
            """将所有tensor属性转移到指定设备"""
            new_sample = ChatMLDatasetSample()
            for key, value in self.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(new_sample, key, value.to(device))
                else:
                    setattr(new_sample, key, value)
            return new_sample

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
            self.base_collator = HiggsAudioSampleCollator(**kwargs)
        else:
            # Fallback collator
            self.pad_token_id = kwargs.get('pad_token_id', 0)
    
    def __call__(self, batch: List[ChatMLDatasetSample]):
        if HIGGS_AVAILABLE and hasattr(self, 'base_collator'):
            # 1. 调用官方的、底层的 collator，让它完成所有复杂的填充和对齐工作
            batch_input = self.base_collator(batch)
            
            # batch_input.audio_out_ids 是经过填充和处理的，其长度与模型输出的 audio_logits 长度完全一致。
            label_audio_ids = batch_input.audio_out_ids
            
            # 2. 转换为我们的扩展类，并传入这个完美的标签
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
                label_audio_ids=label_audio_ids, # <-- 使用我们新定义的、对齐的标签
                reward=batch_input.reward,
            )
            
            return extended_batch
        else:
            # Fallback implementation
            input_ids_list = []
            attention_mask_list = []
            label_ids_list = []
            
            for sample in batch:
                input_ids_list.append(sample.input_ids)
                attention_mask = torch.ones_like(sample.input_ids)
                attention_mask_list.append(attention_mask)
                
                if hasattr(sample, 'label_ids'):
                    label_ids_list.append(sample.label_ids)
                else:
                    label_ids_list.append(sample.input_ids)
            
            # Pad sequences
            max_len = max(len(ids) for ids in input_ids_list)
            
            padded_input_ids = []
            padded_attention_mask = []
            padded_label_ids = []
            
            for i in range(len(input_ids_list)):
                pad_length = max_len - len(input_ids_list[i])
                
                padded_input = torch.cat([
                    input_ids_list[i],
                    torch.full((pad_length,), self.pad_token_id, dtype=torch.long)
                ])
                padded_input_ids.append(padded_input)
                
                padded_mask = torch.cat([
                    attention_mask_list[i],
                    torch.zeros(pad_length, dtype=torch.long)
                ])
                padded_attention_mask.append(padded_mask)
                
                padded_label = torch.cat([
                    label_ids_list[i],
                    torch.full((pad_length,), -100, dtype=torch.long)
                ])
                padded_label_ids.append(padded_label)
            
            return ExtendedHiggsAudioBatchInput(
                input_ids=torch.stack(padded_input_ids),
                attention_mask=torch.stack(padded_attention_mask),
                label_ids=torch.stack(padded_label_ids),
                audio_features=None,
                audio_feature_attention_mask=None,
                audio_out_ids=None,
                audio_out_ids_start=None,
                audio_out_ids_start_group_loc=None,
                audio_in_ids=None,
                audio_in_ids_start=None,
                label_audio_ids=None,
                reward=None,
            )


def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    chinese_to_english_punct = {
        "，": ", ", "。": ".", "：": ":", "；": ";", "？": "?", "！": "!",
        "（": "(", "）": ")", "【": "[", "】": "]", "《": "<", "》": ">",
        """: '"', """: '"', "'": "'", "'": "'", "、": ",", "--": "-",
        "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }

    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)

    return text


def _build_system_message_with_audio_prompt(system_message):
    """Build system message with audio prompts"""
    contents = []

    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN):]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    
    ret = Message(
        role="system",
        content=contents,
    )
    return ret


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
        device: Optional[str] = None,
    ):
        """
        用于 Higgs Audio 模型训练的数据集。
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.task_type = task_type
        self.sample_rate = sample_rate
        self.use_metadata = use_metadata
        self.ref_audio_in_system_message = ref_audio_in_system_message
        self.device = device
        
        valid_tasks = [
            "zero_shot_voice_cloning", 
            "single_speaker_smart_voice", 
            "multi_speaker_smart_voice", 
            "multi_speaker_voice_cloning"
        ]
        if self.task_type not in valid_tasks:
            raise ValueError(f"Invalid task_type: {self.task_type}. Must be one of {valid_tasks}")
        
        self.actual_num_codebooks = self._detect_codebook_size()
        
        if use_metadata and (self.data_dir / "metadata.json").exists():
            self.samples = self._load_samples_from_metadata()
        else:
            logger.warning(f"metadata.json not found in {data_dir}. Scanning directory instead.")
            self.samples = self._load_samples_from_directory()
            
        if not self.samples:
            raise RuntimeError(f"No valid samples found in {data_dir} for task '{self.task_type}'. Please check your data and metadata.json.")
            
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir} for task: {self.task_type}")

    def _detect_codebook_size(self) -> int:
        """通过编码一个测试音频来动态检测音频 tokenizer 的 codebook 数量。"""
        try:
            audio_files = list(self.data_dir.glob("*.wav")) + list(self.data_dir.glob("*.mp3"))
            if audio_files:
                test_audio_path = str(audio_files[0])
                test_tokens = self._encode_audio_tokens(test_audio_path)
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
        
        # 预处理路径，使其成为绝对路径
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
                samples.append({
                    "audio_file": str(audio_path),
                    "transcript_file": str(transcript_path),
                    "audio_id": audio_path.stem,
                })
        return samples

    def _load_audio_waveform(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
            return waveform.squeeze(0), self.sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return torch.zeros(1), self.sample_rate

    def _encode_audio_tokens(self, audio_path: str) -> Optional[torch.Tensor]:
        if not self.audio_tokenizer: 
            return None
        try:
            return self.audio_tokenizer.encode(audio_path)
        except Exception as e:
            logger.error(f"Failed to encode audio {audio_path}: {e}")
            return None

    def _normalize_transcript(self, transcript: str) -> str:
        """Normalize transcript text"""
        transcript = normalize_chinese_punctuation(transcript)
        transcript = transcript.replace("(", " ").replace(")", " ")
        transcript = transcript.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")
        
        for tag, replacement in [("[laugh]", "<SE>[Laughter]</SE>"), ("[cough]", "<SE>[Cough]</SE>"),]:
            transcript = transcript.replace(tag, replacement)
            
        lines = transcript.split("\n")
        transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
        
        if transcript and not any(transcript.endswith(c) for c in ".!?,\";'</SE_e></SE>"):
            transcript += "."
            
        return transcript.strip()

    def _get_scene_description(self, sample: Dict) -> str:
        scene = sample.get("scene", "a quiet room")
        return f"Audio is recorded from {scene.replace('_', ' ')}."

    def _detect_speaker_tags(self, transcript: str) -> List[str]:
        return sorted(set(re.findall(r"\[(SPEAKER\d+)\]", transcript)))
    
    def _create_messages_for_task(self, sample: Dict, transcript: str) -> List[Message]:
        """
        根据任务类型为给定的样本创建消息列表（提示）。
        """
        messages = []
        speaker_tags = self._detect_speaker_tags(transcript)
        scene_prompt = self._get_scene_description(sample) if self.use_metadata else None

        if self.task_type == "zero_shot_voice_cloning":
            ref_audio = sample.get("ref_audio_file")
            if not ref_audio:
                logger.warning(f"Sample {sample['id']} is for zero_shot_voice_cloning but has no 'ref_audio_file'.")
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
                logger.warning(f"Sample {sample['audio_id']} is for multi_speaker_voice_cloning but has no valid 'ref_speakers' list in metadata.")
                return [Message(role="system", content=MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE)]
            
            for i, ref_info in enumerate(ref_speakers):
                speaker_tag = ref_info.get("speaker_tag", f"[SPEAKER{i}]")
                ref_audio_path = str(self.data_dir / ref_info["ref_audio_file"])
                ref_transcript = ref_info.get("ref_transcript", "This is a voice sample.")
                messages.append(Message(role="user", content=f"{speaker_tag} {ref_transcript}"))
                messages.append(Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)))
            
            messages.insert(0, Message(role="system", content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"))

        elif self.task_type == "multi_speaker_smart_voice" and len(speaker_tags) > 1:
            speaker_desc_l = [f"{tag}: {'feminine' if i % 2 == 0 else 'masculine'}" for i, tag in enumerate(speaker_tags)]
            scene_desc = f"{scene_prompt}\n\n" + "\n".join(speaker_desc_l) if scene_prompt else "\n".join(speaker_desc_l)
            messages.append(Message(role="system", content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"))
        
        else: # single_speaker_smart_voice 或其他回退情况
            content = "Generate audio following instruction."
            if scene_prompt:
                content += f"\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>"
            messages.append(Message(role="system", content=content))
            
        return messages

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ChatMLDatasetSample:
        sample = self.samples[idx]
        
        try:
            with open(sample["transcript_file"], 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            transcript = self._normalize_transcript(transcript)
            
            # 1. 构建消息历史（提示）
            messages = self._create_messages_for_task(sample, transcript)
            
            # 2. 添加当前样本的目标对话
            messages.append(Message(role="user", content=transcript))
            messages.append(Message(role="assistant", content=AudioContent(audio_url=sample["audio_file"])))

            chatml_sample = ChatMLSample(messages=messages)
            
            # 3. 使用处理函数处理 ChatML 样本
            input_tokens, label_tokens, audio_contents, audio_label_contents, _ = prepare_chatml_sample(
                chatml_sample, self.tokenizer
            )

            # 4. 处理音频数据
            context_audio_tokens = []
            for audio_content in (audio_contents or []):
                if audio_content.audio_url:
                    tokens = self._encode_audio_tokens(audio_content.audio_url)
                    if tokens is not None: 
                        context_audio_tokens.append(tokens)

            label_audio_tokens = []
            for audio_label_content in (audio_label_contents or []):
                if audio_label_content.audio_url:
                    tokens = self._encode_audio_tokens(audio_label_content.audio_url)
                    if tokens is not None: 
                        label_audio_tokens.append(tokens)

            # 5. 拼接张量
            if context_audio_tokens:
                audio_ids_concat = torch.cat(context_audio_tokens, dim=1)
                audio_ids_start = torch.tensor([0] + [t.shape[1] for t in context_audio_tokens[:-1]], dtype=torch.long).cumsum(0)
            else:
                audio_ids_concat = torch.zeros((self.actual_num_codebooks, 0), dtype=torch.long)
                audio_ids_start = torch.tensor([0], dtype=torch.long)

            label_audio_ids = torch.cat(label_audio_tokens, dim=1) if label_audio_tokens else None

            # 为 ChatMLDatasetSample 准备其他字段
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
            
            # 将所有张量移动到指定设备
            return dataset_sample.to(self.device) if self.device else dataset_sample

        except Exception as e:
            logger.error(f"Error processing sample at index {idx} (ID: {sample.get('audio_id', 'N/A')}): {e}", exc_info=True)
            # 返回下一个样本，避免因单个损坏样本导致训练中断
            return self.__getitem__((idx + 1) % len(self))


class HiggsAudioModelWrapper(nn.Module):
    """Wrapper for Higgs Audio v2 model to enable training"""
    
    def __init__(self, model_path: str, device: str = 'cuda', args=None):
        super().__init__()
        if HIGGS_AVAILABLE:
            self.model = HiggsAudioModel.from_pretrained(
                config=HiggsAudioConfig.from_pretrained(model_path),
                pretrained_model_name_or_path=model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
            self.config = self.model.config
        else:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            self.config = self.model.config
        self.model = self.model.to(device)
        
        if args:
            if args.freeze_audio_tower:
                self.model.freeze_audio_tower()
            if args.freeze_audio_encoder_proj:
                self.model.freeze_audio_encoder_proj()
            if args.freeze_llm:
                self.model.freeze_llm()


    @property
    def device(self):
        return self.model.device
          
    def forward(self, **kwargs):
        if self.model.device != kwargs['input_ids'].device:
            self.model = self.model.to(kwargs['input_ids'].device)

        if HIGGS_AVAILABLE:
            return self.model(**kwargs)
        else:
            input_ids = kwargs.get('input_ids')
            attention_mask = kwargs.get('attention_mask')
            labels = kwargs.get('label_ids')
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = None
            if labels is not None:
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
            return {"loss": loss, "logits": outputs.logits}


class HiggsAudioTrainer(Trainer):
    """Custom trainer for Higgs Audio v2"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 添加这行，从model中获取config
        self.config = self.model.config
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss computation"""
        if isinstance(inputs, ExtendedHiggsAudioBatchInput):
            model_inputs = {}
            for attr_name in ['input_ids', 'attention_mask', 'label_ids', 
                            'audio_features', 'audio_feature_attention_mask',
                            'audio_out_ids', 'audio_out_ids_start', 
                            'audio_in_ids', 'audio_in_ids_start',
                            'label_audio_ids']:
                attr_value = getattr(inputs, attr_name, None)
                if attr_value is not None:
                    model_inputs[attr_name] = attr_value
        else:
            model_inputs = {}
            for key, value in inputs.items():
                if key == 'labels':
                    model_inputs['label_ids'] = value
                elif key in ['input_ids', 'attention_mask', 'label_ids',
                            'audio_features', 'audio_feature_attention_mask',
                            'audio_out_ids', 'audio_out_ids_start', 
                            'audio_in_ids', 'audio_in_ids_start',
                            'label_audio_ids']:
                    model_inputs[key] = value
        
        # 确保所有输入都在相同设备上
        for key, value in model_inputs.items():
            if isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(model.device)
        
        outputs = model(**model_inputs)
        
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss


def setup_lora_config(model: nn.Module, lora_config: Dict) -> nn.Module:
    """Setup LoRA configuration for the model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get("rank", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.1),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        auto_mapping=True
    )
    
    model = model.to(device)
    
    if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
        model.model.text_model = get_peft_model(model.model.text_model, peft_config)
    elif hasattr(model, 'model'):
        model.model = get_peft_model(model.model, peft_config)
    else:
        model = get_peft_model(model, peft_config)
    
    model = model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Higgs Audio v2 with LoRA")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="/root/code/higgs-audio-main/model_ckpt")
    parser.add_argument("--audio_tokenizer_path", type=str, default="/root/code/higgs-audio-main/model_ckpt_tokenizer")
    
    # Data arguments
    parser.add_argument("--train_data_dir", type=str, default="/root/code/higgs-audio-main/higgs_training_data_huo")
    parser.add_argument("--eval_data_dir", type=str, default="")

    # Task type arguments
    parser.add_argument("--task_type", type=str, default="single_speaker_smart_voice",
                       choices=["zero_shot_voice_cloning", "single_speaker_smart_voice", 
                               "multi_speaker_smart_voice", "multi_speaker_voice_cloning"],
                       help="Training task type")
    parser.add_argument("--ref_audio_in_system_message", action="store_true", default=False,
                       help="Whether to include reference audio in system message")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output/huo_train-vxx")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=500)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--logging_dir", type=str, default="./logs/huo_train-vxx")
    parser.add_argument("--report_to", type=str, default="tensorboard", 
                       choices=["tensorboard", "wandb", "none"])
    
    # Freeze model components
    parser.add_argument("--freeze_audio_tower", action="store_true", default=False)
    parser.add_argument("--freeze_audio_encoder_proj", action="store_true", default=False)
    parser.add_argument("--freeze_llm", action="store_true", default=False)


    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load audio tokenizer
    if HIGGS_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            audio_tokenizer = load_higgs_audio_tokenizer(args.audio_tokenizer_path, device=device)
        except Exception as e:
            logger.warning(f"Failed to load audio tokenizer: {e}")
            audio_tokenizer = None
    else:
        audio_tokenizer = None
        logger.warning("Audio tokenizer not available, using fallback")
    
    # Load model
    model = HiggsAudioModelWrapper(args.model_path, device='cuda', args=args)
    
    # Setup LoRA
    if args.use_lora:
        lora_config = {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
        model = setup_lora_config(model, lora_config)
        logger.info("LoRA configuration applied")
    
    # Load datasets
    train_dataset = HiggsAudioDataset(
        args.train_data_dir,
        tokenizer,
        audio_tokenizer,
        task_type=args.task_type,
        ref_audio_in_system_message=args.ref_audio_in_system_message
    )
    
    eval_dataset = None
    if args.eval_data_dir:
        eval_dataset = HiggsAudioDataset(
            args.eval_data_dir,
            tokenizer,
            audio_tokenizer,
            task_type=args.task_type,
            ref_audio_in_system_message=args.ref_audio_in_system_message
        )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        fp16=args.fp16,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=args.report_to if args.report_to != "none" else None,
        logging_dir=args.logging_dir,
    )
    
    # Setup data collator
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
                audio_num_codebooks=8,
            )
        except Exception as e:
            logger.warning(f"Failed to setup Higgs collator: {e}. Using fallback.")
            data_collator = ExtendedHiggsAudioSampleCollator(pad_token_id=tokenizer.pad_token_id)
    else:
        data_collator = ExtendedHiggsAudioSampleCollator(pad_token_id=tokenizer.pad_token_id)
        logger.warning("Using fallback collator")
    config = AutoConfig.from_pretrained(args.model_path)
    # Initialize trainer
    trainer = HiggsAudioTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info(f"Starting training for task: {args.task_type}")
    trainer.train()
    
    # Save the final model
    config.save_pretrained(args.output_dir)
    trainer.save_model()
    logger.info(f"Model saved to {args.output_dir}")
    
    # Save LoRA adapters separately
    if args.use_lora:
        lora_output_dir = os.path.join(args.output_dir, "lora_adapters")
        if hasattr(model, 'model') and hasattr(model.model, 'text_model'):
            model.model.text_model.save_pretrained(lora_output_dir)
        elif hasattr(model, 'model'):
            model.model.save_pretrained(lora_output_dir)
        else:
            model.save_pretrained(lora_output_dir)
        logger.info(f"LoRA adapters saved to {lora_output_dir}")


if __name__ == "__main__":
    main()
