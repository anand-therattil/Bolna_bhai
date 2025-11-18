"""Example script for batch generating audio using HiggsAudio."""

import click
import soundfile as sf
import langid
import jieba
import os
import re
import copy
import torchaudio
import tqdm
import yaml
import glob
from pathlib import Path

from loguru import logger
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent

from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import (
    ChatMLDatasetSample,
    prepare_chatml_sample,
)
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from typing import List
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache
from typing import Optional
from dataclasses import asdict
import torch

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    chinese_to_english_punct = {
        "，": ", ",  # comma
        "。": ".",  # period
        "：": ":",  # colon
        "；": ";",  # semicolon
        "？": "?",  # question mark
        "！": "!",  # exclamation mark
        "（": "(",  # left parenthesis
        "）": ")",  # right parenthesis
        "【": "[",  # left square bracket
        "】": "]",  # right square bracket
        "《": "<",  # left angle quote
        "》": ">",  # right angle quote
        """: '"',  # left double quotation
        """: '"',  # right double quotation
        "'": "'",  # left single quotation
        "'": "'",  # right single quotation
        "、": ",",  # enumeration comma
        "--": "-",  # em dash
        "…": "...",  # ellipsis
        "·": ".",  # middle dot
        "「": '"',  # left corner bracket
        "」": '"',  # right corner bracket
        "『": '"',  # left double corner bracket
        "』": '"',  # right double corner bracket
    }

    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)
    return text


def prepare_chunk_text(
    text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1
):
    """Chunk the text into smaller pieces. We will later feed the chunks one by one to the model."""
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            chunks[-1] += "\n\n"
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _build_system_message_with_audio_prompt(system_message):
    contents = []
    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    ret = Message(
        role="system",
        content=contents,
    )
    return ret


class HiggsAudioBatchModelClient:
    def __init__(
        self,
        model_path,
        audio_tokenizer,
        device_id=None,
        max_new_tokens=2048,
        kv_cache_lengths: List[int] = [1024, 4096, 8192],
        use_static_kv_cache=False,
    ):
        if device_id is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = f"cuda:{device_id}"
        self._audio_tokenizer = (
            load_higgs_audio_tokenizer(audio_tokenizer, device=self._device)
            if isinstance(audio_tokenizer, str)
            else audio_tokenizer
        )
        self._model = HiggsAudioModel.from_pretrained(
            model_path,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()
        self._kv_cache_lengths = kv_cache_lengths
        self._use_static_kv_cache = use_static_kv_cache

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self._max_new_tokens = max_new_tokens
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self._config.audio_in_token_idx,
            audio_out_token_id=self._config.audio_out_token_idx,
            audio_stream_bos_id=self._config.audio_stream_bos_id,
            audio_stream_eos_id=self._config.audio_stream_eos_id,
            encode_whisper_embed=self._config.encode_whisper_embed,
            pad_token_id=self._config.pad_token_id,
            return_audio_in_tokens=self._config.encode_audio_in_tokens,
            use_delay_pattern=self._config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self._config.audio_num_codebooks,
        )
        self.kv_caches = None
        if use_static_kv_cache:
            self._init_static_kv_cache()

    def _init_static_kv_cache(self):
        cache_config = copy.deepcopy(self._model.config.text_config)
        cache_config.num_hidden_layers = self._model.config.text_config.num_hidden_layers
        if self._model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self._model.config.audio_dual_ffn_layers)
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self._model.device,
                dtype=self._model.dtype,
            )
            for length in sorted(self._kv_cache_lengths)
        }
        if "cuda" in self._device:
            logger.info(f"Capturing CUDA graphs for each KV cache length")
            self._model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    @torch.inference_mode()
    def generate_single(
        self,
        messages,
        audio_ids,
        chunked_text,
        generation_chunk_buffer_size,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        seed=123,
    ):
        """Generate audio for a single text input"""
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None
        sr = 24000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []
        
        for idx, chunk_text in enumerate(chunked_text):
            generation_messages.append(
                Message(
                    role="user",
                    content=chunk_text,
                )
            )
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            try:
                input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self._tokenizer)
            except:
                input_tokens, label_tokens, audio_contents, audio_label_contents, speaker_id = prepare_chatml_sample(chatml_sample, self._tokenizer)
            postfix = self._tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
            input_tokens.extend(postfix)

            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
                if context_audio_ids
                else None,
                audio_ids_start=torch.cumsum(
                    torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0
                )
                if context_audio_ids
                else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self._collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)

            if self._use_static_kv_cache:
                self._prepare_kv_caches()

            outputs = self._model.generate(
                **batch,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=seed,
            )

            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1])
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)

            generation_messages.append(
                Message(
                    role="assistant",
                    content=AudioContent(audio_url=""),
                )
            )
            if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
                generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
                generation_messages = generation_messages[(-2 * generation_chunk_buffer_size) :]

        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)
        concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids.unsqueeze(0))[0, 0]
        text_result = self._tokenizer.decode(outputs[0][0])
        return concat_wv, sr, text_result

    def generate_batch(self, batch_data_list, **generation_kwargs):
        """Generate audio for a batch of inputs"""
        results = []
        for i, data in enumerate(tqdm.tqdm(batch_data_list, desc="Processing batch")):
            logger.info(f"Processing item {i+1}/{len(batch_data_list)}")
            try:
                waveform, sr, text_output = self.generate_single(**data, **generation_kwargs)
                results.append({
                    'waveform': waveform,
                    'sample_rate': sr,
                    'text_output': text_output,
                    'success': True,
                    'error': None
                })
            except Exception as e:
                logger.error(f"Error processing item {i+1}: {str(e)}")
                results.append({
                    'waveform': None,
                    'sample_rate': None,
                    'text_output': None,
                    'success': False,
                    'error': str(e)
                })
        return results


def prepare_generation_context(scene_prompt, ref_audio, ref_audio_in_system_message, audio_tokenizer, speaker_tags):
    """Prepare the context for generation."""
    system_message = None
    messages = []
    audio_ids = []
    if ref_audio is not None:
        num_speakers = len(ref_audio.split(","))
        speaker_info_l = ref_audio.split(",")
        voice_profile = None
        if any([speaker_info.startswith("profile:") for speaker_info in ref_audio.split(",")]):
            ref_audio_in_system_message = True
        if ref_audio_in_system_message:
            speaker_desc = []
            for spk_id, character_name in enumerate(speaker_info_l):
                if character_name.startswith("profile:"):
                    if voice_profile is None:
                        with open(f"{CURR_DIR}/voice_prompts/profile.yaml", "r", encoding="utf-8") as f:
                            voice_profile = yaml.safe_load(f)
                    character_desc = voice_profile["profiles"][character_name[len("profile:") :].strip()]
                    speaker_desc.append(f"SPEAKER{spk_id}: {character_desc}")
                else:
                    speaker_desc.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")
            if scene_prompt:
                system_message = (
                    "Generate audio following instruction."
                    "\n\n"
                    f"<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
                )
            else:
                system_message = (
                    "Generate audio following instruction.\n\n"
                    + f"<|scene_desc_start|>\n"
                    + "\n".join(speaker_desc)
                    + "\n<|scene_desc_end|>"
                )
            system_message = _build_system_message_with_audio_prompt(system_message)
        else:
            if scene_prompt:
                system_message = Message(
                    role="system",
                    content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>",
                )
        voice_profile = None
        for spk_id, character_name in enumerate(ref_audio.split(",")):
            if not character_name.startswith("profile:"):
                prompt_audio_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name}.wav")
                prompt_text_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name}.txt")
                assert os.path.exists(prompt_audio_path), (
                    f"Voice prompt audio file {prompt_audio_path} does not exist."
                )
                assert os.path.exists(prompt_text_path), f"Voice prompt text file {prompt_text_path} does not exist."
                with open(prompt_text_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                audio_tokens = audio_tokenizer.encode(prompt_audio_path)
                audio_ids.append(audio_tokens)

                if not ref_audio_in_system_message:
                    messages.append(
                        Message(
                            role="user",
                            content=f"[SPEAKER{spk_id}] {prompt_text}" if num_speakers > 1 else prompt_text,
                        )
                    )
                    messages.append(
                        Message(
                            role="assistant",
                            content=AudioContent(
                                audio_url=prompt_audio_path,
                            ),
                        )
                    )
    else:
        if len(speaker_tags) > 1:
            speaker_desc_l = []
            for idx, tag in enumerate(speaker_tags):
                if idx % 2 == 0:
                    speaker_desc = f"feminine"
                else:
                    speaker_desc = f"masculine"
                speaker_desc_l.append(f"{tag}: {speaker_desc}")

            speaker_desc = "\n".join(speaker_desc_l)
            scene_desc_l = []
            if scene_prompt:
                scene_desc_l.append(scene_prompt)
            scene_desc_l.append(speaker_desc)
            scene_desc = "\n\n".join(scene_desc_l)

            system_message = Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
            )
        else:
            system_message_l = ["Generate audio following instruction."]
            if scene_prompt:
                system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
            system_message = Message(
                role="system",
                content="\n\n".join(system_message_l),
            )
    if system_message:
        messages.insert(0, system_message)
    return messages, audio_ids


def preprocess_text(transcript):
    """Preprocess and normalize input text"""
    transcript = normalize_chinese_punctuation(transcript)
    transcript = transcript.replace("(", " ")
    transcript = transcript.replace(")", " ")
    transcript = transcript.replace("°F", " degrees Fahrenheit")
    transcript = transcript.replace("°C", " degrees Celsius")

    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE>[Humming]</SE>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        transcript = transcript.replace(tag, replacement)
    
    lines = transcript.split("\n")
    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    transcript = transcript.strip()

    if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        transcript += "."
    
    return transcript


def load_text_files(input_path):
    """Load text files from input path (file or directory)"""
    text_files = []
    
    if os.path.isfile(input_path):
        if input_path.endswith('.txt'):
            text_files = [input_path]
        else:
            raise ValueError(f"Input file must be a .txt file: {input_path}")
    elif os.path.isdir(input_path):
        text_files = glob.glob(os.path.join(input_path, "*.txt"))
        text_files.sort()
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    if not text_files:
        raise ValueError(f"No .txt files found in: {input_path}")
    
    logger.info(f"Found {len(text_files)} text files to process")
    return text_files


@click.command()
@click.option(
    "--model_path",
    type=str,
    default="/root/code/higgs-audio-main/model_ckpt",
    help="Model checkpoint path.",
)
@click.option(
    "--audio_tokenizer",
    type=str,
    default="/root/code/higgs-audio-main/model_ckpt_tokenizer",
    help="Audio tokenizer path.",
)
@click.option(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="The maximum number of new tokens to generate.",
)
@click.option(
    "--input_path",
    type=str,
    required=True,
    help="Path to input text file or directory containing .txt files.",
)
@click.option(
    "--output_dir",
    type=str,
    default="batch_output",
    help="Output directory for generated audio files.",
)
@click.option(
    "--scene_prompt",
    type=str,
    default=f"{CURR_DIR}/scene_prompts/quiet_indoor.txt",
    help="Scene description prompt file path.",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature for generation.",
)
@click.option(
    "--top_k",
    type=int,
    default=50,
    help="Top-k sampling parameter.",
)
@click.option(
    "--top_p",
    type=float,
    default=0.95,
    help="Top-p sampling parameter.",
)
@click.option(
    "--ras_win_len",
    type=int,
    default=7,
    help="RAS window length.",
)
@click.option(
    "--ras_win_max_num_repeat",
    type=int,
    default=2,
    help="RAS max repeat count.",
)
@click.option(
    "--ref_audio",
    type=str,
    default=None,
    help="Reference audio for voice cloning.",
)
@click.option(
    "--ref_audio_in_system_message",
    is_flag=True,
    default=False,
    help="Include reference audio in system message.",
)
@click.option(
    "--chunk_method",
    default=None,
    type=click.Choice([None, "speaker", "word"]),
    help="Text chunking method.",
)
@click.option(
    "--chunk_max_word_num",
    default=200,
    type=int,
    help="Maximum words per chunk for word chunking.",
)
@click.option(
    "--chunk_max_num_turns",
    default=1,
    type=int,
    help="Maximum turns per chunk for speaker chunking.",
)
@click.option(
    "--generation_chunk_buffer_size",
    default=None,
    type=int,
    help="Maximum chunks to keep in buffer.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for generation.",
)
@click.option(
    "--device_id",
    type=int,
    default=None,
    help="GPU device ID.",
)
@click.option(
    "--use_static_kv_cache",
    type=int,
    default=1,
    help="Use static KV cache for faster generation.",
)
@click.option(
    "--batch_size",
    type=int,
    default=4,
    help="Batch size for processing (currently processes sequentially but loads in batches).",
)
def main(
    model_path,
    audio_tokenizer,
    max_new_tokens,
    input_path,
    output_dir,
    scene_prompt,
    temperature,
    top_k,
    top_p,
    ras_win_len,
    ras_win_max_num_repeat,
    ref_audio,
    ref_audio_in_system_message,
    chunk_method,
    chunk_max_word_num,
    chunk_max_num_turns,
    generation_chunk_buffer_size,
    seed,
    device_id,
    use_static_kv_cache,
    batch_size,
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    if device_id is None:
        if torch.cuda.is_available():
            device_id = 0
            device = "cuda:0"
        else:
            device_id = None
            device = "cpu"
    else:
        device = f"cuda:{device_id}"
    
    # Load tokenizer and model
    logger.info("Loading audio tokenizer and model...")
    audio_tokenizer_obj = load_higgs_audio_tokenizer(audio_tokenizer, device=device)
    
    model_client = HiggsAudioBatchModelClient(
        model_path=model_path,
        audio_tokenizer=audio_tokenizer_obj,
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        use_static_kv_cache=use_static_kv_cache,
    )
    
    # Load scene prompt
    if scene_prompt is not None and scene_prompt != "empty" and os.path.exists(scene_prompt):
        with open(scene_prompt, "r", encoding="utf-8") as f:
            scene_prompt_text = f.read().strip()
    else:
        scene_prompt_text = None
    
    # Load text files
    text_files = load_text_files(input_path)
    
    # Process files in batches
    for batch_start in range(0, len(text_files), batch_size):
        batch_end = min(batch_start + batch_size, len(text_files))
        batch_files = text_files[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(text_files)-1)//batch_size + 1}")
        
        # Prepare batch data
        batch_data_list = []
        batch_output_paths = []
        
        for file_path in batch_files:
            # Load and preprocess text
            with open(file_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
            
            transcript = preprocess_text(transcript)
            
            # Extract speaker tags
            pattern = re.compile(r"\[(SPEAKER\d+)\]")
            speaker_tags = sorted(set(pattern.findall(transcript)))
            
            # Prepare generation context
            messages, audio_ids = prepare_generation_context(
                scene_prompt=scene_prompt_text,
                ref_audio=ref_audio,
                ref_audio_in_system_message=ref_audio_in_system_message,
                audio_tokenizer=audio_tokenizer_obj,
                speaker_tags=speaker_tags,
            )
            
            # Prepare chunks
            chunked_text = prepare_chunk_text(
                transcript,
                chunk_method=chunk_method,
                chunk_max_word_num=chunk_max_word_num,
                chunk_max_num_turns=chunk_max_num_turns,
            )
            
            # Prepare data for this file
            file_data = {
                'messages': messages,
                'audio_ids': audio_ids,
                'chunked_text': chunked_text,
                'generation_chunk_buffer_size': generation_chunk_buffer_size,
            }
            
            batch_data_list.append(file_data)
            
            # Prepare output path
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.wav")
            batch_output_paths.append(output_path)
        
        # Generate audio for batch
        generation_kwargs = {
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'ras_win_len': ras_win_len,
            'ras_win_max_num_repeat': ras_win_max_num_repeat,
            'seed': seed,
        }
        
        results = model_client.generate_batch(batch_data_list, **generation_kwargs)
        
        # Save results
        for i, (result, output_path) in enumerate(zip(results, batch_output_paths)):
            if result['success']:
                sf.write(output_path, result['waveform'], result['sample_rate'])
                logger.info(f"Saved audio: {output_path}")
            else:
                logger.error(f"Failed to generate audio for {batch_files[i]}: {result['error']}")
    
    logger.info(f"Batch processing completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()