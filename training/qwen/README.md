# Qwen Fine-tuning Pipeline

A complete pipeline for fine-tuning Qwen 0.6B model using LoRA (Low-Rank Adaptation) for Polymarket-specific tasks.

## Files Overview

- [`finetune.py`](finetune.py) - Main training script using Unsloth and LoRA
- [`merge_lora.py`](merge_lora.py) - Merges LoRA weights with base model
- [`inference.py`](inference.py) - Standard inference using merged model
- [`vllm_inference.py`](vllm_inference.py) - High-performance inference using vLLM
- [`stream_inference.py`](stream_inference.py) - Real-time streaming inference using async vLLM
- [`sample_dataset.jsonl`](sample_dataset.jsonl) - Sample training data for Polymarket betting

## Required Packages

```bash
pip install torch transformers datasets
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
pip install trl peft accelerate bitsandbytes
pip install rich  # For pretty console output
pip install vllm  # For high-performance inference
```

## Training Pipeline

### 1. Fine-tuning ([`finetune.py`](finetune.py))

Fine-tunes Qwen3-0.6B using LoRA with 4-bit quantization:

- **Model**: `unsloth/Qwen3-0.6B`
- **LoRA Config**: r=32, alpha=64, dropout=0.1
- **Training**: 4 epochs, batch size 8, learning rate 2e-4
- **Output**: Saves LoRA adapter to `qwen_lora_model/`

**Usage:**
```bash
python finetune.py
```

**Requirements:**
- Expects `final_data.jsonl` in current directory
- Uses Alpaca template for instruction formatting
- GPU with ~8GB VRAM recommended

### 2. Model Merging ([`merge_lora.py`](merge_lora.py))

Merges LoRA weights with base model for deployment:

- **Input**: `qwen_lora_model/` (LoRA adapter)
- **Output**: `qwen3_merged_model/` (merged 16-bit model)

**Usage:**
```bash
python merge_lora.py
```

### 3. Standard Inference ([`inference.py`](inference.py))

Basic inference using the merged model:

- Loads merged model in 8-bit precision
- Optimized for inference with `FastLanguageModel.for_inference()`
- Tests with sample Polymarket prompts

**Usage:**
```bash
python inference.py
```

### 4. High-Performance Inference ([`vllm_inference.py`](vllm_inference.py))

Production-ready inference using vLLM:

- **Features**: Batch processing, interactive mode, performance metrics
- **Model**: Supports both merged and quantized models
- **Output**: Rich formatted responses with timing stats

**Usage:**
```bash
python vllm_inference.py
```

### 5. Streaming Inference ([`stream_inference.py`](stream_inference.py))

Real-time streaming inference using async vLLM:

- **Features**: Token-by-token streaming, async processing, interactive mode
- **Model**: Uses merged model with async engine
- **Output**: Real-time response streaming with completion timing

**Usage:**
```bash
python stream_inference.py
```

## Dataset Format

Training data should be in JSONL format with instruction-input-output structure:

```json
{
  "instruction": "Explain how moneyline betting works on Polymarket",
  "input": "",
  "output": "Moneyline betting involves predicting which team will win..."
}
```

The [`sample_dataset.jsonl`](sample_dataset.jsonl) contains 20 examples about Polymarket betting concepts.

## Model Configuration

- **Base Model**: Qwen3-0.6B (600M parameters)
- **Quantization**: 4-bit NF4 during training, 8-bit for inference
- **Sequence Length**: 512 tokens
- **LoRA Rank**: 32 (targets all attention and MLP layers)
- **Training Precision**: bfloat16

## Hardware Requirements

- **Training**: GPU with 8GB+ VRAM (RTX 3080/4070 or better)
- **Inference**: GPU with 2.5GB+ VRAM
- **CPU**: 8GB+ RAM recommended

## Output Structure

```
qwen_outputs/          # Training checkpoints
qwen_lora_model/       # LoRA adapter weights
qwen3_merged_model/    # Final merged model
```

## Performance Notes

- Training takes ~30-60 minutes on RTX 4090
- vLLM inference: ~1-2 second depending on GPU utilisation
- Standard inference: ~1-2 second