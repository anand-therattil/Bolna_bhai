# 🚀 GPT-2 Finetuning & Inference Pipeline

1. **Finetune GPT-2** on instruction → response pairs
2. **Run inference** using:

   * Standard PyTorch (`inference.py`)
   * High-speed vLLM engine (`vllm_inference.py`)
   * Fully **streaming** token-by-token inference (`stream_inference.py`)

---

# 📂 Files Overview

| File                    | Purpose                                                       |
| ----------------------- | ------------------------------------------------------------- |
| **finetune.py**         | Train GPT-2 on your custom `<instruction, response>` dataset. |
| **inference.py**        | Run normal inference using PyTorch, with speed optimizations. |
| **vllm_inference.py**   | Run inference using vLLM, extremely fast.                     |
| **stream_inference.py** | Stream generated tokens in real time using async vLLM.        |
| **sample_dataset.txt**  | Text file containing your `Instruction:` → `Response:` pairs. |

---

# 📘 1. `finetune.py` — Train Your GPT-2 Model

This script **parses your dataset**, **formats it**, **tokenizes it**, and **finetunes GPT-2** using HuggingFace `Trainer`.

### ✔ What this script does

* Loads GPT-2 model & tokenizer
* Parses `sample_dataset.txt` with the format:

```
Instruction: ...
Response: ...
```

* Converts each example into:

```
<|instruction|>YOUR TEXT<|response|>YOUR TEXT<|endoftext|>
```

* Tokenizes and creates dataset
* Splits into train/test
* Finetunes GPT-2
* Saves the final model to:

```
./gpt2-finetuned-final
```

* Tests a sample inference at the end

---

### ▶ How to run training

```bash
python finetune.py
```

Make sure `sample_dataset.txt` exists in the same directory.

---

### 📌 Output Files

After training, you'll get:

```
gpt2-finetuned-final/
    config.json
    pytorch_model.bin
    tokenizer.json
    tokenizer_config.json
```

These are required for all inference scripts.

---

# 📘 2. `inference.py` — Regular PyTorch Inference (Fast)

This script loads your finetuned model and generates outputs using plain PyTorch.

### ✔ Key features

* Loads your trained model
* Moves to GPU if available
* Uses FP16 for speed
* Uses `torch.compile()` if supported
* Interactive chat-like interface
* Extracts only the `<|response|>` section

---

### ▶ How to run

```bash
python inference.py
```

Example:

```
Instruction: What is machine learning?
Response: ...
```

---

# 📘 3. `vllm_inference.py` — Ultra-Fast Inference with vLLM

This script uses the **vLLM engine**, offering:

* Massive speedups (2×–20× faster than HF pipelines)
* Better GPU memory utilization
* Production-grade serving quality

### ✔ What it does

* Loads your finetuned model using vLLM engine
* Generates responses in a single shot
* Strips special tokens
* Measures inference time

---

### ▶ How to run

```bash
python vllm_inference.py
```

Example usage:

```
Instruction: Explain liquidity in simple words.
```

The output includes:

* Response
* Time taken

---

# 📘 4. `stream_inference.py` — Token-Streaming with vLLM (Async)

This script provides **real-time streaming output**, similar to ChatGPT typing.

### ✔ Features

* Asynchronous vLLM engine
* Streams tokens as they are generated
* Useful for UI, terminals, chatbots
* Shows incremental output

---

### ▶ How to run streaming inference

```bash
python stream_inference.py
```

Example stream:

```
Response: L...iquidity refers to how eas...ily an asset...
```

---

# 📘 Dataset Format (`sample_dataset.txt`)

Your training file must follow this structure:

```
Instruction: Explain arbitrage.
Response: Arbitrage is...

Instruction: What is a moneyline bet?
Response: A moneyline bet is...
```

⚠️ No blank spaces inside a pair.
⚠️ Multiple examples can be stacked.

---

# 📘 Prompt Format Used in All Scripts

During training and inference, your model always receives:

```
<|instruction|>YOUR INSTRUCTION<|response|>
```

The model learns to generate everything **after `<|response|>`**.

---

# 📘 When to Use Each Script

| Goal                       | Script                    |
| -------------------------- | ------------------------- |
| Train the model            | `finetune.py`             |
| Simple inference           | `inference.py`            |
| High-performance inference | `vllm_inference.py`       |
| Live token streaming       | `stream_inference.py`     |
| Deploy as API (future)     | Extend vLLM-based scripts |

---

# 📘 Requirements

Install dependencies:

```bash
pip install transformers datasets torch vllm accelerate
```

If using GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

# ✔ Summary

You now have a **full GPT-2 fine-tuning pipeline**, including:

* Dataset parsing
* Model training
* Normal + high-speed + streaming inference
* It takes only 2GB of VRAM, you can lower it even more by reducing the gpu utilization

This structure is scalable and production-ready.

---
