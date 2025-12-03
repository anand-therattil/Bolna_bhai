from unsloth import FastLanguageModel
import torch

lora_path     = "qwen_lora_model"
merged_output = "qwen3_merged_model"
max_seq_length = 512

print("🚀 Loading your LoRA model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = lora_path,
    max_seq_length = max_seq_length,
    load_in_4bit = True,        # MUST match training load
    dtype = torch.bfloat16,
)

print("🔗 Merging LoRA → base model (16-bit)...")
model.save_pretrained_merged(
    merged_output,
    tokenizer,
    save_method = "merged_16bit",
)

print("🎉 Merge complete:", merged_output)
