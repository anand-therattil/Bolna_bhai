import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# ---------------------------
# LOAD MODEL + TOKENIZER
# ---------------------------
model_path = "qwen3_merged_model"   

model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    max_seq_length = 512,
    load_in_4bit = False,        # Merged model is in 16-bit
    load_in_8bit = True,
    fast_inference = True,
    load_in_16bit = False,
    dtype = torch.bfloat16,
)

FastLanguageModel.for_inference(model)   # 2x faster inference

# ---------------------------
# GENERATION FUNCTION
# ---------------------------
def generate(prompt, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------
# TEST THE MODEL
# ---------------------------
prompts = ["Explain Polymarket in simple terms.",
              "What is the capital of France?",
              "How to deposit money in polymarket account?",
              "Write a poem about the sea."]

for prompt in prompts:
    print("\n\n====================\n")
    result = generate(prompt)
    print(result)


