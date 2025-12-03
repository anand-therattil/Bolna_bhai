import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load your finetuned model
model_path = './gpt2-finetuned-final'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

# SPEED OPTIMIZATIONS
# 1. Use FP16 if on GPU
if device == 'cuda':
    model = model.half()

# 2. Compile model (PyTorch 2.0+)
try:
    model = torch.compile(model, mode="reduce-overhead")
except:
    pass

# 3. Disable gradients
torch.set_grad_enabled(False)

def generate_response(instruction, max_new_tokens=200, temperature=0.7, top_p=0.9):
    """Generate a response for a given instruction"""
    # Format the prompt
    prompt = f"<|instruction|>{instruction}<|response|>"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    # Generate with optimizations
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the response part
    if '<|response|>' in generated_text:
        response = generated_text.split('<|response|>')[1]
        response = response.replace('<|endoftext|>', '').strip()
        return response
    
    return generated_text

# Interactive mode
if __name__ == "__main__":
    print("=== Fast GPT-2 Inference ===")
    print("Type your instructions (or 'quit' to exit)\n")
    
    while True:
        user_input = input("Instruction: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if user_input:
            start = time.time()
            response = generate_response(user_input)
            elapsed = time.time() - start
            print(f"\nResponse: {response}")
            print(f"Time: {elapsed:.3f} seconds\n")