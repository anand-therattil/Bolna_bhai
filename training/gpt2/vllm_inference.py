import time
from vllm import LLM, SamplingParams

# Load vLLM model (FP16 for speed)
llm = LLM(
    model="./gpt2-finetuned-final",
    dtype="float16",
    gpu_memory_utilization=0.025
)

# Sampling configuration
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200
)

def generate_response(instruction: str):
    # Build prompt with your training format
    prompt = f"<|instruction|>{instruction}<|response|>"

    # Measure inference time
    start = time.time()
    out = llm.generate([prompt], params)[0]
    elapsed = time.time() - start

    # Extract text
    text = out.outputs[0].text

    # Cut only the response part
    if "<|response|>" in text:
        text = text.split("<|response|>")[1]
    text = text.replace("<|endoftext|>", "").strip()

    return text, elapsed


if __name__ == "__main__":
    print("=== Ultra-Fast GPT-2 Inference with vLLM ===")
    print("Type your instructions (or 'quit' to exit)\n")

    while True:
        user_input = input("Instruction: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        response, t = generate_response(user_input)

        print("\nResponse:", response)
        print(f"Time taken: {t:.4f} seconds\n")
