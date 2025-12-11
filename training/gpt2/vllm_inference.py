import time
from vllm import LLM, SamplingParams

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200
)

def generate_response(llm, instruction: str):
    prompt = f"<|instruction|>{instruction}<|response|>"
    start = time.time()
    out = llm.generate([prompt], params)[0]
    elapsed = time.time() - start

    text = out.outputs[0].text
    if "<|response|>" in text:
        text = text.split("<|response|>")[1]
    text = text.replace("<|endoftext|>", "").strip()
    return text, elapsed


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    print("Initializing vLLM engine...")
    llm = LLL = LLM(
        model="./gpt2-finetuned-final",
        dtype="float16",
        gpu_memory_utilization=0.025
    )

    print("=== Ultra-Fast GPT-2 Inference with vLLM ===")
    print("Type your instructions (or 'quit' to exit)\n")

    while True:
        user_input = input("Instruction: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        response, t = generate_response(llm, user_input)

        print("\nResponse:", response)
        print(f"Time taken: {t:.4f} seconds\n")
