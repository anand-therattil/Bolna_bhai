import time
import asyncio
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

# Configure async engine
engine_args = AsyncEngineArgs(
    model="./gpt2-finetuned-polymarket-final",
    dtype="float16",
    gpu_memory_utilization=0.025
)

# Initialize async engine
llm = AsyncLLMEngine.from_engine_args(engine_args)

# Sampling configuration
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200,
)

async def generate_response_streaming(instruction: str):
    """Stream tokens as they're generated"""
    prompt = f"<|instruction|>{instruction}<|response|>"
    
    request_id = f"request-{time.time()}"
    
    # Start streaming generation
    results_generator = llm.generate(prompt, params, request_id)
    
    async for request_output in results_generator:
        text = request_output.outputs[0].text
        
        # Clean up special tokens
        if "<|response|>" in text:
            text = text.split("<|response|>")[1]
        text = text.replace("<|endoftext|>", "").strip()
        
        yield text

async def main():
    print("=== Ultra-Fast GPT-2 Streaming Inference with vLLM ===")
    print("Type your instructions (or 'quit' to exit)\n")
    
    while True:
        user_input = input("Instruction: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        print("\n🔄 Streaming response...\n")
        print("Response: ", end="", flush=True)
        
        start = time.time()
        prev_text = ""
        
        async for current_text in generate_response_streaming(user_input):
            # Print only the new tokens
            new_tokens = current_text[len(prev_text):]
            if new_tokens:
                print(new_tokens, end="", flush=True)
            prev_text = current_text
        
        elapsed = time.time() - start
        print(f"\n\n✓ Complete! Time taken: {elapsed:.4f} seconds\n")

if __name__ == "__main__":
    asyncio.run(main())