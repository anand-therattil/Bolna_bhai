import time
import asyncio
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model="./gpt2-finetuned-final",
    dtype="float16",
    gpu_memory_utilization=0.025
)

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=200,
)

async def generate_response_streaming(llm, instruction: str):
    prompt = f"<|instruction|>{instruction}<|response|>"
    request_id = f"request-{time.time()}"
    results_generator = llm.generate(prompt, params, request_id)

    async for request_output in results_generator:
        text = request_output.outputs[0].text
        if "<|response|>" in text:
            text = text.split("<|response|>")[1]
        text = text.replace("<|endoftext|>", "").strip()
        yield text


async def main():
    print("Initializing vLLM engine...")

    llm = AsyncLLMEngine.from_engine_args(engine_args)

    print("=== Ultra-Fast GPT-2 Streaming Inference with vLLM ===")

    while True:
        user_input = input("Instruction: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        print("\n🔄 Streaming response...\nResponse: ", end="", flush=True)

        prev_text = ""
        start = time.time()

        async for current_text in generate_response_streaming(llm, user_input):
            new = current_text[len(prev_text):]
            if new:
                print(new, end="", flush=True)
            prev_text = current_text

        print(f"\n\n✓ Completed in {time.time() - start:.3f} sec\n")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    asyncio.run(main())
