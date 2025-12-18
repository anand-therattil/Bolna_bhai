import time
import asyncio
import re
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs


# -----------------------
# Prompt Formatting
# -----------------------
def format_prompt_alpaca(instruction, input_text=""):
    return f"""Below is an instruction about Polymarket. Write a clear, accurate answer. The input may be empty.

### Instruction:
{instruction}

### Input :
{input_text}

### Response:
"""


def keep_only_first_response(text: str) -> str:
    split_point = re.search(r"###\s*Input", text, flags=re.IGNORECASE)
    if split_point:
        text = text[:split_point.start()]
    return text.strip()


# -----------------------
# vLLM Async Engine Setup
# -----------------------
engine_args = AsyncEngineArgs(
    model="qwen3_merged_model",       # <---- your local model path
    dtype="float16",
    gpu_memory_utilization=0.025,
    trust_remote_code=True,
    max_model_len=512,
)

params = SamplingParams(
    temperature=0.1,
    top_p=0.9,
    max_tokens=512,
    repetition_penalty=1.1,
)


# -----------------------
# Streaming Generator
# -----------------------
async def generate_response_streaming(llm, instruction: str):
    prompt = format_prompt_alpaca(instruction)

    request_id = f"request-{time.time()}"
    result_stream = llm.generate(prompt, params, request_id)

    async for request_output in result_stream:
        text = request_output.outputs[0].text
        cleaned = keep_only_first_response(text)
        yield cleaned


# -----------------------
# Main Interactive Loop
# -----------------------
async def main():
    print("Initializing vLLM async engine...")
    llm = AsyncLLMEngine.from_engine_args(engine_args)

    print("\n=== 🚀 Streaming vLLM Inference (Local Model) ===\n")

    while True:
        user_input = input("Enter instruction (or quit): ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            break

        print("\n🔄 Streaming Response:\n")
        prev_text = ""
        start = time.time()

        async for partial in generate_response_streaming(llm, user_input):
            new = partial[len(prev_text):]
            if new:
                print(new, end="", flush=True)
            prev_text = partial

        print(f"\n\n✓ Completed in {time.time() - start:.3f} sec\n")


# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    asyncio.run(main())