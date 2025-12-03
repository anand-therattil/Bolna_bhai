# vllm_inference.py
from vllm import LLM, SamplingParams
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import time
import re

console = Console()


def keep_only_first_response(text: str) -> str:
    # Find where the garbage section starts
    split_point = re.search(r"###\s*Input", text, flags=re.IGNORECASE)

    if split_point:
        # Keep everything BEFORE the first "### Input Format"
        text = text[:split_point.start()]

    # Strip trailing whitespace/newlines
    text = text.strip()

    return text

def format_prompt_alpaca(instruction, input_text=""):
    """Format prompt using the same Alpaca template used in training"""
    return f"""Below is an instruction about Polymarket. Write a clear, accurate answer. The input may be empty. 

### Instruction:
{instruction}

### Input :
{input_text}

### Response:
"""

def generate_vllm(llm, prompts, params):
    """Generate responses using vLLM for batch processing"""
    start_time = time.time()
    outputs = llm.generate(prompts, params)
    generation_time = time.time() - start_time
    
    results = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        results.append({
            "prompt": prompt,
            "response": generated_text,
            "tokens": len(output.outputs[0].token_ids)
        })
    
    return results, generation_time

def main():
    # ---------------------------
    # INITIALIZE VLLM
    # ---------------------------
    console.print(Panel.fit(
        "[bold cyan]🚀 Initializing vLLM Engine[/bold cyan]",
        border_style="bright_blue",
        box=box.DOUBLE
    ))

    # Initialize the LLM
    llm = LLM(
        model="qwen3_int8_gptq",
        quantization="compressed-tensors",
        #model="qwen3_merged_model",  # Path to merged model
        trust_remote_code=True,
        dtype="bfloat16",  # or "float16" depending on your GPU
        gpu_memory_utilization=0.1,  # Adjust based on your GPU memory
        max_model_len=512,
    )

    console.print("[bold green]✓ vLLM Engine Loaded![/bold green]")

    # ---------------------------
    # SAMPLING PARAMETERS
    # ---------------------------
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=512,
        repetition_penalty=1.1,
        # stop=["</s>", "\n\n"],  # Add stop tokens if needed
    )

    # ---------------------------
    # TEST THE MODEL
    # ---------------------------
    console.print(Panel.fit(
        "[bold magenta]🧪 Testing Model Performance[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED
    ))

    # Test prompts
    test_prompts = [
        "Explain Polymarket in simple terms.",
        "How to deposit money in polymarket account?",
        "What are prediction markets?",
        "How does Polymarket ensure fairness?",
        "What is the output of 267 * 568968 ?"
    ]

    # Format prompts with Alpaca template
    formatted_prompts = [format_prompt_alpaca(prompt) for prompt in test_prompts]

    # Generate responses
    results, gen_time = generate_vllm(llm, formatted_prompts, sampling_params)

    # Display results in a nice table
    table = Table(title="Model Inference Results", box=box.ROUNDED)
    table.add_column("Prompt", style="cyan", width=40)
    table.add_column("Response", style="green", width=60)
    table.add_column("Tokens", style="yellow")

    for result in results:
        # Extract original prompt (without template)
        original_prompt = result["prompt"].split("### Instruction:\n")[1].split("\n\n### Input :")[0]
        table.add_row(
            original_prompt[:40] + "..." if len(original_prompt) > 40 else original_prompt,
            result["response"][:100] + "..." if len(result["response"]) > 100 else result["response"],
            str(result["tokens"])
        )

    console.print(table)

    # Performance stats
    console.print(Panel.fit(
        f"[bold cyan]📊 Performance Stats:[/bold cyan]\n"
        f"• Total prompts: {len(test_prompts)}\n"
        f"• Total generation time: {gen_time:.2f}s\n"
        f"• Average time per prompt: {gen_time/len(test_prompts):.2f}s\n"
        f"• Throughput: {len(test_prompts)/gen_time:.2f} prompts/second",
        border_style="cyan",
        box=box.ROUNDED
    ))

    # ---------------------------
    # INTERACTIVE MODE (Optional)
    # ---------------------------
    console.print("\n[bold yellow]💬 Interactive Mode - Type 'quit' to exit[/bold yellow]")

    while True:
        user_input = console.input("\n[bold cyan]Enter your prompt:[/bold cyan] ")
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            console.print("[bold red]Goodbye![/bold red]")
            break

        formatted_input = format_prompt_alpaca(user_input)
        # print("Formatted Input:", formatted_input)

        start = time.time()
        outputs = llm.generate([formatted_input], sampling_params)
        end = time.time()
        
        response = outputs[0].outputs[0].text
        response = keep_only_first_response(response)
        
        console.print(Panel(
            f"[bold green]{response}[/bold green]",
            title=f"Response (Generated in {end-start:.2f}s)",
            border_style="green",
            box=box.ROUNDED
        ))

if __name__ == "__main__":
    main()