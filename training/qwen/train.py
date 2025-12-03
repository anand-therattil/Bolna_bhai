from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich import box
console = Console()
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# ---------------------------
#  LOAD MODEL
# ---------------------------
console.print(Panel.fit(
    "[bold cyan]🚀 Loading Model...[/bold cyan]",
    border_style="bright_blue",
    box=box.DOUBLE
))

max_seq_length = 512  
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-0.6B",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=torch.bfloat16,
    quantization_config={
        "bnb_4bit_compute_dtype": torch.float16,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,  
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=64,  
    lora_dropout=0.1,  
    use_gradient_checkpointing="unsloth",
)

model_name = "unsloth/Qwen3-0.6B"
console.print(Panel.fit(
    f"[bold cyan]📌 Loaded Model:[/bold cyan] [bold white]{model_name}[/bold white]\n"
    f"[bold green]• Max Seq Length:[/bold green] {max_seq_length}\n"
    f"[bold yellow]• Quantization:[/bold yellow] 4-bit NF4\n"
    f"[bold magenta]• Precision:[/bold magenta] bfloat16\n"
    f"[bold cyan]• LoRA Rank:[/bold cyan] 128\n"
    f"[bold cyan]• LoRA Alpha:[/bold cyan] 256",
    border_style="bright_blue",
    box=box.ROUNDED
))

# ---------------------------
#  LOAD JSONL DATA
# ---------------------------
console.print(Panel.fit(
    "[bold magenta]📄 Loading JSONL Dataset[/bold magenta]",
    border_style="magenta",
    box=box.ROUNDED
))

dataset = load_dataset("json", data_files="final_data.jsonl")["train"]

ALPACA_TEMPLATE = """Below is an instruction about Polymarket. Write a clear, accurate answer.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def format(example):
    return {
        "text": ALPACA_TEMPLATE.format(
            example["instruction"],
            example["input"],
            example["output"]
        )
    }

dataset = dataset.map(format)

# ---------------------------
#  TRAINING
# ---------------------------
console.print(Panel.fit(
    "[bold green]🔥 Starting Training...[/bold green]",
    border_style="green",
    box=box.HEAVY
))

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        optim="adamw_8bit",
        output_dir="qwen_outputs",
        report_to="none",
        save_strategy="epoch",
        save_total_limit=3,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )
)

# Better progress tracking
console.print(f"[bold yellow]Total training steps: ~{len(dataset) * 10 // (2 * 16)}[/bold yellow]")

with Progress() as progress:
    task = progress.add_task("[green]Training model...", total=100)
    trainer.train()
    for _ in range(100):
        progress.update(task, advance=1)

console.print(Panel.fit(
    "[bold green]✅ Training Completed![/bold green]",
    border_style="green",
))

# ---------------------------
#  SAVE MODEL
# ---------------------------
console.print(Panel.fit(
    "[bold yellow]💾 Saving LoRA Model...[/bold yellow]",
    border_style="yellow",
    box=box.MINIMAL
))

model.save_pretrained("qwen_lora_model")
tokenizer.save_pretrained("qwen_lora_model")

console.print(Panel.fit(
    "[bold white on blue]🎉 Model saved to qwen_lora_model/[/bold white on blue]",
    box=box.DOUBLE_EDGE,
    border_style="bright_blue"
))

console.print(Panel.fit(
    f"[bold cyan]📊 Training Statistics:[/bold cyan]\n"
    f"• Epochs: 10\n"
    f"• Effective Batch Size: {2 * 16}\n"
    f"• LoRA Parameters: r=128, alpha=256\n"
    f"• Learning Rate: 1e-4 (cosine schedule)\n"
    f"• Max Sequence Length: {max_seq_length}",
    border_style="cyan",
    box=box.ROUNDED
))



