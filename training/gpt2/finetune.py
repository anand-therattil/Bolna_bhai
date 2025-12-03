import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import re

# Load model and tokenizer
model_name = "openai-community/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token (GPT-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Parse your txt file
def parse_instruction_response_file(file_path):
    """Parse the instruction-response format from txt file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by "Instruction:" to get each example
    examples = re.split(r'\nInstruction:', content)
    examples = [ex.strip() for ex in examples if ex.strip()]
    
    data = []
    for example in examples:
        # Handle first example which may not have leading "Instruction:"
        if not example.startswith('Instruction:'):
            example = 'Instruction:' + example
        
        # Split into instruction and response
        parts = example.split('Response:', 1)
        if len(parts) == 2:
            instruction = parts[0].replace('Instruction:', '').strip()
            response = parts[1].strip()
            data.append({
                'instruction': instruction,
                'response': response
            })
    
    return data

# Load your data
data = parse_instruction_response_file('sample_dataset.txt')

# Format data for training with special tokens
def format_instruction(example):
    """Format as: <|instruction|>TEXT<|response|>TEXT<|endoftext|>"""
    text = f"<|instruction|>{example['instruction']}<|response|>{example['response']}<|endoftext|>"
    return {'text': text}

# Create dataset
formatted_data = [format_instruction(ex) for ex in data]
dataset = Dataset.from_list(formatted_data)

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length'
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Split into train/validation (90/10 split)
split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./gpt2-finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='steps',
    eval_steps=50,
    save_steps=100,
    save_total_limit=3,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    report_to='none',  # Change to 'wandb' if you use Weights & Biases
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
    data_collator=data_collator,
)

# Train
print("Starting training...")
trainer.train()

# Save model
trainer.save_model('./gpt2-finetuned-final')
tokenizer.save_pretrained('./gpt2-finetuned-final')

print("Training complete! Model saved to ./gpt2-finetuned-final")

# Test inference
print("\n--- Testing the model ---")
model.eval()
test_instruction = "Explain how moneyline betting works"
prompt = f"<|instruction|>{test_instruction}<|response|>"

inputs = tokenizer(prompt, return_tensors='pt')
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}
    model = model.cuda()

outputs = model.generate(
    **inputs,
    max_length=200,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"\nPrompt: {test_instruction}")
print(f"Generated: {generated_text}")