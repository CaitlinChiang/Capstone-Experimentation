import os
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import sys
from datasets import Dataset

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')

# Set save directory
save_directory = "./singapore-save-flan-t5"
os.makedirs(save_directory, exist_ok=True)

base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_directory)

# Directory with transcript data
data_directory = "./data/transcripts"

# Prepare data loading function
def load_data(data_dir):
    samples = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                # Process the text (e.g., split into input-output if needed)
                # For simplicity, use the whole text as the "input" and "output"
                samples.append({"input": text, "output": text})
    return samples

# Load dataset
data = load_data(data_directory)

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data)

# Tokenization function
def preprocess_function(example):
    model_input = tokenizer(
        example["input"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        example["output"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    model_input["labels"] = labels["input_ids"]
    return model_input

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned-flan-t5",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",  # Disable reporting to WandB or similar tools
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),  # Use mixed precision if a GPU is available
    seed=42
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print("Fine-tuned model and tokenizer saved successfully!")
