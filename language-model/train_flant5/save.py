from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')

# Set save directory
save_directory = "./test-save-flan-t5"
os.makedirs(save_directory, exist_ok=True)

# Save model and tokenizer
try:
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print("Model and tokenizer saved successfully!")
except Exception as e:
    print(f"Error: {e}")
