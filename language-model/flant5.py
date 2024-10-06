from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')

hypotheses = [
  "As you're leaving, can cache pulls RSI really quickly?",
  "As you're leaving, can cache pulls RSI really quickly?",
  "As you're leaving, can cache pulls RSI really quickly?",
  "As you're leaving, can cache pulls RSI really quickly?",
  "As you're leaving, can cache pulls RSI really quickly?",
  "As you're leaving, can cache pulls RSI really quickly?",
  "As your leaving can cache pulls RSI really quickly.",
  "As your leaving can cache pulls RSI really quickly.",
  "As your leaving can cache pulls RSI really quickly.",
  "As your leaving can cache pulls RSI really quickly."
]

input_text = "Rank or choose the most likely sentence: " + " | ".join(hypotheses)
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Best hypothesis:", result)
