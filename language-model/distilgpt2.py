import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the DistilGPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
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

def score_sentence(sentence, model, tokenizer):
  inputs = tokenizer(sentence, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
  return outputs.loss.item()

scores = [(hypothesis, score_sentence(hypothesis, model, tokenizer)) for hypothesis in hypotheses]
best_hypothesis = sorted(scores, key=lambda x: x[1])[0]
print("Best hypothesis:", best_hypothesis[0])
