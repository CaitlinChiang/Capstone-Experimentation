# IMPORTS
import sys
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.hypotheses_samples.hypotheses import hypotheses

model = GPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

def score_sentence(sentence, model, tokenizer):
  inputs = tokenizer(sentence, return_tensors="pt")
  with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
  return outputs.loss.item()

scores = [(hypothesis, score_sentence(hypothesis, model, tokenizer)) for hypothesis in hypotheses]
best_hypothesis = sorted(scores, key=lambda x: x[1])[0]

print("Best hypothesis:", best_hypothesis[0])
