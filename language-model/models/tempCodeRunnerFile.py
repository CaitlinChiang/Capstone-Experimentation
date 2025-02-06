# IMPORTS
import sys
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.hypotheses_samples.n_best_transcriptions import hypotheses  # Assumes structured hypotheses
from metrics import evaluate_model  # Evaluate model performance

# Load DistilGPT model and tokenizer
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Add a padding token to the tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def score_sentence(sentence, model, tokenizer):
    """
    Compute the loss for a given sentence using DistilGPT.
    """
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()

def rank_transcriptions(transcriptions, model, tokenizer):
    """
    Rank transcriptions based on the language modeling loss and return the best one.
    """
    scores = [(hypothesis, score_sentence(hypothesis, model, tokenizer)) for hypothesis in transcriptions]
    best_hypothesis = sorted(scores, key=lambda x: x[1])[0][0]  # Select the hypothesis with the lowest loss
    return best_hypothesis

# Generate model predictions
model_predictions = []
for data in hypotheses:
    best_transcription = rank_transcriptions(data["samples"], model, tokenizer)
    model_predictions.append(best_transcription)

# Evaluate the model predictions against the ground truth
results = evaluate_model(hypotheses, model_predictions)

# Print the best hypotheses and evaluation results
for i, (data, prediction) in enumerate(zip(hypotheses, model_predictions)):
    print(f"Sample {i + 1}")
    print(f"Ground Truth: {data['truth']}")
    print(f"Best Hypothesis: {prediction}")
    print("-" * 30)

print("\nEvaluation Results:")
print(results)
