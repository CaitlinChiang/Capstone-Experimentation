# IMPORTS
import os
from datasets import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from metrics import evaluate_model  # For evaluation metrics
import sys
import torch

# Load BART tokenizer and model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def score_sentence(sentence, model, tokenizer):
    """
    Compute the likelihood of a sentence using BART. Lower loss indicates better likelihood.
    """
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    labels = inputs["input_ids"].clone()  # Labels are identical to input for scoring
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], labels=labels)
    return outputs.loss.item()  # Lower is better

def rank_transcriptions_bart(transcriptions, model, tokenizer):
    """
    Rank transcriptions using BART by scoring each hypothesis and selecting the best one.
    """
    scores = [(hypothesis, score_sentence(hypothesis, model, tokenizer)) for hypothesis in transcriptions]
    best_hypothesis = sorted(scores, key=lambda x: x[1])[0][0]  # Select the hypothesis with the lowest loss
    return best_hypothesis

# Load hypotheses dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.hypotheses_samples.n_best_transcriptions import hypotheses  # Assumes structured hypotheses

# Check if hypotheses is structured correctly
if isinstance(hypotheses[0], str):
    hypotheses = [{"samples": [hypothesis], "truth": hypothesis} for hypothesis in hypotheses]

# Generate model predictions
model_predictions = []
for data in hypotheses:
    best_transcription = rank_transcriptions_bart(data["samples"], model, tokenizer)
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
