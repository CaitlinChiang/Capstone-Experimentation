# IMPORTS
import sys
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.hypotheses_samples.n_best_transcriptions import hypotheses  # Assumes structured hypotheses
from metrics import evaluate_model  # Evaluate model performance

# Load T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')

def score_sentence(sentence, model, tokenizer):
    """
    Compute the score for a sentence using T5. Lower loss indicates better likelihood.
    """
    input_text = f"Score the likelihood of the sentence: {sentence}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    labels = input_ids.clone()  # Labels are identical to the input for scoring
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss.item()  # Lower is better

def rank_transcriptions(transcriptions, model, tokenizer):
    """
    Rank transcriptions based on the T5-computed likelihood and return the best one.
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
