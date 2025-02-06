# IMPORTS
import sys
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.hypotheses_samples.n_best_transcriptions import hypotheses  # Assumes structured hypotheses
from metrics import evaluate_model  # For metrics evaluation

def rank_transcriptions_distilbert(transcriptions, model, tokenizer):
    """
    Rank transcriptions using a pre-trained DistilBERT model.
    """
    # Tokenize and prepare inputs
    inputs = tokenizer(transcriptions, return_tensors="pt", padding=True, truncation=True)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Run the model and get logit scores
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[:, 1]  # Probability of the positive class
    
    # Get the index of the best transcription
    best_index = torch.argmax(scores).item()
    best_transcription = transcriptions[best_index]
    
    return best_transcription

# Load DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if hypotheses is structured correctly
if isinstance(hypotheses[0], str):
    hypotheses = [{"samples": [hypothesis], "truth": hypothesis} for hypothesis in hypotheses]

# Generate model predictions
model_predictions = []
for data in hypotheses:
    best_transcription = rank_transcriptions_distilbert(data["samples"], model, tokenizer)
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
