# IMPORTS
import sys
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.hypotheses_samples.n_best_transcriptions import hypotheses
from metrics import evaluate_model

def rank_transcriptions(transcriptions, model_name="microsoft/MiniLM-L12-H384-uncased"):
    """
    Rank transcriptions using a pre-trained sequence classification model.
    """
    # Load pre-trained MiniLM model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize and prepare inputs
    inputs = tokenizer(transcriptions, return_tensors="pt", padding=True, truncation=True)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Run the model and get logit scores
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[:, 1]  # Select the probability of the positive class
    
    # Get the index of the best transcription
    best_index = torch.argmax(scores).item()
    best_transcription = transcriptions[best_index]
    
    return best_transcription, scores.cpu().numpy()

# Generate model predictions
model_predictions = []
for data in hypotheses:
    best_transcription, _ = rank_transcriptions(data["samples"])
    model_predictions.append(best_transcription)

# Evaluate the model
results = evaluate_model(hypotheses, model_predictions)
print("Evaluation Results:", results)