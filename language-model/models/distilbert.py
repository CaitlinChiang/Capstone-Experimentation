# IMPORTS
import sys
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.hypotheses_samples.hypotheses import hypotheses

def rank_transcriptions_distilbert(transcriptions, model_name="distilbert-base-uncased"):
    # Load pre-trained DistilBERT model and tokenizer
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
        scores = torch.softmax(outputs.logits, dim=1)[:, 1]  # Select the probability of the positive class if binary
    
    # Get the index of the best transcription
    best_index = torch.argmax(scores).item()
    best_transcription = transcriptions[best_index]
    
    return best_transcription, scores.cpu().numpy()

best_transcription, scores = rank_transcriptions_distilbert(hypotheses)
print("Best Transcription:", best_transcription)
print("Scores:", scores)
