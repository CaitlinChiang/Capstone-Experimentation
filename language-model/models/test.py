# IMPORTS
import sys
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.hypotheses_samples.n_best_transcriptions import hypotheses
from metrics import evaluate_model

def rank_transcriptions(transcriptions, model_type="minilm", model_name="microsoft/MiniLM-L12-H384-uncased"):
    """
    Rank transcriptions using either MiniLM (classification) or DistilGPT (language modeling).
    """
    # Load the appropriate model and tokenizer
    if model_type == "minilm":
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == "distilgpt":
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Score transcriptions
    scores = []
    for transcription in transcriptions:
        inputs = tokenizer(transcription, return_tensors="pt", padding=True, truncation=True).to(device)
        
        if model_type == "minilm":
            # Use classification logits
            with torch.no_grad():
                outputs = model(**inputs)
                score = torch.softmax(outputs.logits, dim=1)[:, 1].item()  # Positive class score
        elif model_type == "distilgpt":
            # Use language modeling loss
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                score = outputs.loss.item()  # Lower loss is better
        
        scores.append((transcription, score))
    
    # Sort hypotheses (ascending for DistilGPT, descending for MiniLM)
    if model_type == "distilgpt":
        best_transcription = sorted(scores, key=lambda x: x[1])[0][0]  # Lower loss is better
    elif model_type == "minilm":
        best_transcription = sorted(scores, key=lambda x: x[1], reverse=True)[0][0]  # Higher score is better
    
    return best_transcription, [s[1] for s in scores]

# Generate model predictions
model_type = "distilgpt"  # Change to "minilm" for MiniLM
model_name = "distilgpt2" if model_type == "distilgpt" else "microsoft/MiniLM-L12-H384-uncased"

model_predictions = []
for data in hypotheses:
    best_transcription, _ = rank_transcriptions(data["samples"], model_type=model_type, model_name=model_name)
    model_predictions.append(best_transcription)

# Evaluate the model
results = evaluate_model(hypotheses, model_predictions)
print("Evaluation Results:", results)
