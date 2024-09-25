# Imports
from transformers import BertModel, BertTokenizer
import torch

def extract_embeddings(hypotheses, model_name='bert-base-uncased'):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Process each hypothesis
    embeddings = []
    for hypothesis in hypotheses:
        # Tokenize the hypothesis
        inputs = tokenizer(hypothesis, return_tensors="pt", padding=True, truncation=True)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract the embedding of [CLS] token, which is commonly used as a summary of the sentence
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    
    return embeddings
