from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set the `eos_token` as the `pad_token`
tokenizer.pad_token = tokenizer.eos_token

def tokenize_hypotheses(hypotheses):
    """
    Tokenize the hypotheses using a pre-trained GPT-2 tokenizer.
    :param hypotheses: List of cleaned hypotheses.
    :return: Tokenized input ready for the model.
    """
    concatenated_hypotheses = " ".join(hypotheses)
    
    # Tokenize the input with padding enabled
    tokenized_input = tokenizer(concatenated_hypotheses, return_tensors="pt", padding=True)
    
    return tokenized_input

# Example N-best hypotheses
n_best_hypotheses = ["hello world", "helo wrld", "hello worl"]
tokenized_input = tokenize_hypotheses(n_best_hypotheses)

print(tokenized_input)
