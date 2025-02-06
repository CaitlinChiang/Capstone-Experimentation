# IMPORTS
import sys
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Correct path to the original-model directory
saved_directory = os.path.join(os.path.dirname(__file__), 'original-model')

# Load the model and tokenizer from the saved directory
model = T5ForConditionalGeneration.from_pretrained(saved_directory)
tokenizer = T5Tokenizer.from_pretrained(saved_directory)

# Add the base directory of the project to the Python path
base_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(base_directory)

# Import the hypotheses from hypotheses_samples
try:
    from data.hypotheses_samples.hypotheses import hypotheses
except ModuleNotFoundError as e:
    print("Error: Unable to import hypotheses. Check the file structure and paths.")
    raise e

# Prepare the input text
input_text = "Rank or choose the most likely sentence: " + " | ".join(hypotheses)
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate outputs and decode the best hypotheses
outputs = model.generate(input_ids, max_length=50, num_return_sequences=5, num_beams=5)
n_best = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Function to calculate the loss score for each sentence
def score_sentence(sentence):
    try:
        # Tokenize the input
        inputs = tokenizer(sentence, return_tensors='pt')
        input_ids = inputs['input_ids']

        # Ensure input is not too long for the model
        if input_ids.size(1) > model.config.n_positions:
            print(f"Sentence too long, truncating to {model.config.n_positions} tokens.")
            input_ids = input_ids[:, :model.config.n_positions]

        # Calculate loss
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # The loss is the average negative log-likelihood per token
            loss = outputs.loss.item()

        return loss
    except Exception as e:
        print(f"Error scoring sentence: {sentence}. Exception: {e}")
        return float('inf')  # Assign a very high loss to problematic sentences

# Score each sentence
scores = []
print("Generated hypotheses (n_best):", n_best)  # Debugging: Print the hypotheses
for sentence in n_best:
    sentence = sentence.strip()  # Remove leading/trailing whitespace
    if not sentence:  # Skip empty sentences
        print("Skipping an empty or invalid sentence.")
        continue
    try:
        loss = score_sentence(sentence)
        scores.append((sentence, loss))
    except Exception as e:
        print(f"Error processing sentence: {sentence}. Exception: {e}")
        continue

# Sort the sentences by their scores (lower loss is better)
sorted_sentences = sorted(scores, key=lambda x: x[1])

# Extract the best hypothesis and its rank
if sorted_sentences:
    best_sentence, best_score = sorted_sentences[0]
    best_rank = 1  # Since it's the top-ranked sentence

    # Print the best sentence, its rank, and score
    print(f"Best sentence (Rank {best_rank}): {best_sentence}")
    print(f"Score: {best_score:.4f}")

    # Optionally, print all sentences with their ranks and scores
    print("\nAll sentences ranked by likelihood:")
    for idx, (sentence, score) in enumerate(sorted_sentences):
        print(f"Rank {idx + 1}: Score: {score:.4f} | Sentence: {sentence}")
else:
    print("No valid sentences to process.")
