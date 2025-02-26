# IMPORTS
import sys
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer


current_dir = os.path.dirname(os.path.abspath(__file__))
input_files = os.path.join(current_dir, '../../../data/transcripts')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.hypotheses_samples.hypotheses import hypotheses

model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')

input_text = "Rank or choose the most likely sentence: " + " | ".join(hypotheses)
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=50)
best_hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Best hypothesis:", best_hypothesis)
