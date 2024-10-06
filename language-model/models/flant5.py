# IMPORTS
from transformers import T5ForConditionalGeneration, T5Tokenizer
from ...data.hypotheses_samples.hypotheses import hypotheses

model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')

input_text = "Rank or choose the most likely sentence: " + " | ".join(hypotheses)
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=50)
best_hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Best hypothesis:", best_hypothesis)
