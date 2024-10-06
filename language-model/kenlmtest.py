import kenlm

model = kenlm.Model('/Users/caitlin/_Caitlin/Code/RobustGER-Experimentation/language-model/tatoeba_3gram.bin')

hypotheses = [
  "As you're leaving, can cache pulls RSI really quickly?",
  "As you're leaving, can cache pulls RSI really quickly?",
  "As you're leaving, can cache pulls RSI really quickly?",
  "As you're leaving, can cache pulls RSI really quickly?",
  "As you're leaving, can cache pulls RSI really quickly?",
  "As you're leaving, can cache pulls RSI really quickly?",
  "As your leaving can cache pulls RSI really quickly.",
  "As your leaving can cache pulls RSI really quickly.",
  "As your leaving can cache pulls RSI really quickly.",
  "As your leaving can cache pulls RSI really quickly."
]

scores = [(hypothesis, model.score(hypothesis)) for hypothesis in hypotheses]
best_hypothesis = sorted(scores, key=lambda x: x[1], reverse=True)[0]
print("Best hypothesis:", best_hypothesis[0])
