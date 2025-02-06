# IMPORTS
import sys
import os
import kenlm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data.hypotheses_samples.hypotheses import hypotheses

model = kenlm.Model('/Users/caitlin/_Caitlin/Code/RobustGER-Experimentation/language-model/train_kenlm/output/tatoeba_3gram.bin')

scores = [(hypothesis, model.score(hypothesis)) for hypothesis in hypotheses]
best_hypothesis = sorted(scores, key=lambda x: x[1], reverse=True)[0]

print("Best hypothesis:", best_hypothesis[0])

# UNDERSTAND INSTRUCTIONS