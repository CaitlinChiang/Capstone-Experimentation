# IMPORTS
import os
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, '../../../data/kenlm_train/eng_sentences.tsv')
output_file = os.path.join(current_dir, '../../../data/kenlm_train/tatoeba_sentences.txt')

with open(input_file, 'r') as tsvfile, open(output_file, 'w') as txtfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        txtfile.write(row[2] + '\n')

print(f"Extracted sentences saved to {output_file}")
