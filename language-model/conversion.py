import csv
import kenlm
import subprocess

# Step 1: Extract English sentences from the eng_sentences.tsv file
input_file = '/Users/caitlin/_Caitlin/Code/RobustGER-Experimentation/language-model/eng_sentences.tsv'
output_file = '/Users/caitlin/_Caitlin/Code/RobustGER-Experimentation/language-model/tatoeba_sentences.txt'

# Open the TSV and extract sentences
with open(input_file, 'r') as tsvfile, open(output_file, 'w') as txtfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        txtfile.write(row[2] + '\n')  # Write only the sentence to the text file

print(f"Extracted sentences saved to {output_file}")