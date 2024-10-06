# IMPORTS
import os
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_dir, '../../../data/kenlm_train/tatoeba_sentences.txt')
arpa_output = os.path.join(current_dir, '../../../language-model/output/tatoeba_3gram.arpa')
bin_output = os.path.join(current_dir, '../../../language-model/output/tatoeba_3gram.bin')

train_command = f"/Users/caitlin/_Caitlin/Code/kenlm/build/bin/lmplz -o 3 < {input_file} > {arpa_output}"

try:
    print("Training the 3-gram KenLM model...")
    subprocess.run(train_command, shell=True, check=True)
    print("Model trained successfully!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while training the model: {e}")

binary_command = f"/Users/caitlin/_Caitlin/Code/kenlm/build/bin/build_binary {arpa_output} {bin_output}"

try:
    print("Converting ARPA model to binary format...")
    subprocess.run(binary_command, shell=True, check=True)
    print(f"Binary model saved as {bin_output}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while converting the model: {e}")
