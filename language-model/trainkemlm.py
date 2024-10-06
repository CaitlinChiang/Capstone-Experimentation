import subprocess

# File paths for the input text and output ARPA and binary models
input_file = '/Users/caitlin/_Caitlin/Code/RobustGER-Experimentation/language-model/tatoeba_sentences.txt'
arpa_output = '/Users/caitlin/_Caitlin/Code/RobustGER-Experimentation/language-model/tatoeba_3gram.arpa'
bin_output = '/Users/caitlin/_Caitlin/Code/RobustGER-Experimentation/language-model/tatoeba_3gram.bin'

# Step 1: Train the 3-gram model using KenLM's lmplz tool
# train_command = f"./build/bin/lmplz -o 3 < {input_file} > {arpa_output}"

# Update the paths to where KenLM binaries are located
train_command = f"/Users/caitlin/_Caitlin/Code/kenlm/build/bin/lmplz -o 3 < {input_file} > {arpa_output}"


try:
    print("Training the 3-gram KenLM model...")
    subprocess.run(train_command, shell=True, check=True)
    print("Model trained successfully!")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while training the model: {e}")

# Step 2: Convert the ARPA model to binary format using build_binary
binary_command = f"/Users/caitlin/_Caitlin/Code/kenlm/build/bin/build_binary {arpa_output} {bin_output}"

try:
    print("Converting ARPA model to binary format...")
    subprocess.run(binary_command, shell=True, check=True)
    print(f"Binary model saved as {bin_output}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while converting the model: {e}")
