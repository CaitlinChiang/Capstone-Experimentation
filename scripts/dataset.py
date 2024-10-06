from datasets import load_dataset

# Load the SEA-LION-Pile dataset
dataset = load_dataset("aisingapore/sea-lion-pile")

# Print the first entry of the dataset
print(dataset['train'][0])
