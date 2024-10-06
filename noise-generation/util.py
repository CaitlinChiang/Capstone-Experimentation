# IMPORTS
from datasets import load_dataset
import numpy as np

# API-based loading functions (for using Hugging Face datasets)
def load_clean_speech_from_api():
    dataset = load_dataset("LIUM/tedlium", "release1", split="train", streaming=True, trust_remote_code=True)
    sample = next(iter(dataset))
    clean_speech = sample.get('audio', None)
    sr = sample['audio']['sampling_rate']
    return clean_speech['array'], sr

def load_room_reverb_from_api():
    dataset = load_dataset("benjamin-paine/mit-impulse-response-survey", split="train", streaming=True)
    sample = next(iter(dataset))
    impulse_response = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    return impulse_response, sr

# Function to load filenames from AudioSet and download audio
def load_random_noise_from_api():
    dataset = load_dataset("psiyou/ambient_noise_dataset", split="train", streaming=True)
    sample = next(iter(dataset))
    random_noise = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    return random_noise, sr

def load_interfering_speaker_from_api():
    dataset = load_dataset("MLCommons/peoples_speech", split="train", streaming=True)
    sample = next(iter(dataset))
    interfering_speaker = sample['audio']['array']
    sr = sample['audio']['sampling_rate']
    return interfering_speaker, sr

# Function to loop the audio until it matches the length of the reference audio (clean speech)
def loop_to_match_length(audio, target_length):
    if len(audio) < target_length:
        repeats = int(np.ceil(target_length / len(audio)))
        audio = np.tile(audio, repeats)
    return audio[:target_length]
