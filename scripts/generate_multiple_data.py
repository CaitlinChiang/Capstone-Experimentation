import os
import subprocess
from datasets import load_dataset
import librosa
import numpy as np
import soundfile as sf
import random
from scipy.signal import fftconvolve

# Create directories for saving clean and generated data
os.makedirs('data/clean', exist_ok=True)
os.makedirs('data/generated', exist_ok=True)

# API-based loading functions (for using Hugging Face datasets)
def load_clean_speech_from_api():
    dataset = load_dataset("LIUM/tedlium", "release1", split="train", streaming=True, trust_remote_code=True)
    return dataset

def load_room_reverb_from_api():
    dataset = load_dataset("benjamin-paine/mit-impulse-response-survey", split="train", streaming=True)
    return dataset

def load_random_noise_from_api():
    dataset = load_dataset("psiyou/ambient_noise_dataset", split="train", streaming=True)
    return dataset

def load_interfering_speaker_from_api():
    dataset = load_dataset("MLCommons/peoples_speech", split="train", streaming=True)
    return dataset

# Function to loop the audio until it matches the length of the reference audio (clean speech)
def loop_to_match_length(audio, target_length):
    if len(audio) < target_length:
        repeats = int(np.ceil(target_length / len(audio)))
        audio = np.tile(audio, repeats)
    return audio[:target_length]

# Generate final noisy speech with reverb, noise, and interfering speakers
def generate_narrated_speech(clean_speech, impulse_response, random_noise, interfering_speaker, clean_output_path, noisy_output_path, noise_level=0.1, speaker_level=0.1, reverb_level=0.3, sr=16000):
    try:
        # Apply room reverb
        reverb_speech = fftconvolve(clean_speech, impulse_response)[:len(clean_speech)]
        
        # Loop random noise and interfering speaker to match the length of clean speech
        random_noise_signal = loop_to_match_length(random_noise, len(clean_speech))
        interfering_speaker_signal = loop_to_match_length(interfering_speaker, len(clean_speech))

        # Final mix: clean speech + reverb + noise + interfering speaker
        mixed_speech = clean_speech + reverb_speech + random_noise_signal + interfering_speaker_signal

        # Normalize the output to avoid clipping
        mixed_speech = mixed_speech / np.max(np.abs(mixed_speech))

        # Save the clean and noisy speech to output files
        sf.write(clean_output_path, clean_speech, sr)
        sf.write(noisy_output_path, mixed_speech, sr)
        print(f"Noisy speech saved to {noisy_output_path}, clean speech saved to {clean_output_path}")
    
    except Exception as e:
        print(f"Error: {e}")

# Load datasets
clean_speech_dataset = load_clean_speech_from_api()
room_reverb_dataset = load_room_reverb_from_api()
random_noise_dataset = load_random_noise_from_api()
interfering_speaker_dataset = load_interfering_speaker_from_api()

# Process the first 10 samples
for idx, clean_sample in enumerate(clean_speech_dataset.take(10)):
    try:
        clean_speech = clean_sample['audio']['array']
        sr = clean_sample['audio']['sampling_rate']

        # Get the corresponding reverb, noise, and interfering speaker
        reverb_sample = next(iter(room_reverb_dataset))
        impulse_response = reverb_sample['audio']['array']

        noise_sample = next(iter(random_noise_dataset))
        random_noise = noise_sample['audio']['array']

        speaker_sample = next(iter(interfering_speaker_dataset))
        interfering_speaker = speaker_sample['audio']['array']

        # File paths for clean and noisy speech
        clean_output_path = f'data/clean/clean_speech_{idx+1}.wav'
        noisy_output_path = f'data/generated/noisy_speech_{idx+1}.wav'

        # Generate and save narrated speech
        generate_narrated_speech(clean_speech, impulse_response, random_noise, interfering_speaker, clean_output_path, noisy_output_path, sr=sr)

    except Exception as e:
        print(f"Error processing sample {idx+1}: {e}")
