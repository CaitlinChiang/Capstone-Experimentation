from datasets import load_dataset
import librosa
import numpy as np
import soundfile as sf
import random
from scipy.signal import fftconvolve

# API-based loading functions (for using Hugging Face datasets)
def load_clean_speech_from_api():
    dataset = load_dataset("LIUM/tedlium", "release1", split="train")
    clean_speech = dataset[0]['audio']['array']
    sr = dataset[0]['audio']['sampling_rate']
    return clean_speech, sr
    
def load_room_reverb_from_api():
    dataset = load_dataset("benjamin-paine/mit-impulse-response-survey")
    impulse_response = dataset['train'][0]['audio']['array']
    sr = dataset['train'][0]['audio']['sampling_rate']
    return impulse_response, sr

def load_random_noise_from_api():
    dataset = load_dataset("yangwang825/audioset")
    random_noise = dataset['train'][0]['audio']['array']
    sr = dataset['train'][0]['audio']['sampling_rate']
    return random_noise, sr

def load_interfering_speaker_from_api():
    dataset = load_dataset("MLCommons/peoples_speech")
    interfering_speaker = dataset['train'][0]['audio']['array']
    sr = dataset['train'][0]['audio']['sampling_rate']
    return interfering_speaker, sr

# Function to loop the audio until it matches the length of the reference audio (clean speech)
def loop_to_match_length(audio, target_length):
    if len(audio) < target_length:
        repeats = int(np.ceil(target_length / len(audio)))
        audio = np.tile(audio, repeats)
    return audio[:target_length]

# Generate final noisy speech with reverb, noise, and interfering speakers
def generate_narrated_speech(output_file, noise_level=0.03, speaker_level=0.04, reverb_level=0.2, sr=16000):
    try:
        # Load all audio components from Hugging Face datasets
        clean_speech, sr = load_clean_speech_from_api()
        impulse_response, sr = load_room_reverb_from_api()
        random_noise, sr = load_random_noise_from_api()  # Load random noise from AudioSet API
        interfering_speaker, sr = load_interfering_speaker_from_api()

        # Apply room reverb (scaled down such that clean speech is dominant)
        reverb_speech = fftconvolve(clean_speech, impulse_response)[:len(clean_speech)]
        reverb_speech *= reverb_level

        # Loop random noise and interfering speaker to match the length of clean speech
        random_noise_signal = loop_to_match_length(random_noise, len(clean_speech))
        interfering_speaker_signal = loop_to_match_length(interfering_speaker, len(clean_speech))

        # Scale down the random noise and interfering speaker signals
        random_noise_signal *= noise_level
        interfering_speaker_signal *= speaker_level

        # Final mix: clean speech (dominant) + reverb + noise + interfering speaker
        mixed_speech = clean_speech + reverb_speech + random_noise_signal + interfering_speaker_signal

        # Normalize the output to avoid clipping
        mixed_speech = mixed_speech / np.max(np.abs(mixed_speech))

        # Save the final noisy speech to the output file
        sf.write(output_file, mixed_speech, sr)
        print(f"Noisy speech saved to {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

# Generate the noisy speech
output_path = 'output_noisy_speech.wav'
generate_narrated_speech(output_path)