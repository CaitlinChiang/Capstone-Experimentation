# IMPORTS
import os
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from util import (
    load_clean_speech_from_api,
    load_room_reverb_from_api,
    load_random_noise_from_api,
    load_interfering_speaker_from_api,
    loop_to_match_length
)

# Generate final noisy speech with reverb, noise, and interfering speakers
def generate_clean_louder(comparison_file, output_file, noise_level=0.1, speaker_level=0.1, reverb_level=0.3, sr=16000):
    try:
        # Load all audio components from Hugging Face datasets
        clean_speech, sr = load_clean_speech_from_api()
        impulse_response, sr = load_room_reverb_from_api()
        random_noise, sr = load_random_noise_from_api()
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
        sf.write(comparison_file, clean_speech, sr)
        sf.write(output_file, mixed_speech, sr)
        print(f"Noisy speech saved to {output_file}, original speech saved to {comparison_file}")
    
    except Exception as e:
        print(f"Error: {e}")

# Generate the noisy speech
current_dir = os.path.dirname(os.path.abspath(__file__))
clean_speech_dir = os.path.join(current_dir, '../../../data/clean')
generated_speech_dir = os.path.join(current_dir, '../../../data/generated_clean_louder')

clean_speech_path = os.path.join(clean_speech_dir, 'cleaned_speech.wav')
noisy_speech_path = os.path.join(generated_speech_dir, 'output_noisy_speech.wav')
generate_clean_louder(clean_speech_path, noisy_speech_path)
