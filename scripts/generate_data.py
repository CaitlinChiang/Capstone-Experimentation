# Import
import librosa
import numpy as np
import soundfile as sf

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def normalize_audio(audio):
    # Normalize the audio to a range of -1 to 1
    return audio / np.max(np.abs(audio))

def mix_audio(clean, noise, snr_db):
    # Mix clean speech with noise at a specific SNR
    # Normalize both signals
    clean = normalize_audio(clean)
    noise = normalize_audio(noise)
    
    # Calculate the power of the clean and noise signals
    clean_power = np.sum(clean**2)
    noise_power = np.sum(noise**2)
    
    # Calculate the scaling factor for the noise to achieve the desired SNR
    scaling_factor = np.sqrt(clean_power / (noise_power * 10**(snr_db / 10)))
    noise_scaled = noise * scaling_factor
    
    # Mix the clean signal with scaled noise
    noisy_signal = clean + noise_scaled
    return noisy_signal

def save_audio(file_path, audio, sr):
    sf.write(file_path, audio, sr)

# Example usage
clean_speech, sr = load_audio('clean_speech.wav')
noise, _ = load_audio('noise.wav')

# Mix with SNR of 10 dB
noisy_speech = mix_audio(clean_speech, noise, 10)

# Save the result
save_audio('noisy_speech.wav', noisy_speech, sr)
