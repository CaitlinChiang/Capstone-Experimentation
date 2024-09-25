# Imports
import sys
import os

# Path to your local Whisper repository
whisper_path = '/Users/caitlin/_Caitlin/Code/whisper'  # Update with the correct path

# Insert the path to the top of the list
sys.path.insert(0, whisper_path)

import whisper

def run_asr_inference(audio_path, model_name='base', beam_size=5):
    # Load the Whisper model
    model = whisper.load_model(model_name)

    # Transcribe the audio file
    options = {"beam_size": beam_size, "best_of": beam_size, "task": "transcribe"}
    result = model.transcribe(audio_path, **options)

    # Extract n-best hypotheses
    n_best_hypotheses = result['text'].split("\n")

    return n_best_hypotheses
