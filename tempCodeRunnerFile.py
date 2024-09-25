# Imports
from scripts.asr_inference import run_asr_inference
from scripts.extract_embeddings import extract_embeddings

if __name__ == "__main__":
  # Load the audio file
  audio_path = "data/sp30_train_sn0.wav"

  """
  Run ASR inference to get n-best hypotheses.

  Instead of producing just a single transcription, the ASR model generates the top-N best hypotheses. 
  These hypotheses are different possible transcriptions that the model considers most likely, ranked by their confidence scores.
  """

  n_best_hypotheses = run_asr_inference(audio_path, model_name='base', beam_size=5)
  print("N-best Hypotheses:", n_best_hypotheses)

  # Extract language-space noise embeddings from n-best hypotheses
  embeddings = extract_embeddings(n_best_hypotheses)
  print("Embeddings:", embeddings)
