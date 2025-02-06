# Imports
import threading
from queue import Queue

chunk_duration = 10
overlap = 2
buffer = []

# Divide the incoming live audio stream into overlapping chunks of a fixed duration
# Chunking is performed such that no word is missed
def chunk_audio_stream(audio_stream):
  while True:
    # Read audio from the stream in small chunks
    chunk = audio_stream.read(chunk_duration + overlap)
    if not chunk: break
    buffer.append(chunk)


# Use a worker thread or async function to process the audio chunks from the buffer.
# Each chunk is passed to the Whisper ASR model for transcription.
transcription_queue = Queue()

def transcribe_audio(buffer):
    while True:
        if buffer:
            chunk = buffer.pop(0)  # Get the first chunk
            transcription = whisper_model.transcribe(chunk)
            transcription_queue.put(transcription)
