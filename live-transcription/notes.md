ASR's 2 Forms:

1. Batch ASR: Take a bunch of speech, and produce a bunch of text
2. Streaming ASR: When the model needs to produce an output as the speaker is saying things, with a delay not more than a few seconds
- Expect a lower accuracy 


Issues of Whisper:

- ASR is trained to process audio in batches of 30 seconds
- Cannot feed audio to whisper longer than 30 seconds


The Issue of Chunking into 30 Second Clips:

- Latency is > 30 seconds
- Might split in the middle of a word


whisper_streaming

- Whisper-streaming feeds increasingly larger audio chunks into Whisper until an end-of-sentence marker is detected (complete sentences & better accuracy)
- The LocalAgreement algorithm confirms output tokens only after they are generated in two consecutive audio buffers
- This helps distinguish between confirmed and unconfirmed transcription results, allowing for real-time feedback with potential corrections
- Whisper-streaming uses the previous sentence as prompt tokens for the model, providing additional context and improving accuracy.
