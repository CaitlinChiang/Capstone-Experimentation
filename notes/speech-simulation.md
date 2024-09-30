# Datasets Used
[GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech)

[MIT Impuse Response Survey](https://huggingface.co/datasets/benjamin-paine/mit-impulse-response-survey)

[AudioSet](https://huggingface.co/datasets/yangwang825/audioset)

[People's speech](https://huggingface.co/datasets/MLCommons/peoples_speech)

[CSTR VCTK Dataset](https://huggingface.co/datasets/CSTR-Edinburgh/vctk)

[TED-LIUM Dataset](https://huggingface.co/datasets/LIUM/tedlium)

[Ambient Noise Dataset](https://huggingface.co/datasets/psiyou/ambient_noise_dataset)


# Other Datasets
[MultiLingual LibriSpeech (English only)](https://huggingface.co/datasets/parler-tts/mls_eng)

[GLOBE Dataset](https://huggingface.co/datasets/MushanW/GLOBE)

[Librispeech](https://huggingface.co/datasets/openslr/librispeech_asr)


# Version 01: Narrated Speech Simulation
1. (Main Speech) Utilize [clean speech](https://huggingface.co/datasets/LIUM/tedlium)
2. (Ambient Noise) Inject [room reverberations](https://huggingface.co/datasets/benjamin-paine/mit-impulse-response-survey)
3. (Random Noise) Inject [random noise](https://huggingface.co/datasets/psiyou/ambient_noise_dataset)
4. (Multi-Talk) Inject [interfering speakers](https://huggingface.co/datasets/MLCommons/peoples_speech)
- Loop noise to match the length of the clean main speech


# Version 02: Spontaneous Accented Speech Simulation
1. (Main Speech) Utilize [accented speech](https://huggingface.co/datasets/CSTR-Edinburgh/vctk)
2. (Ambient Noise) Inject [room reverberations](https://huggingface.co/datasets/benjamin-paine/mit-impulse-response-survey)
3. (Random Noise) Inject [random noise](https://huggingface.co/datasets/psiyou/ambient_noise_dataset)
4. (Multi-Talk) Inject [interfering speakers](https://huggingface.co/datasets/MLCommons/peoples_speech)


# Download Datasets Commands
python -c "from datasets import load_dataset; load_dataset('LIUM/tedlium', 'release1')"

python -c "from datasets import load_dataset; load_dataset('benjamin-paine/mit-impulse-response-survey')"

python -c "from datasets import load_dataset; load_dataset('yangwang825/audioset')"

python -c "from datasets import load_dataset; load_dataset('MLCommons/peoples_speech')"
