# TalkRecap
Summarize your podcasts/calls effortlessly

TalkRecap is a Python-based script designed to simplify the process of recording, transcribing, and summarizing calls/podcasts. With TalkRecap, you can easily convert lengthy calls/ podcast episodes into concise summaries, saving you time and effort.

## Features:
- Record the audio through the microphone
- Transcribe audio files with speaker diarization
- Summarize transcripts using advanced natural language processing techniques
- Customize summary length and generate speaker-specific summaries

## Installation:

Need to install some special repositories

- Install Whisper:
  ```python
  pip install git+https://github.com/openai/whisper.git

- Install the Forked version of Pydiar
  ```python
  pip install --ignore-requires-python -e git+https://github.com/Gurpreethgnis/pydiar.git#egg=pydiar

## Usage:
1. Record or upload your podcast audio file.
2. Use PodSum to transcribe the audio and perform speaker diarization.
3. Customize summary parameters and generate concise summaries.
4. Incorporate the summaries into your podcast show notes or social media posts.

## Example:
```python
from TalkRecap import transcribe_audio_with_diarization, summarize_text

# Transcribe audio with speaker diarization
transcribed_text = transcribe_audio_with_diarization(audio_data, segments)

# Summarize transcribed text
summarized_text = summarize_text(transcribed_text)
