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
import os
from TalkRecap.utils import read_audio_file, record_audio_to_file, loadSummarizer, transcribe_audio_with_diarization, perform_diarization, summarize_speaker

# Initialize parameters
summary_length = 250
file_name = 'Sample_1'
audio_folder = os.path.join(os.getcwd(),'audio_files') # Change this for storing the recordings to a different folder
file_path = os.path.join(audio_folder, file_name+'.wav')
model, tokenizer = loadSummarizer ()

# Get audio file
print ('-'*50)
audio_data = get_audio_file(file_path)

# Perform speaker diarization
print ('-'*50)
print ("\n Performing speaker diarization ")
segments = perform_diarization(file_path)

# Transcribe audio with speaker information
print ('-'*50)
print ("\n Transcribing audio ")
transcribed_text, transcribed_dict = transcribe_audio_with_diarization(audio_data, segments)

# Transcribe audio with speaker information
print ('-'*50)
print ("\n summarizing each speaker ")
summarized_dict = summarize_speaker(model, transcribed_dict,tokenizer, summary_length = summary_length)
