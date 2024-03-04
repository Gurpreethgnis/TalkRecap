import os, whisper, torch
import numpy as np
from tqdm import tqdm
import soundfile as sf
import sounddevice as sd
from pydub import AudioSegment
from pydiar.util.misc import optimize_segments
from pydiar.models import BinaryKeyDiarizationModel, Segment
from transformers import BartForConditionalGeneration, BartTokenizer

def initialize_parameters(summary_length = 250, file_name = 'Sample_1', audio_folder = os.path.join(os.getcwd(),'audio_files')):
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
    
    file_path = os.path.join(audio_folder, file_name+'.wav')
    model, tokenizer = loadSummarizer ()
    
    return summary_length, file_path, audio_folder, model, tokenizer

def get_audio_file(file_path):
    if not os.path.exists(file_path):
        # Record the audio file
        print ("\nStarting to record the audio at "+str(file_path))
        audio_data = record_audio_to_file(file_path)
    else:
        #read audio file
        print ("\nAudio file already exists. Reading the file at "+str(file_path))
        audio_data = read_audio_file(file_path)
    return audio_data

def read_audio_file(file_path):
    audio_data, _ = sf.read(file_path)
    audio_data = audio_data.astype(np.float32)  # Ensure audio data is float32
    return audio_data

def record_audio_to_file(file_path, sampling_rate=16000):
    """
    Records audio from the microphone until stopped by the user and saves it to a file.
    """
    print("Recording... Press Ctrl+C to stop.")
    try:
        # Infinite recording until KeyboardInterrupt
        with sd.InputStream(samplerate=sampling_rate, channels=1, dtype='float32') as stream:
            audio_frames = []
            try:
                while True:
                    frame, _ = stream.read(sampling_rate)  # Read one second at a time
                    audio_frames.append(frame)
            except KeyboardInterrupt:
                print("Recording stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    audio_data = np.concatenate(audio_frames, axis=0)
    sf.write(file_path, audio_data, samplerate=16000)
    return audio_data

def loadSummarizer(model_name = "bart-large-cnn"):
    # Load model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return model,tokenizer

def perform_diarization(file_path):
    # Perform speaker diarization
    segments = perform_speaker_diarization(file_path)
    # Convert segment boundaries to float32
    for segment in segments:
        segment.start = np.float32(segment.start)
        segment.end = np.float32(getattr(segment, 'end', segment.start))
    return segments

def transcribe_audio(audio_data, segments, sampling_rate=16000, model = whisper.load_model("base")):
    """
    Transcribes audio data using the specified Whisper model, considering speaker diarization segments.

    Args:
    - audio_data (np.ndarray): Audio data to transcribe.
    - segments (list): List of Segment objects representing speaker diarization segments.
    - sampling_rate (int): Sampling rate of the audio data.
    - model (whisper.model.Model): Loaded Whisper model for transcription.

    Returns:
    - dict: Transcribed text with speaker information.
    - str: Transcribed text with speaker identifier.
    """
    transcribed_text_dict = {}
    transcribed_text_combined = ""
    for i in tqdm(range(len(segments))):
        segment=segments[i]
        speaker_id = str(segment.speaker_id)
        start_time = segment.start
        end_time = start_time + segment.length  # Calculate end time
        speaker_audio = audio_data[int(start_time * sampling_rate):int(end_time * sampling_rate)]
        result = model.transcribe(speaker_audio, verbose=None)
        text = result["text"]
        if speaker_id not in transcribed_text_dict:
            transcribed_text_dict[speaker_id] = []
        transcribed_text_dict[speaker_id].append(text)
        transcribed_text_combined += f"\n### Speaker {speaker_id}\n{text}\n"
    return transcribed_text_combined, transcribed_text_dict

def perform_speaker_diarization(input_file = "Sample.wav", sample_rate = 16000):
    audio = AudioSegment.from_wav(input_file)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)
    diarization_model = BinaryKeyDiarizationModel()
    segments = diarization_model.diarize(
        sample_rate, np.array(audio.get_array_of_samples())
        )
    optimized_segments = optimize_segments(segments)
    return optimized_segments

def chunk_article(article, tokenizer, max_length):
    # Tokenize the article into tokens
    tokens = tokenizer.tokenize(article)
    # Initialize chunks
    chunks = []
    current_chunk = []
    current_length = 0
    for token in tokens:
        current_chunk.append(token)
        current_length += 1
        if current_length == max_length:
            chunks.append(tokenizer.convert_tokens_to_ids(current_chunk))
            current_chunk = []
            current_length = 0
    # Add the last chunk if it's not empty
    if current_length > 0:
        chunks.append(tokenizer.convert_tokens_to_ids(current_chunk))
    return chunks

def summarize_chunks(chunks, model, tokenizer, summary_percent = 10):
    summaries = []
    for chunk in chunks:
        input_ids = torch.tensor([chunk])
        max_length = np.int32(len(chunk) * summary_percent * 0.01) 
        if max_length < 30:
            max_length = 35
        summary_ids = model.generate(input_ids, num_beams=4, max_length=max_length, min_length=30, length_penalty=2.0, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def summarize_speaker(model, transcribed_dict, tokenizer, summary_percent = 10):
    summary_dict={}
    for key in transcribed_dict:
        #print (key)
        # Chunk the article
        chunks = chunk_article(''.join(transcribed_dict[key]), tokenizer, max_length=1024 - tokenizer.num_special_tokens_to_add())
        # Summarize each chunk
        summaries = summarize_chunks(chunks, model, tokenizer, summary_percent = summary_percent)
        
        # Option to further summarize the summaries or combine them
        final_summary = " ".join(summaries)

        summary_dict[key] = final_summary
        #print(final_summary)
    return summary_dict

def print_results(transcribed_dict, summarized_dict):
    for key in transcribed_dict:
        print ("\n"+"_"*50)
        print ("Speaker: "+key)
        print ("_"*50)
        print ("### Transcribed Text ###\n")
        print ('\n'.join(transcribed_dict[key]))
        print ("."*50)
        print ("### Summarized Text ###\n")
        print (summarized_dict[key])