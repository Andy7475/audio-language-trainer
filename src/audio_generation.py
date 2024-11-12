import asyncio
from datetime import datetime
import html
import io
import multiprocessing
import os
import sys
import time
import uuid
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import IPython.display as ipd
import librosa
import numpy as np
import soundfile as sf
from google.cloud import texttospeech, texttospeech_v1
from mutagen.mp4 import MP4, MP4Cover
from pydub import AudioSegment

from src.config_loader import config

def clean_tts_text(text: str) -> str:
    """
    Clean and prepare text for TTS processing by:
    1. Decoding HTML entities
    2. Handling any special characters or formatting
    
    Args:
        text: Input text that may contain HTML entities or special characters
        
    Returns:
        Cleaned text ready for TTS processing
    """
    # Decode HTML entities (like &#39; to ')
    cleaned_text = html.unescape(text)
    
    # Add any additional text cleaning steps here if needed
    # For example, handling other special characters or formatting
    
    return cleaned_text

def clean_translated_content(
    content: Union[str, Tuple[str, str], List[Dict[str, str]]]
) -> Union[str, Tuple[str, str], List[Dict[str, str]]]:
    """
    Clean translated content in various formats:
    - Single string
    - Tuple of (original, translated)
    - List of dialogue dictionaries
    
    Args:
        content: Translated content in various formats
        
    Returns:
        Cleaned content in the same format as input
    """
    if isinstance(content, str):
        return clean_tts_text(content)
    elif isinstance(content, tuple):
        return (content[0], clean_tts_text(content[1]))
    elif isinstance(content, list) and all(isinstance(d, dict) for d in content):
        return [
            {**d, 'text': clean_tts_text(d['text'])}
            for d in content
        ]
    else:
        raise ValueError(f"Unsupported content format: {type(content)}")
    
def setup_ffmpeg():
    ffmpeg_path = r"C:\Program Files\ffmpeg-7.0-essentials_build\bin"

    if os.path.exists(ffmpeg_path):
        # Add FFmpeg to the PATH
        os.environ["PATH"] += os.pathsep + ffmpeg_path
        print(f"FFmpeg path added to system PATH: {ffmpeg_path}")
    else:
        print(f"FFmpeg path not found: {ffmpeg_path}")
        print("Please check the installation directory.")


setup_ffmpeg()


def text_to_speech_worker(
    text: str, language_code: str, voice_name: str, speaking_rate: float
) -> bytes:
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=speaking_rate
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content


def text_to_speech_multiprocessing(
    texts: List[str],
    language_codes: List[str],
    voice_names: List[str],
    speaking_rates: List[float],
) -> List[AudioSegment]:
    with multiprocessing.Pool() as pool:
        audio_contents = pool.starmap(
            text_to_speech_worker,
            zip(texts, language_codes, voice_names, speaking_rates),
        )

    return [AudioSegment.from_mp3(io.BytesIO(content)) for content in audio_contents]


async def async_generate_translated_phrase_audio(
    translated_phrase: Tuple[str, str],
    english_voice_models: Dict = None,
    target_voice_models: Dict = None,
) -> List[AudioSegment]:
    """
    Generate audio for translated phrases, handling HTML entities and special characters.
    
    Args:
        translated_phrase: Tuple of (original_text, translated_text)
        english_voice_models: Configuration for English TTS voices
        target_voice_models: Configuration for target language TTS voices
        
    Returns:
        List of AudioSegment objects containing the generated audio
    """
    # Clean the translated text while preserving the original format
    cleaned_phrase = clean_translated_content(translated_phrase)
    
    texts = [
        cleaned_phrase[0],  # Original English text
        cleaned_phrase[1],  # Translated text (slow)
        cleaned_phrase[1]   # Translated text (normal)
    ]
    
    if english_voice_models is None:
        english_voice_models = config.english_voice_models
    if target_voice_models is None:
        target_voice_models = config.target_language_voice_models

    language_codes = [
        english_voice_models["language_code"],
        target_voice_models["language_code"],
        target_voice_models["language_code"],
    ]
    
    voice_names = [
        english_voice_models["male_voice"],
        target_voice_models["female_voice"],
        target_voice_models["female_voice"],
    ]
    
    speaking_rates = [0.9, config.SPEAKING_RATE_SLOW, 1.0]

    loop = asyncio.get_event_loop()
    audio_segments = await loop.run_in_executor(
        None,
        partial(
            text_to_speech_multiprocessing,
            texts,
            language_codes,
            voice_names,
            speaking_rates,
        ),
    )

    return audio_segments


async def async_process_phrases(phrases, max_concurrency=30):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def limited_task(phrase):
        async with semaphore:
            return await async_generate_translated_phrase_audio(phrase)

    return await asyncio.gather(*[limited_task(phrase) for phrase in phrases])


def text_to_speech(
    text: str,
    language_code: str = None,
    voice_name: str = None,
    speaking_rate: float = 1.0,
) -> AudioSegment:
    client = texttospeech.TextToSpeechClient()

    # Use config values if parameters are not provided
    if language_code is None:
        language_code = config.english_voice_models["language_code"]
    if voice_name is None:
        voice_name = config.english_voice_models["male_voice"]

    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=speaking_rate
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Convert the response to an AudioSegment
    audio_segment = AudioSegment.from_mp3(io.BytesIO(response.audio_content))

    return audio_segment


def export_audio(final_audio: AudioSegment, filename: str = None) -> str:
    """
    Saves the final_audio AudioSegment as an MP3 file.

    Args:
        final_audio (AudioSegment): The audio to be saved.
        filename (str, optional): The filename to save the audio as. If not provided, a UUID will be used.

    Returns:
        str: The filename of the saved audio file.
    """
    if filename is None:
        unique_id = uuid.uuid4()
        filename = f"{unique_id}.mp3"

    final_audio.export(filename, format="mp3")
    return filename


def play_audio(segment: AudioSegment, filename: str = None, autoplay: bool = False):
    """
    Plays an MP3 clip in the Jupyter notebook.

    Args:
        segment (AudioSegment): The audio segment to play.
        filename (str, optional): The filename to save the audio as. If not provided, a temporary file will be created.
        autoplay (bool, optional): Whether to autoplay the audio. Defaults to False.

    Returns:
        IPython.display.Audio: An IPython Audio widget for playing the audio.
    """
    temp_file = filename is None
    filename = export_audio(segment, filename)

    audio = ipd.Audio(filename, autoplay=autoplay)

    if temp_file:
        # Schedule the temporary file for deletion after it's played
        audio._repr_html_()  # This triggers the audio to load in the notebook
        os.remove(filename)

    return audio


def join_audio_segments(audio_segments: list[AudioSegment], gap_ms=100) -> AudioSegment:
    """Joins audio segments together with a tiny gap between each one in ms
    Returns a single joined up audio segment"""
    gap_audio = AudioSegment.silent(duration=gap_ms)
    audio_with_gap = [audio_seg + gap_audio for audio_seg in audio_segments]
    return sum(audio_with_gap)


def speed_up_audio(
    audio_segment: AudioSegment, speed_factor: float = 2.0
) -> AudioSegment:
    """
    Speed up an AudioSegment without changing its pitch, with added de-reverb.

    :param audio_segment: Input AudioSegment
    :param speed_factor: Factor by which to speed up the audio (e.g., 2.0 for double speed)
    :return: Sped up AudioSegment
    """
    # Convert AudioSegment to numpy array
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)

    # Normalize the audio data
    samples = samples / np.iinfo(audio_segment.array_type).max

    # Check if audio is stereo and convert to mono if it is
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)

    # Get sample rate
    sample_rate = audio_segment.frame_rate

    # Simple de-reverb using spectral subtraction
    D = librosa.stft(samples)
    D_mag, D_phase = librosa.magphase(D)

    # Estimate and subtract the reverb
    reverb_time = 0.3  # Adjust this value to control de-reverb strength (in seconds)
    freq = librosa.fft_frequencies(sr=sample_rate)
    reverb_decay = np.exp(
        -np.outer(freq, np.arange(D.shape[1]) / sample_rate) / reverb_time
    )
    D_mag_dereverb = np.maximum(
        D_mag - np.mean(D_mag, axis=1, keepdims=True) * reverb_decay, 0
    )

    # Reconstruct the signal
    D_dereverb = D_mag_dereverb * D_phase
    samples = librosa.istft(D_dereverb)

    # Time stretch audio using librosa
    stretched_audio = librosa.effects.time_stretch(samples, rate=speed_factor)

    # Convert back to int16 for AudioSegment
    stretched_audio = (stretched_audio * np.iinfo(np.int16).max).astype(np.int16)

    # Convert back to AudioSegment
    buffer = io.BytesIO()
    sf.write(buffer, stretched_audio, sample_rate, format="wav", subtype="PCM_16")
    buffer.seek(0)

    return AudioSegment.from_wav(buffer)


def generate_normal_and_fast_audio(
    audio_segments: List[AudioSegment],
) -> Tuple[AudioSegment, AudioSegment]:
    """
    Generate normal speed and (10 copies) of a fast version of the dialogue. Designed to be
    called after generate_audio_from_dialogue as that func returns a list of audio segments

    :param audio_segments: List of AudioSegment objects representing each utterance
    :return: A tuple containing (normal_speed_audio, fast_speed_audio)
    """
    normal_speed = join_audio_segments(audio_segments, gap_ms=500)
    fast_segments = []
    for segment in audio_segments:
        fast_segment = speed_up_audio(segment)
        fast_segments.append(fast_segment)

    fast_audio = join_audio_segments(fast_segments * 10, gap_ms=200)
    return normal_speed, fast_audio


def generate_audio_from_dialogue(
    dialogue: List[Dict[str, str]], 
    in_target_language: bool = True
) -> List[AudioSegment]:
    """
    Generate audio from dialogue, handling HTML entities and special characters.
    
    Args:
        dialogue: List of dialogue utterances
        in_target_language: Whether to use target language voices
        
    Returns:
        List of AudioSegment objects for each utterance
    """
    # Clean the dialogue text while preserving the format
    cleaned_dialogue = clean_translated_content(dialogue)
    
    if in_target_language:
        voice_models = config.target_language_voice_models
    else:
        voice_models = config.english_voice_models

    texts = [utterance["text"] for utterance in cleaned_dialogue]
    language_codes = [voice_models["language_code"]] * len(cleaned_dialogue)
    voice_names = [
        (
            voice_models["male_voice"]
            if utterance["speaker"] == "Sam"
            else voice_models["female_voice"]
        )
        for utterance in cleaned_dialogue
    ]
    speaking_rates = [1.0] * len(cleaned_dialogue)

    return text_to_speech_multiprocessing(
        texts, language_codes, voice_names, speaking_rates
    )


def create_m4a_with_timed_lyrics(
    audio_segments,
    phrases,
    output_file,
    album_name,
    track_title,
    track_number,
    total_tracks=6,
    image_data=None,
):
    # Ensure the output directory exists
    output_dir = "../outputs/"
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the full output path
    full_output_path = os.path.join(output_dir, output_file)

    # Concatenate audio segments
    combined_audio = AudioSegment.empty()
    current_time = 0
    timed_lyrics = []

    for segment, phrase in zip(audio_segments, phrases):
        # Add to the combined audio
        combined_audio += segment

        # Calculate timestamp
        minutes, seconds = divmod(current_time / 1000, 60)
        timestamp = f"[{int(minutes):02d}:{seconds:05.2f}]"

        # Add to timed lyrics
        timed_lyrics.append(f"{timestamp}{phrase}")

        # Update current time
        current_time += len(segment)

    # Export combined audio to M4A
    temp_m4a = full_output_path + "_temp"
    combined_audio.export(temp_m4a, format="ipod")

    # Add metadata to the M4A file
    audio = MP4(temp_m4a)

    # Join all lyrics into a single string
    lyrics_text = "\n".join(timed_lyrics)

    # Add metadata
    audio["\xa9nam"] = track_title  # Track Title
    audio["\xa9alb"] = album_name  # Album Name
    audio["trkn"] = [(track_number, total_tracks)]  # Track Number
    audio["\xa9day"] = str(datetime.now().year)  # Year
    audio["aART"] = "Audio Language Trainer"  # Album Artist
    audio["\xa9lyr"] = lyrics_text  # Lyrics
    audio["\xa9gen"] = "Education"  # Genre set to Education
    audio["pcst"] = True  # Podcast flag set to True

    if image_data:
        audio["covr"] = [MP4Cover(image_data, imageformat=MP4Cover.FORMAT_JPEG)]

    audio.save()

    # Rename the temp file to the desired output name
    os.replace(temp_m4a, full_output_path)
