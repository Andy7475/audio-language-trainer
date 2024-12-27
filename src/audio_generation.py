import asyncio
from datetime import datetime
import html
import io
import multiprocessing
import os
import re
import sys
import time
import uuid
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union
from PIL import Image
import IPython.display as ipd
import librosa
import numpy as np
import soundfile as sf
from google.cloud import texttospeech, texttospeech_v1
from mutagen.mp4 import MP4, MP4Cover
from pydub import AudioSegment
import azure.cognitiveservices.speech as speechsdk
from src.config_loader import VoiceProvider, config, VoiceInfo
from src.translation import tokenize_text
from src.utils import clean_filename
from tqdm import tqdm


def generate_translated_phrase_audio(
    translated_phrases: List[Tuple[str, str]], source_language_audio: bool = False
) -> List[List[AudioSegment]]:
    """
    Generate audio for a list of translated phrases.

    Args:
        translated_phrases: List of tuples (original_text, translated_text)
        english_voice_models: Configuration for English TTS voices
        target_voice_models: Configuration for target language TTS voices

    Returns:
        List of AudioSegment lists, where each inner list contains:
        [english_audio, target_slow_audio, target_normal_audio]
    """

    all_audio_segments = []

    for eng_text, target_text in tqdm(translated_phrases, desc="Generating audio"):
        # Clean the texts
        cleaned_eng = clean_tts_text(eng_text)
        cleaned_target = clean_tts_text(target_text)

        # Generate English audio
        if source_language_audio:
            english_audio = text_to_speech(
                text=cleaned_eng,
                config_language="source",
                gender="MALE",
            )
        else:
            english_audio = AudioSegment.silent(100)

        # Generate slow target language audio with word breaks
        target_slow = slow_text_to_speech(
            text=cleaned_target,
            config_language="target",
            gender="FEMALE",
            speaking_rate=config.SPEAKING_RATE_SLOW,
            word_break_ms=config.WORD_BREAK_MS,
        )

        # Generate normal target language audio
        target_normal = text_to_speech(
            text=cleaned_target,
            config_language="target",
            gender="FEMALE",
            speaking_rate=1.0,
        )

        all_audio_segments.append([english_audio, target_slow, target_normal])

    return all_audio_segments


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
        return [{**d, "text": clean_tts_text(d["text"])} for d in content]
    else:
        raise ValueError(f"Unsupported content format: {type(content)}")


def generate_phrase_english_audio_files(phrases: List[str], output_dir: str) -> None:
    """
    Generate slow and normal English-only speed MP3 files for each phrase and save them to output_dir.

    Args:
        phrases: List of English phrases to convert to audio
        output_dir: Directory where the MP3 files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for phrase in tqdm(phrases):
        # Generate a clean filename for this phrase
        base_filename = clean_filename(phrase)

        if os.path.exists(os.path.join(output_dir, f"{base_filename}.mp3")):
            # file already exists
            print(f"{base_filename} exists, skipping")
            continue

        # Generate the audio for normal speed
        normal_audio = text_to_speech(text=phrase, speaking_rate=1.0)

        # Generate the audio for slow speed
        slow_audio = slow_text_to_speech(
            text=phrase,
        )

        # Save the normal speed version
        normal_filepath = os.path.join(output_dir, f"{base_filename}.mp3")
        normal_audio.export(normal_filepath, format="mp3")

        # Save the slow version
        slow_filepath = os.path.join(output_dir, f"{base_filename}_slow.mp3")
        slow_audio.export(slow_filepath, format="mp3")

        print(f"Generated audio files for phrase: {phrase}")


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


def text_to_speech_google(
    text: str,
    voice_model: VoiceInfo,
    speaking_rate: float = 1.0,
    is_ssml: bool = False,
) -> AudioSegment:
    """
    Convert text to speech using Google Cloud TTS.

    Args:
        text: Text or SSML to convert to speech
        language_code: Language code (e.g., 'en-US')
        voice_name: Name of the voice to use
        speaking_rate: Speed of speech (1.0 is normal speed)
        is_ssml: Whether the input text is SSML

    Returns:
        AudioSegment containing the generated speech
    """
    client = texttospeech.TextToSpeechClient()

    # Create appropriate input type based on whether text is SSML
    if is_ssml:
        synthesis_input = texttospeech.SynthesisInput(ssml=text)
    else:
        synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=voice_model.language_code, name=voice_model.voice_id
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=speaking_rate
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return AudioSegment.from_mp3(io.BytesIO(response.audio_content))


def text_to_speech_azure(
    text: str,
    voice_model: VoiceInfo,
    speaking_rate: float = 1.0,
    is_ssml: bool = False,
) -> AudioSegment:
    """
    Convert text to speech using Azure Speech Service.

    Args:
        text: Text or SSML to convert to speech
        voice_model: VoiceInfo object containing voice details
        speaking_rate: Speed of speech (1.0 is normal speed)
        is_ssml: Whether the input text is SSML

    Returns:
        AudioSegment containing the generated speech
    """
    speech_key = os.getenv("AZURE_API_KEY")
    service_region = os.getenv("AZURE_REGION", "eastus")

    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=service_region
    )
    speech_config.speech_synthesis_voice_name = voice_model.voice_id
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )

    # Create a temporary file for output
    audio_buffer = io.BytesIO()

    def write_to_buffer(evt):
        audio_buffer.write(evt.result.audio_data)

    # Configure speech synthesizer
    pull_stream = speechsdk.audio.PullAudioOutputStream()
    audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # Subscribe to events for audio data
    speech_synthesizer.synthesizing.connect(write_to_buffer)

    try:
        # Handle SSML and non-SSML input appropriately

        if is_ssml:
            # Extract the content between <speak> tags
            content = text[7:-8].strip()  # Remove <speak> and </speak>

            # Wrap with Azure's required SSML format
            azure_ssml = (
                f'<speak xmlns="http://www.w3.org/2001/10/synthesis" '
                f'xmlns:mstts="http://www.w3.org/2001/mstts" '
                f'xmlns:emo="http://www.w3.org/2009/10/emotionml" '
                f'version="1.0" xml:lang="{voice_model.language_code}">'
                f'<voice name="{voice_model.voice_id}">'
                f"{content}"
                f"</voice></speak>"
            )
            result = speech_synthesizer.speak_ssml_async(azure_ssml).get()
        else:
            if speaking_rate != 1.0:
                # Wrap with rate modification in SSML
                text = (
                    f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
                    f'xmlns:mstts="http://www.w3.org/2001/mstts">'
                    f'<voice name="{voice_model.voice_id}">'
                    f'<lang xml:lang="{voice_model.language_code}">'
                    f'<prosody rate="{int((speaking_rate - 1) * 100):+d}%">'
                    f"{text}"
                    f"</prosody>"
                    f"</lang>"
                    f"</voice>"
                    f"</speak>"
                )
                result = speech_synthesizer.speak_ssml_async(text).get()
            else:
                result = speech_synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_buffer.seek(0)
            return AudioSegment.from_mp3(audio_buffer)
        else:
            error_details = f"Reason: {result.reason}"
            if hasattr(result, "cancellation_details"):
                error_details = (
                    f"Reason: {result.cancellation_details.reason}, "
                    f"Error: {result.cancellation_details.error_details}"
                )
            raise Exception(f"Speech synthesis failed: {error_details}")

    except Exception as e:
        raise Exception(f"Azure speech synthesis error: {str(e)}")


def text_to_speech(
    text: str,
    config_language: Literal["source", "target"] = "source",
    gender: Literal["MALE", "FEMALE"] = "MALE",
    speaking_rate: float = 1.0,
    is_ssml: bool = False,
) -> AudioSegment:
    """
    Wrapper that handles diveriting to Azure or Google depending on the settings in the config file
    for language, which then cause the voice models to be either Azure or Google ones.
    Converts text to speech using the configured provider (Google or Azure).

    Args:
        text: Text to convert to speech
        config_language: so we know which model to choose
        gender: target voice models are both male or female
        speaking_rate: Speed of speech (1.0 is normal speed)
        is_ssml: Whether the input text is SSML

    Returns:
        AudioSegment containing the generated speech
    """
    # Use config values if parameters are not provided
    voice_models = config.get_voice_models()

    if config_language == "source":
        voice_model = voice_models[0]
    elif (config_language == "target") & (gender == "FEMALE"):
        voice_model = voice_models[1]
    else:
        voice_model = voice_models[2]

    # Route to appropriate provider
    if voice_model.provider == VoiceProvider.GOOGLE:
        return text_to_speech_google(text, voice_model, speaking_rate, is_ssml)
    elif voice_model.provider == VoiceProvider.AZURE:
        return text_to_speech_azure(text, voice_model, speaking_rate, is_ssml)
    else:
        raise ValueError(f"Unsupported voice provider: {voice_model.provider}")


def slow_text_to_speech(
    text: str,
    config_language: Literal["source", "target"] = "source",
    gender: Literal["MALE", "FEMALE"] = "MALE",
    speaking_rate: float = None,
    word_break_ms: int = None,
) -> AudioSegment:
    """
    Generate slowed down text-to-speech audio with breaks between words using SSML.

    Args:
        text: Text to convert to speech
        language_code: Language code for TTS (defaults to config's target language code)
        voice_name: Name of voice to use (defaults to config's target male voice)
        speaking_rate: Speaking rate (defaults to config.SPEAKING_RATE_SLOW)
        word_break_ms: Break time between words in ms (defaults to config.WORD_BREAK_MS)

    Returns:
        AudioSegment containing the generated speech with word breaks
    """

    if config_language == "source":
        language_code = config.SOURCE_LANGUAGE_CODE
    else:
        language_code = config.TARGET_LANGUAGE_CODE

    if speaking_rate is None:
        speaking_rate = config.SPEAKING_RATE_SLOW
    if word_break_ms is None:
        word_break_ms = config.WORD_BREAK_MS

    word_break_time = str(word_break_ms) + "ms"

    # Clean the text and tokenize it
    cleaned_text = clean_tts_text(text)
    tokens = tokenize_text(cleaned_text, language_code)

    # Create SSML with breaks between words
    ssml_parts = ["<speak>"]
    for i, token in enumerate(tokens):
        ssml_parts.append(token)
        if i < len(tokens) - 1:
            ssml_parts.append(f'<break time="{word_break_time}"/>')
    ssml_parts.append("</speak>")
    ssml_text = " ".join(ssml_parts)

    # Use the main text_to_speech function with SSML
    return text_to_speech(
        text=ssml_text,
        config_language=config_language,
        gender=gender,
        speaking_rate=speaking_rate,
        is_ssml=True,
    )


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
    Generate normal speed and a fast version of the dialogue. Designed to be
    called after generate_audio_from_dialogue as that func returns a list of audio segments

    :param audio_segments: List of AudioSegment objects representing each utterance
    :return: A tuple containing (normal_speed_audio, fast_speed_audio)
    """
    normal_speed = join_audio_segments(audio_segments, gap_ms=500)
    fast_segments = []
    for segment in audio_segments:
        fast_segment = speed_up_audio(segment)
        fast_segments.append(fast_segment)

    fast_audio = join_audio_segments(fast_segments, gap_ms=200)
    return normal_speed, fast_audio


def generate_audio_from_dialogue(
    dialogue: List[Dict[str, str]],
    config_language: Literal["source", "target"] = "target",
) -> List[AudioSegment]:
    """
    Generate audio from dialogue using sequential processing.
    Typicall only generated in the target language

    Args:
        dialogue: List of dialogue utterances
        in_target_language: Whether to use target language voices

    Returns:
        List of AudioSegment objects for each utterance
    """
    # Clean the dialogue text while preserving the format
    cleaned_dialogue = clean_translated_content(dialogue)

    audio_segments = []

    for utterance in tqdm(cleaned_dialogue, desc="Generating dialogue audio"):
        # Select gender based on speaker
        if utterance["speaker"] == "Sam":
            gender = "MALE"
        else:
            gender = "FEMALE"
        # Generate audio for this utterance
        audio = text_to_speech(
            utterance["text"],
            config_language,
            gender,
            speaking_rate=1.0,
        )

        audio_segments.append(audio)

    return audio_segments


def create_m4a_with_timed_lyrics(
    audio_segments: List[AudioSegment],
    phrases: List[str],
    output_file: str,
    album_name: str,
    track_title: str,
    track_number: int,
    total_tracks: int = 6,
    cover_image: Optional[Image.Image] = None,
) -> None:
    """
    Create an M4A file with timed lyrics and metadata.

    Args:
        audio_segments: List of AudioSegment objects to combine
        phrases: List of text phrases matching the audio segments
        output_file: Name of the output file
        album_name: Name of the album
        track_title: Title of this track
        track_number: Number of this track in the album
        total_tracks: Total number of tracks in album
        cover_image: Optional cover artwork as PIL Image
    """
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

    if cover_image:
        jpeg_bytes = cover_image.convert("RGB").tobytes("jpeg", "RGB")
        audio["covr"] = [MP4Cover(jpeg_bytes, imageformat=MP4Cover.FORMAT_JPEG)]

    audio.save()

    # Rename the temp file to the desired output name
    os.replace(temp_m4a, full_output_path)
