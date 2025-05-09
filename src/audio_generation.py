import base64
import html
import io
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import azure.cognitiveservices.speech as speechsdk
import librosa
import numpy as np
import requests
import soundfile as sf
from elevenlabs import ElevenLabs
from google.cloud import texttospeech
from mutagen.mp4 import MP4, MP4Cover
from PIL import Image
from pydub import AudioSegment
from tqdm import tqdm

from src.config_loader import VoiceInfo, VoiceProvider, config
from src.convert import clean_filename
from src.gcs_storage import (
    check_blob_exists,
    get_fast_audio_path,
    get_story_translated_dialogue_path,
    get_utterance_audio_path,
    read_from_gcs,
    upload_to_gcs,
)
from src.translation import tokenize_text


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
                voice_setting="phrases",
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
            voice_setting="phrases",
        )

        # Generate normal target language audio
        target_normal = text_to_speech(
            text=cleaned_target,
            config_language="target",
            gender="FEMALE",
            speaking_rate=1.0,
            voice_setting="phrases",
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
    content: Union[str, Tuple[str, str], List[Dict[str, str]]],
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


def setup_ffmpeg():
    # Try default Windows path first
    default_ffmpeg_path = r"C:\Program Files\ffmpeg-7.0-essentials_build\bin"

    # Search system PATH for ffmpeg
    found_paths = []
    for path in os.environ["PATH"].split(os.pathsep):
        if "ffmpeg" in path.lower():
            found_paths.append(path)

    if os.path.exists(default_ffmpeg_path):
        # Add default FFmpeg to PATH if found
        os.environ["PATH"] += os.pathsep + default_ffmpeg_path
        print(f"Default FFmpeg path added to system PATH: {default_ffmpeg_path}")
    elif found_paths:
        # Print any ffmpeg paths found in system PATH
        print("Found existing FFmpeg paths:")
        for path in found_paths:
            print(f"  {path}")
    else:
        print("No FFmpeg paths found in default location or system PATH")
        print("Please check FFmpeg is installed and added to system PATH")


setup_ffmpeg()


def text_to_speech(
    text: str,
    config_language: Literal["source", "target"] = "source",
    gender: Literal["MALE", "FEMALE"] = "MALE",
    speaking_rate: float = 1.0,
    is_ssml: bool = False,
    voice_setting: Literal["phrases", "stories"] = "phrases",
) -> AudioSegment:
    """
    Wrapper that handles diveriting to Azure, Google, or ElevenLabs depending on the voice configuration.
    Converts text to speech using the configured provider.

    Args:
        text: Text to convert to speech
        config_language: Which language configuration to use ("source" or "target")
        gender: Target voice gender ("MALE" or "FEMALE")
        speaking_rate: Speed of speech (1.0 is normal speed)
        is_ssml: Whether the input text is SSML
        voice_setting: Which voice setting to use ("phrases" or "stories")

    Returns:
        AudioSegment containing the generated speech
    """
    # Get voice models for the specified setting
    voice_models = config.get_voice_models(enum_type=voice_setting)

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
    elif voice_model.provider == VoiceProvider.ELEVENLABS:
        return text_to_speech_elevenlabs(text, voice_model, speaking_rate, is_ssml)
    else:
        raise ValueError(f"Unsupported voice provider: {voice_model.provider}")


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
        voice_model: VoiceInfo object containing voice details
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


def text_to_speech_elevenlabs(
    text: str,
    voice_model: VoiceInfo,
    speaking_rate: float = 1.0,
    is_ssml: bool = False,
) -> AudioSegment:
    """
    Convert text to speech using ElevenLabs TTS API directly with requests.

    Args:
        text: Text to convert to speech (can include <break time="1.0s" /> tags)
        voice_model: VoiceInfo object containing voice details
        speaking_rate: Speed of speech (1.0 is normal speed)
        is_ssml: Whether the input text is standard SSML (will be converted for ElevenLabs)

    Returns:
        AudioSegment containing the generated speech

    Raises:
        ValueError: If SSML is requested, as ElevenLabs doesn't support SSML well
    """
    if is_ssml:
        raise ValueError(
            "SSML is not well supported by ElevenLabs. Please use a Google or Azure voice instead "
            "by updating the voice configuration in preferred_voices.json."
        )

    # Get ElevenLabs API key from environment variable
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not found in environment variables")

    try:
        # Set up API endpoint
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_model.voice_id}"

        # Set up headers
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        }

        # Set up request body
        body = {
            "text": text,
            "model_id": "eleven_multilingual_v2",  # Use multilingual model for language support
        }

        # Apply speaking rate if different from 1.0
        if speaking_rate != 1.0:
            body["voice_settings"] = {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "speed": speaking_rate,
            }

        # Make the API request
        response = requests.post(url, json=body, headers=headers)

        # Check if request was successful
        if response.status_code != 200:
            raise Exception(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

        # Convert response to audio segment
        audio_bytes = io.BytesIO(response.content)
        return AudioSegment.from_mp3(audio_bytes)

    except Exception as e:
        raise Exception(f"ElevenLabs speech synthesis error: {str(e)}")


def slow_text_to_speech(
    text: str,
    config_language: Literal["source", "target"] = "source",
    gender: Literal["MALE", "FEMALE"] = "MALE",
    speaking_rate: float = None,
    word_break_ms: int = None,
    voice_setting: Literal["phrases", "stories"] = "phrases",
) -> AudioSegment:
    """
    Generate slowed down text-to-speech audio with breaks between words.

    Uses SSML for Google and Azure voices, and explicit break tags for ElevenLabs voices.

    Args:
        text: Text to convert to speech
        config_language: Which language configuration to use ("source" or "target")
        gender: Target voice gender ("MALE" or "FEMALE")
        speaking_rate: Speaking rate (defaults to config.SPEAKING_RATE_SLOW)
        word_break_ms: Break time between words in ms (defaults to config.WORD_BREAK_MS)
        voice_setting: Which voice setting to use ("phrases" or "stories")

    Returns:
        AudioSegment containing the generated speech with word breaks

    Raises:
        ValueError: If the break duration exceeds ElevenLabs' limit of 3 seconds
    """
    if config_language == "source":
        language_code = config.SOURCE_LANGUAGE_CODE
    else:
        language_code = config.TARGET_LANGUAGE_CODE

    if speaking_rate is None:
        speaking_rate = config.SPEAKING_RATE_SLOW
    if word_break_ms is None:
        word_break_ms = config.WORD_BREAK_MS

    # Clean the text and tokenize it
    cleaned_text = clean_tts_text(text)
    tokens = tokenize_text(cleaned_text, language_code)

    # Get the voice model to determine the provider
    voice_models = config.get_voice_models(enum_type=voice_setting)

    if config_language == "source":
        voice_model = voice_models[0]
    elif (config_language == "target") & (gender == "FEMALE"):
        voice_model = voice_models[1]
    else:
        voice_model = voice_models[2]

    # Handle different providers
    if voice_model.provider == VoiceProvider.ELEVENLABS:
        # For ElevenLabs, convert milliseconds to seconds and use their break format
        break_sec = word_break_ms / 1000.0

        # Check if break duration exceeds ElevenLabs' limit (3 seconds)
        if break_sec >= 3.0:
            raise ValueError(
                f"Break duration of {break_sec:.2f}s exceeds ElevenLabs' limit of 3 seconds. "
                f"Please use a word_break_ms value less than 3000."
            )

        formatted_break = f' <break time="{break_sec:.2f}s" /> '

        # Join the tokens with ElevenLabs break format
        text_with_breaks = formatted_break.join(tokens)

        # Generate speech with the breaks embedded in the text
        return text_to_speech(
            text=text_with_breaks,
            config_language=config_language,
            gender=gender,
            speaking_rate=speaking_rate,
            is_ssml=False,  # ElevenLabs handles the break tags natively
            voice_setting=voice_setting,
        )
    else:
        # For Google and Azure, use standard SSML
        word_break_time = str(word_break_ms) + "ms"

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
            voice_setting=voice_setting,
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

    fast_audio = join_audio_segments(fast_segments, gap_ms=100)
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
            voice_setting="stories",
        )

        audio_segments.append(audio)

    return audio_segments


def generate_and_upload_fast_audio(
    story_name: str,
    bucket_name: str = config.GCS_PRIVATE_BUCKET,
    collection: str = "LM1000",
    overwrite: bool = False,
) -> Dict[str, str]:
    """
    Generate fast audio versions for each story part and upload them to GCS.

    Args:
        story_name: Name of the story
        bucket_name: GCS bucket name
        collection: Collection name (e.g., 'LM1000')
        overwrite: Whether to overwrite existing files

    Returns:
        Dictionary mapping story parts to GCS URIs of uploaded fast audio files
    """
    language_name = config.TARGET_LANGUAGE_NAME.lower()

    # Get story dialogue
    dialogue_path = get_story_translated_dialogue_path(story_name, collection)
    try:
        story_dialogue = read_from_gcs(bucket_name, dialogue_path, "json")
    except FileNotFoundError:
        print(f"Dialogue not found for {story_name} in {language_name}")
        return {}

    # Store the GCS URIs of uploaded fast audio files
    fast_audio_uris = {}

    # Process each section of the story
    for story_part, dialogue in tqdm(
        story_dialogue.items(), desc=f"Processing {story_name} in {language_name}"
    ):
        # Check if fast audio already exists and we're not overwriting
        fast_audio_path = get_fast_audio_path(story_name, story_part, collection)

        if not overwrite and check_blob_exists(bucket_name, fast_audio_path):
            print(f"Fast audio for {story_part} already exists, skipping.")
            fast_audio_uris[story_part] = f"gs://{bucket_name}/{fast_audio_path}"
            continue

        # Collect audio segments for this story part
        audio_segments = []

        for i, utterance in tqdm(
            enumerate(dialogue.get("translated_dialogue", [])),
            desc=f"Collecting utterance audio for {story_part}",
        ):
            try:
                audio_path = get_utterance_audio_path(
                    story_name,
                    story_part,
                    i,
                    utterance["speaker"],
                    language_name,
                    collection,
                )

                audio_segment = read_from_gcs(bucket_name, audio_path, "audio")
                audio_segments.append(audio_segment)
            except (FileNotFoundError, ValueError) as e:
                print(
                    f"Warning: Audio not found for utterance {i} in {story_part}: {str(e)}"
                )

        if not audio_segments:
            print(f"No audio segments found for {story_part}, skipping.")
            continue

        # Generate fast audio for this story part
        print(f"Generating fast audio for {story_part}...")
        normal_audio, fast_audio = generate_normal_and_fast_audio(audio_segments)

        # Upload fast audio to GCS
        filename = os.path.basename(fast_audio_path)
        base_prefix = os.path.dirname(fast_audio_path)

        uri = upload_to_gcs(
            obj=fast_audio,
            bucket_name=bucket_name,
            file_name=filename,
            base_prefix=base_prefix,
            content_type="audio/mpeg",
        )

        fast_audio_uris[story_part] = uri
        print(f"Uploaded fast audio for {story_part} to {uri}")

    return fast_audio_uris


def create_m4a_with_timed_lyrics(
    audio_segments: List[AudioSegment],
    phrases: List[str],
    output_file: str,
    album_name: str,
    track_title: str,
    track_number: int,
    total_tracks: int = 6,
    cover_image_base64: Optional[str] = None,  # Base64 string of the cover image
    output_dir: Optional[str] = None,
    gcs_bucket_name: Optional[str] = None,
    gcs_base_prefix: str = "",
) -> Union[str, tuple]:
    """
    Create an M4A file with timed lyrics and metadata, saving locally and/or uploading to GCS.

    Args:
        audio_segments: List of AudioSegment objects to combine
        phrases: List of text phrases matching the audio segments
        output_file: Name of the output file
        album_name: Name of the album
        track_title: Title of this track
        track_number: Number of this track in the album
        total_tracks: Total number of tracks in album
        cover_image: Optional cover artwork as PIL Image
        output_dir: Optional local directory to save the file
        gcs_bucket_name: Optional GCS bucket to upload the file
        gcs_base_prefix: Prefix/folder path in the GCS bucket. Defaults to ""

    Returns:
        If saving locally only: Local file path
        If uploading to GCS only: GCS URI
        If both: Tuple of (local_path, gcs_uri)

    Raises:
        ValueError: If neither output_dir nor gcs_bucket_name is provided
    """
    if not output_dir and not gcs_bucket_name:
        raise ValueError("Either output_dir or gcs_bucket_name must be provided")

    # Create a temporary file
    with io.BytesIO() as temp_buffer:
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

        # Export combined audio to M4A in memory
        combined_audio.export(temp_buffer, format="ipod")
        temp_buffer.seek(0)

        # Add metadata to the M4A file
        temp_file_path = (
            f"/tmp/{output_file}_temp"
            if os.path.exists("/tmp")
            else f"{output_file}_temp"
        )
        with open(temp_file_path, "wb") as f:
            f.write(temp_buffer.read())

        audio = MP4(temp_file_path)

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

        if cover_image_base64:
            # Decode base64 string to image data
            try:
                # Remove potential header from base64 string
                if "," in cover_image_base64:
                    cover_image_base64 = cover_image_base64.split(",", 1)[1]

                # Decode base64 to bytes
                image_data = base64.b64decode(cover_image_base64)

                # For M4A cover art, we can use PNG directly for better quality
                # The MP4Cover class accepts raw bytes and can handle PNG format
                audio["covr"] = [MP4Cover(image_data, imageformat=MP4Cover.FORMAT_PNG)]
            except Exception as e:
                print(f"Error processing cover image: {e}")

        audio.save()

        # Get the final file with metadata
        with open(temp_file_path, "rb") as f:
            final_audio_data = f.read()

        # Remove the temporary file
        os.remove(temp_file_path)

        results = []

        # Save locally if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            local_path = os.path.join(output_dir, output_file)
            with open(local_path, "wb") as f:
                f.write(final_audio_data)
            results.append(local_path)

        # Upload to GCS if bucket_name is provided
        if gcs_bucket_name:
            full_path = f"{gcs_base_prefix.rstrip('/')}/{output_file}".lstrip("/")
            gcs_uri = upload_to_gcs(
                final_audio_data, gcs_bucket_name, full_path, content_type="audio/mp4"
            )
            print(f"Uploaded to GCS: {gcs_uri}")
            results.append(gcs_uri)

        # Return appropriate result
        if len(results) == 1:
            return results[0]
        return tuple(results)


def upload_phrases_audio_to_gcs(
    phrase_dict: Dict[str, Dict[str, str]],
    bucket_name: Optional[str] = None,
    upload_english_audio: bool = False,
    overwrite: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate audio for translated phrases one by one and immediately upload to GCS.

    Args:
        phrase_dict: Dictionary with phrase_key as keys and values containing
                    'english' and target language phrases
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PUBLIC_BUCKET)
        upload_english_audio: Whether to upload English audio (default: False)
        overwrite: Whether to overwrite existing files in GCS (default: False)

    Returns:
        Dictionary with the same structure as input but with added 'audio_urls' containing GCS URLs
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    target_language = config.TARGET_LANGUAGE_NAME
    target_language_lower = target_language.lower()

    # Create a copy of the input dictionary for results
    result_dict = {k: v.copy() for k, v in phrase_dict.items()}

    # Process each phrase one at a time
    for phrase_key, phrase_data in tqdm(phrase_dict.items(), desc="Processing phrases"):
        english_text = phrase_data.get("english")
        target_text = phrase_data.get(target_language_lower)

        # Skip if either text is missing
        if not english_text:
            print(f"Skipping phrase {phrase_key} because 'english' is missing")
            continue

        if not target_text:
            print(
                f"Skipping phrase {phrase_key} because '{target_language}' translation is missing"
            )
            continue

        # Create base filename for the audio files
        clean_key = clean_filename(english_text)

        # Define paths for normal and slow audio
        normal_path = (
            f"multimedia/audio/phrases/{target_language_lower}/normal/{clean_key}.mp3"
        )
        slow_path = (
            f"multimedia/audio/phrases/{target_language_lower}/slow/{clean_key}.mp3"
        )

        # Check if files already exist (if not overwriting)
        if not overwrite:
            normal_exists = check_blob_exists(bucket_name, normal_path)
            slow_exists = check_blob_exists(bucket_name, slow_path)

            if normal_exists and slow_exists:
                print(
                    f"Skipping phrase {phrase_key} - audio files already exist (use overwrite=True to replace)"
                )

                # Add the existing GCS URLs to the result
                if "audio_urls" not in result_dict[phrase_key]:
                    result_dict[phrase_key]["audio_urls"] = {}

                result_dict[phrase_key]["audio_urls"][
                    "normal"
                ] = f"gs://{bucket_name}/{normal_path}"
                result_dict[phrase_key]["audio_urls"][
                    "slow"
                ] = f"gs://{bucket_name}/{slow_path}"
                continue

        try:
            # Make sure we don't hit API rate limits
            # ok_to_query_api()

            # Generate audio for this single phrase
            translated_phrase = (english_text, target_text)
            audio_segments = generate_translated_phrase_audio([translated_phrase])[0]

            # Get the audio segments
            english_audio = audio_segments[0]
            slow_audio = audio_segments[1]
            normal_audio = audio_segments[2]

            # Upload normal and slow target audio to GCS
            normal_url = upload_to_gcs(
                normal_audio,
                bucket_name,
                f"{clean_key}.mp3",
                base_prefix=f"multimedia/audio/phrases/{target_language_lower}/normal",
            )

            slow_url = upload_to_gcs(
                slow_audio,
                bucket_name,
                f"{clean_key}.mp3",
                base_prefix=f"multimedia/audio/phrases/{target_language_lower}/slow",
            )

            # Store results
            if "audio_urls" not in result_dict[phrase_key]:
                result_dict[phrase_key]["audio_urls"] = {}

            result_dict[phrase_key]["audio_urls"]["normal"] = normal_url
            result_dict[phrase_key]["audio_urls"]["slow"] = slow_url

            # Optionally upload English audio
            if upload_english_audio:
                english_path = f"multimedia/audio/phrases/english/{clean_key}.mp3"

                # Check if English audio exists if not overwriting
                if not overwrite and check_blob_exists(bucket_name, english_path):
                    print(
                        f"Skipping English audio for phrase {phrase_key} - file already exists"
                    )
                    result_dict[phrase_key]["audio_urls"][
                        "english"
                    ] = f"gs://{bucket_name}/{english_path}"
                else:
                    english_url = upload_to_gcs(
                        english_audio,
                        bucket_name,
                        f"{clean_key}.mp3",
                        base_prefix="multimedia/audio/phrases/english",
                    )
                    result_dict[phrase_key]["audio_urls"]["english"] = english_url

            print(f"Successfully processed and uploaded audio for phrase: {phrase_key}")

        except Exception as e:
            print(f"Error processing phrase {phrase_key}: {str(e)}")
            # Continue with the next phrase
            continue

    return result_dict


def generate_and_upload_audio_for_utterance(
    utterance: Dict,
    story_part: str,
    utterance_index: int,
    story_name: str,
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
) -> str:
    """
    Generate audio for a single utterance and upload it to GCS.

    Args:
        utterance: Dictionary with 'speaker' and 'text' keys
        story_part: Part of the story (e.g., 'introduction')
        utterance_index: Index of the utterance in the dialogue list
        language_name: Target language name
        story_name: Name of the story
        collection: Collection name (e.g., 'LM1000', 'LM2000')
        bucket_name: Optional bucket name
        config_language: Whether to use source or target language config

    Returns:
        GCS URI of the uploaded audio file
    """
    if bucket_name is None:

        bucket_name = config.GCS_PRIVATE_BUCKET

    language_name = config.TARGET_LANGUAGE_NAME.lower()
    # Ensure story_name is properly formatted
    story_name = (
        f"story_{story_name}" if not story_name.startswith("story_") else story_name
    )

    # Determine gender based on speaker
    gender = "MALE" if utterance["speaker"] == "Sam" else "FEMALE"

    # Generate audio for this utterance
    audio = text_to_speech(
        utterance["text"],
        config_language="target",
        gender=gender,
        speaking_rate=1.0,
        voice_setting="stories",
    )

    audio_path = get_utterance_audio_path(
        story_name,
        story_part,
        utterance_index,
        utterance["speaker"],
        language_name,
        collection,
    )
    # Upload the audio
    gcs_uri = upload_to_gcs(obj=audio, bucket_name=bucket_name, file_name=audio_path)

    return gcs_uri


def generate_dialogue_audio_and_upload(
    translated_dialogue_dict: Dict,
    story_name: str,
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    """
    Generate audio for a translated dialogue dictionary and upload to GCS.

    Args:
        translated_dialogue_dict: Dictionary containing the translated dialogue
        story_name: Name of the story
        language_name: Target language name
        collection: Collection name (e.g., 'LM1000', 'LM2000')
        bucket_name: Optional bucket name
        config_language: Whether to use source or target language config

    Returns:
        Dictionary mapping story parts to lists of GCS URIs for the audio files
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Ensure story_name is properly formatted
    story_name = (
        f"story_{story_name}" if not story_name.startswith("story_") else story_name
    )
    language_name = config.TARGET_LANGUAGE_NAME.lower()

    # Generate and upload audio for each utterance in each story part
    audio_uris = {}

    for story_part, part_data in tqdm(
        translated_dialogue_dict.items(), desc="Processing story parts"
    ):
        if "translated_dialogue" in part_data:
            # Create a directory structure if it doesn't exist
            audio_dir_path = (
                f"{collection}/stories/{story_name}/audio/{language_name}/{story_part}"
            )
            audio_uris[story_part] = []

            # Process each utterance in the dialogue
            for utterance_index, utterance in enumerate(
                tqdm(
                    part_data["translated_dialogue"],
                    desc=f"Generating translated audio for {story_part}",
                    leave=False,
                )
            ):
                # First check if this audio file already exists
                audio_path = get_utterance_audio_path(
                    story_name,
                    story_part,
                    utterance_index,
                    utterance["speaker"],
                    language_name,
                    collection,
                )

                if check_blob_exists(bucket_name, audio_path) and not overwrite:
                    print(f"Audio file already exists: gs://{bucket_name}/{audio_path}")
                    audio_uri = f"gs://{bucket_name}/{audio_path}"
                else:
                    # Generate and upload audio
                    audio_uri = generate_and_upload_audio_for_utterance(
                        utterance,
                        story_part,
                        utterance_index,
                        story_name,
                        collection,
                        bucket_name,
                    )
                    print(f"Generated and uploaded: {audio_uri}")

                audio_uris[story_part].append(audio_uri)

    return audio_uris
