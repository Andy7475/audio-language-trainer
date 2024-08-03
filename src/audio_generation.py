import io
import os
import sys
import uuid
from typing import Dict, List, Optional, Tuple
import asyncio
import IPython.display as ipd
import librosa
import numpy as np
import soundfile as sf
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
from pydub import AudioSegment
from tqdm import tqdm

from src.config_loader import config
from src.translation import translate_from_english


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


async def async_text_to_speech(
    text: str,
    language_code: str = None,
    voice_name: str = None,
    speaking_rate: float = 1.0,
) -> AudioSegment:
    client = texttospeech.TextToSpeechClient()

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

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        ),
    )

    audio_segment = AudioSegment.from_mp3(io.BytesIO(response.audio_content))
    return audio_segment


async def async_generate_translated_phrase_audio(
    translated_phrase: tuple[str, str],
    english_voice_models: dict = None,
    target_voice_models: dict = None,
) -> AudioSegment:
    if english_voice_models is None:
        english_voice_models = config.english_voice_models
    if target_voice_models is None:
        target_voice_models = config.target_language_voice_models

    english_audio_task = asyncio.create_task(
        async_text_to_speech(
            text=translated_phrase[0],
            language_code=english_voice_models["language_code"],
            voice_name=english_voice_models["male_voice"],
            speaking_rate=0.9,
        )
    )

    target_audio_slow_task = asyncio.create_task(
        async_text_to_speech(
            text=translated_phrase[1],
            language_code=target_voice_models["language_code"],
            voice_name=target_voice_models["female_voice"],
            speaking_rate=config.SPEAKING_RATE_SLOW,
        )
    )

    target_audio_fast_task = asyncio.create_task(
        async_text_to_speech(
            text=translated_phrase[1],
            language_code=target_voice_models["language_code"],
            voice_name=target_voice_models["female_voice"],
            speaking_rate=1.0,
        )
    )

    english_audio, target_audio_slow, target_audio_fast = await asyncio.gather(
        english_audio_task, target_audio_slow_task, target_audio_fast_task
    )

    THINKING_GAP = AudioSegment.silent(duration=config.THINKING_GAP_MS)

    phrase_audio = join_audio_segments(
        [
            english_audio,
            THINKING_GAP,
            target_audio_fast,
            target_audio_slow,
        ],
        gap_ms=500,
    )
    return phrase_audio


async def async_process_phrases(phrases, max_concurrency=10):
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


def generate_translated_phrase_audio(
    translated_phrase: tuple[str, str],
    english_voice_models: dict = None,
    target_voice_models: dict = None,
) -> AudioSegment:
    """Creates and english, thinking pause, slow target, fast target audio segment"""

    # Use config values if parameters are not provided
    if english_voice_models is None:
        english_voice_models = config.english_voice_models
    if target_voice_models is None:
        target_voice_models = config.target_language_voice_models

    english_audio = text_to_speech(
        text=translated_phrase[0],
        language_code=english_voice_models["language_code"],
        voice_name=english_voice_models["male_voice"],
        speaking_rate=0.9,
    )

    target_audio_slow = text_to_speech(
        text=translated_phrase[1],
        language_code=target_voice_models["language_code"],
        voice_name=target_voice_models["female_voice"],
        speaking_rate=config.SPEAKING_RATE_SLOW,
    )

    target_audio_fast = text_to_speech(
        text=translated_phrase[1],
        language_code=target_voice_models["language_code"],
        voice_name=target_voice_models["female_voice"],
        speaking_rate=1.0,
    )

    THINKING_GAP = AudioSegment.silent(duration=config.THINKING_GAP_MS)

    phrase_audio = join_audio_segments(
        [
            english_audio,
            THINKING_GAP,
            target_audio_fast,
            target_audio_slow,
            # target_audio_fast,
        ],
        gap_ms=500,  # GAP AT END
    )
    return phrase_audio


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
    dialogue: List[Dict[str, str]], in_target_language=True
) -> List[AudioSegment]:
    """
    Generate audio from a dialogue in the target language, using different voices for each speaker.
    It will do the audio in the target language so assumes the dialogue is translated.

    :param dialogue: List of dictionaries containing 'speaker' and 'text' keys, already translated
    :return: List of AudioSegments of the entire dialogue in the target language
    """
    audio_segments = []

    # Get voice models from config
    if in_target_language:
        voice_models = config.target_language_voice_models
    else:
        voice_models = config.english_voice_models

    for utterance in tqdm(dialogue):
        speaker = utterance["speaker"]
        text = utterance["text"]

        # Choose voice based on speaker
        if speaker == "Sam":
            target_voice = voice_models["male_voice"]
        else:  # 'Alex' or any other speaker
            target_voice = voice_models["female_voice"]

        # Generate audio for translated text (normal and slow speed)
        target_audio_normal = text_to_speech(
            text=text,
            language_code=voice_models["language_code"],
            voice_name=target_voice,
        )

        audio_segments.append(target_audio_normal)

    return audio_segments
