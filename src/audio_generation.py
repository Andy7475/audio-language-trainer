import io
import os
import sys
import uuid
from typing import Dict, List, Optional

import IPython.display as ipd
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
from pydub import AudioSegment


def setup_ffmpeg():
    ffmpeg_path = r"C:\Program Files\ffmpeg-7.0-essentials_build\bin"

    if os.path.exists(ffmpeg_path):
        # Add FFmpeg to the PATH
        os.environ["PATH"] += os.pathsep + ffmpeg_path
        print(f"FFmpeg path added to system PATH: {ffmpeg_path}")
    else:
        print(f"FFmpeg path not found: {ffmpeg_path}")
        print("Please check the installation directory.")


# Call this function at the start of your script
setup_ffmpeg()


def get_optimal_voice_models(
    language_code: str,
    preferred_country_code: Optional[str] = None,
    cheap_voices: bool = False,
) -> Dict[str, str]:
    """Returns a dictionary of keys: language_code - (e.g. en-GB), male_voice and female_voice from Google TTS services. voice models to use based on a 2-char language code and preferred country code,
    e.g. en and GB would prefer british english voices over en and US. if cheap = True it picks poor quality but cheaper voices
    which is useful for testing. Otherwise it priorities voices in the rank Studio > Neural2 > Wavenet > Standard
    which you'll want for production.

    These voice codes are needed for TTS and translation services at Google API"""
    # Initialize clients
    tts_client = texttospeech.TextToSpeechClient()
    translate_client = translate.Client()

    # Check if the language is supported by Google Translate
    supported_languages = [
        lang["language"] for lang in translate_client.get_languages()
    ]
    if language_code not in supported_languages:
        raise ValueError(
            f"Language code '{language_code}' is not supported by Google Translate"
        )

    # Get all available voices
    voices = tts_client.list_voices().voices

    # Filter voices for the given language code
    language_voices = [
        v
        for v in voices
        if any(lc.startswith(language_code) for lc in v.language_codes)
    ]

    if not language_voices:
        raise ValueError(f"No voices found for language code '{language_code}'")

    # Determine the best language code
    available_codes = set(sum((list(v.language_codes) for v in language_voices), []))
    if preferred_country_code:
        preferred_full_code = f"{language_code}-{preferred_country_code.upper()}"
        if preferred_full_code in available_codes:
            best_language_code = preferred_full_code
        else:
            print(
                f"Preferred country code '{preferred_country_code}' not available for {language_code}"
            )
            best_language_code = sorted(available_codes)[0]
    else:
        best_language_code = sorted(available_codes)[0]

    # Print a message if there are multiple country codes available
    if len(available_codes) > 1:
        print(
            f"Multiple country codes available for {language_code}: {', '.join(sorted(available_codes))}"
        )

    def select_best_voice(
        voices: List[texttospeech.Voice], gender: texttospeech.SsmlVoiceGender
    ) -> texttospeech.Voice:
        # Filter voices by gender
        gender_voices = [v for v in voices if v.ssml_gender == gender]

        # Prefer voices matching the best_language_code
        matching_voices = [
            v for v in gender_voices if best_language_code in v.language_codes
        ]

        voices_to_choose = matching_voices if matching_voices else gender_voices

        # Define voice type preference order
        voice_types = (
            ["Standard", "Wavenet", "Neural", "Studio"]
            if cheap_voices
            else ["Studio", "Neural", "Wavenet", "Standard"]
        )

        # Prefer voice types in the specified order
        for voice_type in voice_types:
            typed_voices = [v for v in voices_to_choose if voice_type in v.name]
            if typed_voices:
                # Return the first voice of this type
                return typed_voices[0]

        # If no preferred voice types found, return the first available voice
        return voices_to_choose[0] if voices_to_choose else None

    best_male_voice = select_best_voice(
        language_voices, texttospeech.SsmlVoiceGender.MALE
    )
    best_female_voice = select_best_voice(
        language_voices, texttospeech.SsmlVoiceGender.FEMALE
    )

    return {
        "language_code": best_language_code,
        "male_voice": best_male_voice.name if best_male_voice else None,
        "female_voice": best_female_voice.name if best_female_voice else None,
    }


def text_to_speech(
    text: str,
    language_code: str = "en-US",
    voice_name: str = "en-US-Wavenet-D",
    speaking_rate: float = 1.0,
) -> AudioSegment:
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
