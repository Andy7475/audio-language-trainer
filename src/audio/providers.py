"""Text-to-speech provider implementations."""

import io
import os

import azure.cognitiveservices.speech as speechsdk
import requests
from google.cloud import texttospeech
from pydub import AudioSegment

from .constants import (
    VoiceProvider,
    SPEAKING_RATE_NORMAL,
    DEFAULT_WORD_BREAK_MS,
)
from .text_processing import clean_tts_text, tokenize_text
from .voices import VoiceInfo
from ..connections.gcloud_auth import get_texttospeech_client


def text_to_speech(
    text: str,
    voice_model: VoiceInfo,
    speaking_rate: float = SPEAKING_RATE_NORMAL,
    is_ssml: bool = False,
) -> AudioSegment:
    """
    Convert text to speech using the provider specified in voice_model.

    Args:
        text: Text to convert to speech
        voice_model: VoiceInfo object containing provider and voice details
        speaking_rate: Speed of speech (1.0 is normal speed)
        is_ssml: Whether the input text is SSML

    Returns:
        AudioSegment containing the generated speech

    Raises:
        ValueError: If provider is unsupported or voice is invalid
    """
    # Detect Google Chirp3 HD voice (no SSML, uses markup and pause tags)

    if voice_model.provider == VoiceProvider.GOOGLE:
        return _text_to_speech_google(text, voice_model, speaking_rate, is_ssml)
    elif voice_model.provider == VoiceProvider.AZURE:
        return _text_to_speech_azure(text, voice_model, speaking_rate, is_ssml)
    elif voice_model.provider == VoiceProvider.ELEVENLABS:
        return _text_to_speech_elevenlabs(text, voice_model, speaking_rate, is_ssml)
    else:
        raise ValueError(f"Unsupported voice provider: {voice_model.provider}")


def slow_text_to_speech(
    text: str,
    voice_model: VoiceInfo,
    speaking_rate: float,
    word_break_ms: int = DEFAULT_WORD_BREAK_MS,
) -> AudioSegment:
    """
    Generate slowed down text-to-speech audio with breaks between words.

    Uses different markup based on provider:
    - Google Chirp3 HD: [pause] tags in markup
    - Google (non-Chirp) and Azure: SSML with <break> tags
    - ElevenLabs: <break time="X.XXs" /> tags

    Args:
        text: Text to convert to speech
        voice_model: VoiceInfo object containing provider and voice details
        speaking_rate: Speaking rate (typically 0.5 to 1.0 for slow speech)
        word_break_ms: Break time between words in milliseconds

    Returns:
        AudioSegment containing the generated speech with word breaks

    Raises:
        ValueError: If the break duration exceeds provider limits
    """
    # Clean the text and tokenize it
    cleaned_text = clean_tts_text(text)
    tokens = tokenize_text(cleaned_text, voice_model.language_code)

    if voice_model.provider == VoiceProvider.ELEVENLABS:
        return _slow_text_to_speech_elevenlabs(
            tokens, voice_model, speaking_rate, word_break_ms
        )
    elif (
        voice_model.provider == VoiceProvider.GOOGLE and "Chirp" in voice_model.voice_id
    ):
        return _slow_text_to_speech_google_chirp(tokens, voice_model, speaking_rate)
    else:
        # For Google (non-Chirp) and Azure, use standard SSML
        return _slow_text_to_speech_ssml(
            tokens, voice_model, speaking_rate, word_break_ms
        )


def _text_to_speech_google(
    text: str,
    voice_model: VoiceInfo,
    speaking_rate: float = SPEAKING_RATE_NORMAL,
    is_ssml: bool = False,
) -> AudioSegment:
    """
    Convert text to speech using Google Cloud TTS.

    Args:
        text: Text or SSML to convert to speech
        voice_model: VoiceInfo object containing voice details
        speaking_rate: Speed of speech (1.0 is normal speed)
        is_ssml: Whether the input text is SSML
        use_markup: Whether to use markup field (for Chirp3 HD)

    Returns:
        AudioSegment containing the generated speech
    """
    client = get_texttospeech_client()

    if is_ssml:
        synthesis_input = texttospeech.SynthesisInput(ssml=text)
    else:
        synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=voice_model.language_code,
        name=voice_model.voice_id,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    return AudioSegment.from_mp3(io.BytesIO(response.audio_content))


def _text_to_speech_azure(
    text: str,
    voice_model: VoiceInfo,
    speaking_rate: float = SPEAKING_RATE_NORMAL,
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

    Raises:
        ValueError: If Azure API key is not configured
        Exception: If speech synthesis fails
    """
    speech_key = os.getenv("AZURE_API_KEY")
    if not speech_key:
        raise ValueError("AZURE_API_KEY environment variable not set")

    service_region = os.getenv("AZURE_REGION", "eastus")

    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        region=service_region,
    )
    speech_config.speech_synthesis_voice_name = voice_model.voice_id
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )

    audio_buffer = io.BytesIO()

    def write_to_buffer(evt):
        audio_buffer.write(evt.result.audio_data)

    # Configure speech synthesizer
    pull_stream = speechsdk.audio.PullAudioOutputStream()
    audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config,
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
            if speaking_rate != SPEAKING_RATE_NORMAL:
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


def _text_to_speech_elevenlabs(
    text: str,
    voice_model: VoiceInfo,
    speaking_rate: float = SPEAKING_RATE_NORMAL,
    is_ssml: bool = False,
) -> AudioSegment:
    """
    Convert text to speech using ElevenLabs TTS API.

    Args:
        text: Text to convert to speech (can include <break time="X.XXs" /> tags)
        voice_model: VoiceInfo object containing voice details
        speaking_rate: Speed of speech (1.0 is normal speed)
        is_ssml: Whether the input text is standard SSML

    Returns:
        AudioSegment containing the generated speech

    Raises:
        ValueError: If SSML is requested or API key is missing
        Exception: If API request fails
    """
    if is_ssml:
        raise ValueError(
            "SSML is not well supported by ElevenLabs. Please use a Google or Azure voice instead."
        )

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY environment variable not set")

    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_model.voice_id}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        }

        body = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
        }

        if speaking_rate != SPEAKING_RATE_NORMAL:
            body["voice_settings"] = {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "speed": speaking_rate,
            }

        response = requests.post(url, json=body, headers=headers)

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

        audio_bytes = io.BytesIO(response.content)
        return AudioSegment.from_mp3(audio_bytes)

    except Exception as e:
        raise Exception(f"ElevenLabs speech synthesis error: {str(e)}")


def _slow_text_to_speech_elevenlabs(
    tokens: list[str],
    voice_model: VoiceInfo,
    speaking_rate: float,
    word_break_ms: int,
) -> AudioSegment:
    """Generate slow speech for ElevenLabs using <break> tags."""
    break_sec = word_break_ms / 1000.0

    # Check if break duration exceeds ElevenLabs' limit (3 seconds)
    if break_sec >= 3.0:
        raise ValueError(
            f"Break duration of {break_sec:.2f}s exceeds ElevenLabs' limit of 3 seconds. "
            f"Please use a word_break_ms value less than 3000."
        )

    formatted_break = f' <break time="{break_sec:.2f}s" /> '
    text_with_breaks = formatted_break.join(tokens)

    return text_to_speech(
        text=text_with_breaks,
        voice_model=voice_model,
        speaking_rate=speaking_rate,
        is_ssml=False,
    )


def _slow_text_to_speech_google_chirp(
    tokens: list[str],
    voice_model: VoiceInfo,
    speaking_rate: float,
) -> AudioSegment:
    """Generate slow speech for Google Chirp3 HD using [pause] tags."""
    pause_tag = " [pause] "
    text_with_breaks = pause_tag.join(tokens)

    return text_to_speech(
        text=text_with_breaks,
        voice_model=voice_model,
        speaking_rate=speaking_rate,
        is_ssml=False,
    )


def _slow_text_to_speech_ssml(
    tokens: list[str],
    voice_model: VoiceInfo,
    speaking_rate: float,
    word_break_ms: int,
) -> AudioSegment:
    """Generate slow speech for Google (non-Chirp) and Azure using SSML."""
    word_break_time = str(word_break_ms) + "ms"
    ssml_parts = ["<speak>"]

    for i, token in enumerate(tokens):
        ssml_parts.append(token)
        if i < len(tokens) - 1:
            ssml_parts.append(f'<break time="{word_break_time}"/>')

    ssml_parts.append("</speak>")
    ssml_text = " ".join(ssml_parts)

    return text_to_speech(
        text=ssml_text,
        voice_model=voice_model,
        speaking_rate=speaking_rate,
        is_ssml=True,
    )
