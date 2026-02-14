"""High-level audio generation functions."""

from typing import List, Literal, Tuple

from google.cloud import texttospeech
from pydub import AudioSegment

from src.audio.constants import (
    SPEAKING_RATE_NORMAL,
    SPEAKING_RATE_SLOW,
    DEFAULT_WORD_BREAK_MS,
    AUDIO_SPEED_FAST,
)
from src.audio.processing import join_audio_segments, speed_up_audio
from src.audio.providers import text_to_speech, slow_text_to_speech
from src.audio.voices import VoiceInfo, get_voice_model
from src.connections.gcloud_auth import (
    get_texttospeech_long_client,
    setup_authentication,
)
from src.storage import gcs_uri_from_file_path, PUBLIC_BUCKET


def generate_translation_audio(
    translated_text: str,
    voice_model: VoiceInfo,
    speed: Literal["normal", "slow"] = "normal",
    speaking_rate: float = SPEAKING_RATE_NORMAL,
    word_break_ms: int = DEFAULT_WORD_BREAK_MS,
    split_on_space: bool = False,
) -> AudioSegment:
    """
    Generate audio for translated text at specified speed.

    All audio is linked to a translation. This function generates audio
    for a single translation in either normal or slow speed.

    Args:
        translated_text: Translated text to convert to speech
        voice_model: VoiceInfo object containing provider and voice details
        speed: "normal" for natural speed, "slow" for slowed speech with word breaks
        speaking_rate: Speaking rate multiplier (1.0 = normal)
        word_break_ms: Break time between words in milliseconds (for slow speech)

    Returns:
        AudioSegment containing the generated speech

    Raises:
        ValueError: If speed is not "normal" or "slow"
        Exception: If TTS generation fails
    """
    if speed not in ("normal", "slow"):
        raise ValueError(f"speed must be 'normal' or 'slow', got {speed}")

    if speed == "slow":
        return slow_text_to_speech(
            text=translated_text,
            voice_model=voice_model,
            speaking_rate=SPEAKING_RATE_SLOW,
            word_break_ms=word_break_ms,
            split_on_space=split_on_space,
        )
    else:
        return text_to_speech(
            text=translated_text,
            voice_model=voice_model,
            speaking_rate=speaking_rate,
        )


def generate_fast_audio(
    audio_segment: AudioSegment, speed_factor: float = AUDIO_SPEED_FAST
) -> AudioSegment:
    """
    Generate fast audio from a normal audio segment using local audio processing.

    This function speeds up audio locally without making any API calls.
    It uses librosa for time-stretching to maintain pitch.

    Args:
        audio_segment: Audio segment to speed up
        speed_factor: Factor by which to speed up (default: 2.0 for 2x speed)

    Returns:
        AudioSegment with sped-up audio

    Raises:
        ValueError: If speed_factor is <= 0
        Exception: If audio processing fails
    """
    return speed_up_audio(audio_segment, speed_factor)


def generate_normal_and_fast_audio(
    audio_segments: List[AudioSegment],
    normal_gap_ms: int = 500,
    fast_gap_ms: int = 100,
) -> Tuple[AudioSegment, AudioSegment]:
    """
    Generate both normal speed and fast speed versions from a list of audio segments.

    Joins segments with appropriate gaps, then creates a 2x speed version
    of the normal speed audio using local audio processing.

    Args:
        audio_segments: List of AudioSegment objects
        normal_gap_ms: Gap between segments in normal speed audio (default: 500ms)
        fast_gap_ms: Gap between segments in fast speed audio (default: 100ms)

    Returns:
        Tuple of (normal_speed_audio, fast_speed_audio)

    Raises:
        ValueError: If audio_segments is empty
        Exception: If audio processing fails
    """
    if not audio_segments:
        raise ValueError("audio_segments cannot be empty")

    # Join segments at normal speed
    normal_audio = join_audio_segments(audio_segments, gap_ms=normal_gap_ms)

    # Create fast version by speeding up the segments individually
    fast_segments = [generate_fast_audio(segment) for segment in audio_segments]
    fast_audio = join_audio_segments(fast_segments, gap_ms=fast_gap_ms)

    return normal_audio, fast_audio


def text_to_speech_long_form(
    text: str,
    filename: str,
    bucket_name: str = PUBLIC_BUCKET,
    language_code: str = "en-GB",
    gender: Literal["MALE", "FEMALE"] = "MALE",
    timeout: int = 300,
) -> str:
    """
    Generate long-form audio from text using Google's Long Audio Synthesis API.

    This function uses Google's specialized long-form TTS API which supports
    text up to 1 million bytes. The audio is generated directly to GCS as WAV format.

    Note: Currently only LINEAR16 (WAV) encoding is supported by the Long Audio API.

    Args:
        text: Text to convert to speech (up to 1 million bytes)
        filename: Output filename without extension (e.g., "my_story")
        bucket_name: GCS bucket name for output
        language_code: BCP-47 language code (default: "en-GB")
        gender: Voice gender (default: "MALE")
        timeout: Timeout in seconds for the operation (default: 300)

    Returns:
        str: Public URL to the generated WAV file

    Raises:
        RuntimeError: If audio generation fails or times out
        ValueError: If parameters are invalid
    """
    # Get the appropriate voice model
    voice_info = get_voice_model(language_code, gender, audio_type="story")

    # Get authenticated project ID
    _, project_id = setup_authentication()

    # Create the output GCS URI (must be WAV for long-form audio)
    file_path = f"long_form_audio/{filename}.wav"
    output_gcs_uri = gcs_uri_from_file_path(file_path, bucket_name)

    # Get the long-form TTS client
    client = get_texttospeech_long_client()

    # Configure the synthesis input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Configure audio output as LINEAR16 (WAV) - currently the only supported format
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    # Configure voice selection
    voice = texttospeech.VoiceSelectionParams(
        language_code=voice_info.language_code,
        name=voice_info.voice_id,
    )

    # Set up the request
    parent = f"projects/{project_id}/locations/us-central1"
    request = texttospeech.SynthesizeLongAudioRequest(
        parent=parent,
        input=synthesis_input,
        audio_config=audio_config,
        voice=voice,
        output_gcs_uri=output_gcs_uri,
    )

    print(f"Generating long-form audio for '{filename}'...")
    print(f"Text length: {len(text)} characters")
    print(f"Voice: {voice_info.voice_id} ({language_code}, {gender})")
    print(f"Output: {output_gcs_uri}")

    # Start the long-running operation
    operation = client.synthesize_long_audio(request=request)

    try:
        # Wait for the operation to complete
        operation.result(timeout=timeout)
        print("(y) Audio generation complete!")

        # Return the public URL
        public_url = f"https://storage.googleapis.com/{bucket_name}/{file_path}"
        return public_url

    except Exception as e:
        raise RuntimeError(f"Long-form audio generation failed: {e}")
