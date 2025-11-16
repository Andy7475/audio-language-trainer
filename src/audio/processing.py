"""Audio processing utilities: joining, speeding up, and exporting audio."""

import io
import uuid
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from src.audio.constants import (
    AUDIO_SPEED_FAST,
    AUDIO_SPEED_NORMAL,
    DE_REVERB_TIME,
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_GAP_MS,
    DEFAULT_FRAME_RATE,
)


def join_audio_segments(
    audio_segments: List[AudioSegment],
    gap_ms: int = DEFAULT_GAP_MS,
) -> AudioSegment:
    """
    Join multiple audio segments together with a gap between each.

    Args:
        audio_segments: List of AudioSegment objects to join
        gap_ms: Gap duration in milliseconds between segments (default: 100ms)

    Returns:
        Single AudioSegment with all segments joined

    Raises:
        ValueError: If audio_segments list is empty
    """
    if not audio_segments:
        raise ValueError("audio_segments cannot be empty")

    if len(audio_segments) == 1:
        return audio_segments[0]

    gap_audio = AudioSegment.silent(duration=gap_ms)
    result = AudioSegment.empty()

    for i, segment in enumerate(audio_segments):
        result += segment
        if i < len(audio_segments) - 1:
            result += gap_audio

    return result


def speed_up_audio(
    audio_segment: AudioSegment,
    speed_factor: float = AUDIO_SPEED_FAST,
) -> AudioSegment:
    """
    Speed up an AudioSegment without changing its pitch, with de-reverb applied.

    Uses librosa for time-stretching to maintain pitch while speeding up.

    Args:
        audio_segment: Input AudioSegment
        speed_factor: Factor by which to speed up the audio (e.g., 2.0 for double speed)

    Returns:
        Sped up AudioSegment

    Raises:
        ValueError: If speed_factor is <= 0
        Exception: If audio processing fails
    """
    if speed_factor <= 0:
        raise ValueError(f"speed_factor must be positive, got {speed_factor}")

    if speed_factor == AUDIO_SPEED_NORMAL:
        return audio_segment

    try:
        # Convert AudioSegment to numpy array
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

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
        freq = librosa.fft_frequencies(sr=sample_rate)
        reverb_decay = np.exp(
            -np.outer(freq, np.arange(D.shape[1]) / sample_rate) / DE_REVERB_TIME
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

    except Exception as e:
        raise Exception(f"Audio speed-up processing error: {str(e)}")


def export_audio(
    audio_segment: AudioSegment,
    filename: Optional[str] = None,
    format: str = DEFAULT_AUDIO_FORMAT,
) -> str:
    """
    Export an AudioSegment to a file.

    Args:
        audio_segment: The AudioSegment to export
        filename: Output filename. If None, a UUID-based filename is generated.
        format: Audio format (default: "mp3")

    Returns:
        Path to the exported audio file

    Raises:
        Exception: If export fails
    """
    if filename is None:
        filename = f"{uuid.uuid4()}.{format}"

    try:
        audio_segment.export(filename, format=format)
        return filename
    except Exception as e:
        raise Exception(f"Audio export error: {str(e)}")
