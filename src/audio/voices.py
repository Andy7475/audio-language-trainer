"""Voice configuration management and loading from preferred_voices.json."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

from src.audio.constants import VoiceProvider


@dataclass
class VoiceInfo:
    """Information about a specific voice for text-to-speech."""

    provider: VoiceProvider
    voice_id: str
    language_code: str


def load_voices_from_json(voices_file: Optional[Path] = None) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict]]]]:
    """
    Load voice configurations from preferred_voices.json.

    Args:
        voices_file: Path to preferred_voices.json. Defaults to src/preferred_voices.json

    Returns:
        Dictionary structure: {language_code: {type: {gender: {provider, voice_id}}}}

    Raises:
        FileNotFoundError: If voices_file doesn't exist
        json.JSONDecodeError: If voices_file is not valid JSON
    """
    if voices_file is None:
        voices_file = Path(__file__).parent.parent / "preferred_voices.json"

    if not voices_file.exists():
        raise FileNotFoundError(f"Voices file not found: {voices_file}")

    with open(voices_file, "r") as f:
        return json.load(f)


def get_voice_model(
    language_code: str,
    gender: Literal["MALE", "FEMALE"],
    audio_type: Literal["flashcards", "story"] = "flashcards",
    voices_config: Optional[Dict] = None,
) -> VoiceInfo:
    """
    Get voice information for a specific language, gender, and audio type.

    Args:
        language_code: BCP47 language code (e.g., "fr-FR", "en-GB")
        gender: Voice gender ("MALE" or "FEMALE")
        audio_type: Type of audio being generated ("flashcards" or "story")
        voices_config: Pre-loaded voices configuration. If None, loads from file.

    Returns:
        VoiceInfo object containing provider and voice_id

    Raises:
        ValueError: If language_code, gender, or audio_type not found in configuration
        FileNotFoundError: If voices_config is None and voices file doesn't exist
    """
    if voices_config is None:
        voices_config = load_voices_from_json()

    if language_code not in voices_config:
        raise ValueError(f"Unsupported language: {language_code}")

    lang_config = voices_config[language_code]
    if audio_type not in lang_config:
        raise ValueError(f"Audio type '{audio_type}' not configured for {language_code}")

    type_config = lang_config[audio_type]
    gender_lower = gender.lower()

    if gender_lower not in type_config:
        raise ValueError(f"Gender '{gender}' not available for {language_code} {audio_type}")

    voice_config = type_config[gender_lower]
    provider = VoiceProvider(voice_config["provider"])

    return VoiceInfo(
        provider=provider,
        voice_id=voice_config["voice_id"],
        language_code=language_code,
    )


def get_voice_models(
    language_code: str,
    audio_type: Literal["flashcards", "story"] = "flashcards",
    voices_config: Optional[Dict] = None,
) -> tuple[VoiceInfo, VoiceInfo]:
    """
    Get both male and female voice models for a language and audio type.

    Args:
        language_code: BCP47 language code (e.g., "fr-FR", "en-GB")
        audio_type: Type of audio being generated ("flashcards" or "story")
        voices_config: Pre-loaded voices configuration. If None, loads from file.

    Returns:
        Tuple of (female_voice, male_voice)

    Raises:
        ValueError: If language_code or audio_type not found in configuration
        FileNotFoundError: If voices_config is None and voices file doesn't exist
    """
    female = get_voice_model(language_code, "FEMALE", audio_type, voices_config)
    male = get_voice_model(language_code, "MALE", audio_type, voices_config)
    return female, male
