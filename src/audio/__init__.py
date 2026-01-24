"""Audio module for text-to-speech, audio processing, and voice management."""

from src.audio.constants import (
    AUDIO_SPEED_FAST,
    AUDIO_SPEED_NORMAL,
    DEFAULT_GAP_MS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WORD_BREAK_MS,
    SPEAKING_RATE_NORMAL,
    SPEAKING_RATE_SLOW,
    VoiceProvider,
)
from src.audio.generation import (
    generate_fast_audio,
    generate_normal_and_fast_audio,
    generate_translation_audio,
)
from src.audio.processing import (
    export_audio,
    join_audio_segments,
    speed_up_audio,
)
from src.audio.providers import (
    slow_text_to_speech,
    text_to_speech,
)
from src.audio.text_processing import (
    clean_tts_text,
)
from src.audio.voices import (
    VoiceInfo,
    get_voice_model,
    get_voice_models,
    load_voices_from_json,
)

__all__ = [
    # Constants
    "AUDIO_SPEED_FAST",
    "AUDIO_SPEED_NORMAL",
    "DEFAULT_GAP_MS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_WORD_BREAK_MS",
    "SPEAKING_RATE_NORMAL",
    "SPEAKING_RATE_SLOW",
    "VoiceProvider",
    # Voice management
    "VoiceInfo",
    "load_voices_from_json",
    "get_voice_model",
    "get_voice_models",
    # Text processing
    "clean_tts_text",
    # TTS providers
    "text_to_speech",
    "slow_text_to_speech",
    # Audio processing
    "join_audio_segments",
    "speed_up_audio",
    "export_audio",
    # High-level generation
    "generate_translation_audio",
    "generate_fast_audio",
    "generate_normal_and_fast_audio",
]
