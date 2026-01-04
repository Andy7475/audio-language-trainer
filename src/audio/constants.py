"""Audio processing constants and configuration."""

from enum import Enum
from pydub import AudioSegment
from pathlib import Path
# Audio processing constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_RATE = 44100
DEFAULT_AUDIO_FORMAT = "mp3"

# Speaking rates
SPEAKING_RATE_SLOW = 0.85
SPEAKING_RATE_NORMAL = 1.0

# Audio segment gaps (in milliseconds)
DEFAULT_GAP_MS = 100
DEFAULT_STORY_GAP_MS = 500

# Word break timing for slow speech
DEFAULT_WORD_BREAK_MS = 250

# Audio speed factors
AUDIO_SPEED_NORMAL = 1.0
AUDIO_SPEED_FAST = 2.0

# De-reverb settings for audio processing
DE_REVERB_TIME = 0.3  # Adjust to control de-reverb strength (in seconds)

INTER_UTTERANCE_GAP = AudioSegment.silent(duration=100)

# Locate project root (two levels up from src/audio), then data/transition.mp3
# Use resolve() to handle symlinks and ensure an absolute path.
project_root = Path(__file__).resolve().parents[2]
audio_file = project_root / "data" / "transition.mp3"

STORY_PART_TRANSITION = AudioSegment.from_file(audio_file, format="mp3")


class VoiceProvider(str, Enum):
    """Supported text-to-speech providers."""

    GOOGLE = "google"
    AZURE = "azure"
    ELEVENLABS = "elevenlabs"


class AudioType(str, Enum):
    """Types of audio content being generated."""

    FLASHCARD = "flashcard"
    STORY = "story"
