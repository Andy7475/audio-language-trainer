# Audio Module Guide

The audio module in `src/audio/` provides text-to-speech (TTS), audio processing, and voice management capabilities. It's been refactored to remove global configuration dependencies and use parameter-based APIs.

## Overview

The audio module is split into logical submodules:

- **`voices.py`** - Voice configuration loading from `preferred_voices.json`
- **`providers.py`** - TTS provider implementations (Google, Azure, ElevenLabs)
- **`text_processing.py`** - Text cleaning and tokenization
- **`processing.py`** - Audio manipulation (joining, speeding up, exporting)
- **`generation.py`** - High-level generation functions
- **`constants.py`** - Constants and configuration values

## Key Concepts

### Voice Models
All TTS operations require a `VoiceInfo` object that specifies:
- **provider**: Which TTS service to use (Google, Azure, ElevenLabs)
- **voice_id**: The specific voice identifier for that provider
- **language_code**: BCP47 language code (e.g., "fr-FR", "en-GB")

### Audio Types
The module supports two audio types, loaded from `preferred_voices.json`:
- **flashcards**: For phrase/flashcard audio (typically higher quality)
- **story**: For narrative/story audio (may use different voices)

### Speaking Rates
- **SPEAKING_RATE_NORMAL** (1.0): Natural speech speed
- **SPEAKING_RATE_SLOW** (0.85): Slower speech for learning

## Usage Examples

### 1. Loading Voice Configurations

```python
from src.audio import load_voices_from_json, get_voice_model, get_voice_models

# Load all voice configurations
all_voices = load_voices_from_json()

# Get a specific voice for French female flashcard audio
voice = get_voice_model(
    language_code="fr-FR",
    gender="FEMALE",
    audio_type="flashcards"
)
print(f"Provider: {voice.provider}, Voice ID: {voice.voice_id}")

# Get both male and female voices for a language
female_voice, male_voice = get_voice_models("fr-FR", "flashcards")
```

### 2. Basic Text-to-Speech

```python
from src.audio import text_to_speech, get_voice_model

# Get a voice model
voice = get_voice_model("fr-FR", "FEMALE", "flashcards")

# Generate normal speed audio
audio = text_to_speech(
    text="Bonjour, comment allez-vous?",
    voice_model=voice,
    speaking_rate=1.0
)

# Speaking rate can be adjusted (0.5 for half-speed, 2.0 for double-speed)
audio_fast = text_to_speech(
    text="Bonjour!",
    voice_model=voice,
    speaking_rate=1.2
)
```

### 3. Slow Speech with Word Breaks

```python
from src.audio import slow_text_to_speech, get_voice_model
from src.audio import SPEAKING_RATE_SLOW, DEFAULT_WORD_BREAK_MS

voice = get_voice_model("fr-FR", "FEMALE", "flashcards")

# Generate slow speech with breaks between words
audio = slow_text_to_speech(
    text="Bonjour, comment allez-vous?",
    voice_model=voice,
    speaking_rate=SPEAKING_RATE_SLOW,  # 0.85x speed
    word_break_ms=250  # 250ms break between words
)

# Different providers handle breaks differently:
# - Google Chirp3 HD: Uses [pause] tags
# - Google (non-Chirp) & Azure: Uses SSML <break> tags
# - ElevenLabs: Uses <break time="X.XXs" /> tags
# The module handles all this automatically based on voice_model.provider
```

### 4. High-Level Generation Functions

```python
from src.audio import (
    generate_translation_audio,
    generate_fast_audio,
    generate_normal_and_fast_audio,
    get_voice_model
)

voice = get_voice_model("fr-FR", "FEMALE", "flashcards")

# Generate audio at different speeds from the same text
normal_audio = generate_translation_audio(
    translated_text="Bonjour",
    voice_model=voice,
    speed="normal"
)

slow_audio = generate_translation_audio(
    translated_text="Bonjour",
    voice_model=voice,
    speed="slow"
)

# Generate fast audio locally (no API call!)
fast_audio = generate_fast_audio(normal_audio, speed_factor=2.0)

# Generate both normal and fast from segments
segments = [normal_audio]  # In practice, multiple audio segments
normal, fast = generate_normal_and_fast_audio(segments)
```

### 5. Audio Processing

```python
from src.audio import (
    join_audio_segments,
    speed_up_audio,
    export_audio,
    generate_fast_audio
)

# Join multiple audio segments with gaps
segments = [audio1, audio2, audio3]
joined = join_audio_segments(segments, gap_ms=500)

# Speed up audio locally (maintains pitch)
fast_audio = speed_up_audio(audio, speed_factor=2.0)
# or use the high-level function
fast_audio = generate_fast_audio(audio)

# Export to file
filename = export_audio(audio, filename="output.mp3", format="mp3")
```

### 6. Text Processing

```python
from src.audio import clean_tts_text, tokenize_text

# Clean HTML entities and special characters
text = "Bonjour &#39;from&#39; France"
cleaned = clean_tts_text(text)  # Returns: Bonjour 'from' France

# Tokenize text into words
tokens = tokenize_text("Bonjour comment ça va", language_code="fr-FR")
# Returns: ["Bonjour", "comment", "ça", "va"]
```

## Complete Example: Generating Translation Audio

```python
from src.audio import (
    load_voices_from_json,
    get_voice_model,
    generate_translation_audio,
    export_audio
)

# Initialize
voices_config = load_voices_from_json()
voice = get_voice_model(
    language_code="fr-FR",
    gender="FEMALE",
    audio_type="flashcards",
    voices_config=voices_config
)

# Generate audio at different speeds
normal = generate_translation_audio(
    translated_text="Bonjour, comment allez-vous?",
    voice_model=voice,
    speed="normal"
)

slow = generate_translation_audio(
    translated_text="Bonjour, comment allez-vous?",
    voice_model=voice,
    speed="slow"
)

# Export to files
normal_file = export_audio(normal, "normal_speed.mp3")
slow_file = export_audio(slow, "slow_speed.mp3")
```

## Error Handling

The module raises exceptions for errors. Always use try/except:

```python
from src.audio import text_to_speech, get_voice_model

try:
    # Non-existent language
    voice = get_voice_model("xx-XX", "FEMALE", "flashcards")
except ValueError as e:
    print(f"Invalid language: {e}")

try:
    # Missing API keys
    audio = text_to_speech(text, voice_model)
except ValueError as e:
    print(f"Missing credentials: {e}")
except Exception as e:
    print(f"TTS generation failed: {e}")
```

## Voice Provider Details

### Google Cloud TTS
- Requires: `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to credentials JSON
- Supports: SSML and markup formats
- Chirp3 HD voices: Use markup format with [pause] tags for breaks
- Other voices: Use SSML with <break> tags for breaks

### Azure Speech Service
- Requires: `AZURE_API_KEY` and `AZURE_REGION` environment variables
- Supports: SSML format
- Uses: <break> tags in SSML for word breaks
- Default region: "eastus" if not specified

### ElevenLabs
- Requires: `ELEVENLABS_API_KEY` environment variable
- Supports: Native <break time="X.XXs" /> tags (not standard SSML)
- Limitation: 3-second maximum break duration
- Model: Multilingual v2

## Voice Configuration Files

Voices are loaded from `src/preferred_voices.json`, which has this structure:

```json
{
  "language-code": {
    "flashcards": {
      "female": {
        "provider": "google|azure|elevenlabs",
        "voice_id": "provider-specific-voice-id"
      },
      "male": {
        "provider": "google|azure|elevenlabs",
        "voice_id": "provider-specific-voice-id"
      }
    },
    "story": {
      "female": { ... },
      "male": { ... }
    }
  }
}
```

## Performance Notes

- **Fast Audio Generation**: Uses local audio processing (librosa) - no API calls required
- **TTS Generation**: Requires API calls to Google, Azure, or ElevenLabs
- **Audio Joining**: Fast operation, minimal memory overhead
- **Speed Up**: Uses spectral subtraction for de-reverb, may take a few seconds

## Migration from Old Code

If migrating from the old `ARCHIVE/audio_generation.py`:

**Old way:**
```python
from src.config_loader import config
audio = text_to_speech("Hello", gender="FEMALE")
```

**New way:**
```python
from src.audio import text_to_speech, get_voice_model
voice = get_voice_model("en-GB", "FEMALE", "flashcards")
audio = text_to_speech("Hello", voice_model=voice)
```

Key differences:
- No global `config` object
- Explicit `voice_model` parameter (use `get_voice_model()`)
- Language codes are explicit parameters
- No automatic gender/language mapping - pass them explicitly

## Constants

Available constants in `src.audio.constants`:

```python
from src.audio import (
    SPEAKING_RATE_SLOW,      # 0.85
    SPEAKING_RATE_NORMAL,    # 1.0
    AUDIO_SPEED_NORMAL,      # 1.0
    AUDIO_SPEED_FAST,        # 2.0
    DEFAULT_GAP_MS,          # 100 (gap between audio segments)
    DEFAULT_WORD_BREAK_MS,   # 250 (break between words in slow speech)
    DEFAULT_SAMPLE_RATE,     # 16000
    DEFAULT_FRAME_RATE,      # 44100
    VoiceProvider,           # Enum: GOOGLE, AZURE, ELEVENLABS
)
```

## Troubleshooting

**"Unsupported language" error:**
- Check that your language code exists in `preferred_voices.json`
- Language codes must be in BCP47 format (e.g., "fr-FR", not "fr")

**"API request failed" from ElevenLabs:**
- Verify `ELEVENLABS_API_KEY` is set correctly
- Check that the voice_id in preferred_voices.json is valid
- ElevenLabs breaks are limited to 3 seconds maximum

**"Speech synthesis failed" from Azure:**
- Verify `AZURE_API_KEY` and `AZURE_REGION` are set
- Check that the voice name is valid for the region
- Some voice names are region-specific

**Audio processing/speed up errors:**
- Ensure librosa and soundfile are installed: `uv pip install librosa soundfile`
- Check that input audio is in a supported format

