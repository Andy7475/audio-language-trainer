import json
import os
import time

# Assume these imports are available
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple

import azure.cognitiveservices.speech as speechsdk
from langcodes import Language, LanguageTagError
import pycountry
from google.cloud import texttospeech
import azure.cognitiveservices.speech as speechsdk
import pysnooper


class VoiceProvider(Enum):
    GOOGLE = "google"
    AZURE = "azure"
    NONE = "none"  # Added for when no provider is available


class VoiceType(Enum):
    STANDARD = "standard"
    WAVENET = "wavenet"
    NEURAL2 = "neural2"
    NEURAL = "neural"
    STUDIO = "studio"
    JOURNEY = "journey"
    NONE = "none"  # Added for when no voice type is available


@dataclass
class VoiceInfo:
    name: str
    provider: VoiceProvider
    voice_type: VoiceType
    gender: str
    language_code: str
    country_code: str
    voice_id: str


class VoiceManager:
    """Manages voice selection with lazy loading of voice data.

    Terminology:    language_code = language-country pair like fr-FR or en-GB
                    language_alpha = the language ISO code - usually 2 ALPHA standard but can be 3 (e.g. fr, en)
                    country_code = the 2 ALPHA country code (GB, FR)"""

    def __init__(self):
        self.voices: Dict[str, List[VoiceInfo]] = {}
        self.voice_type_ranking = [
            VoiceType.STUDIO,
            VoiceType.NEURAL2,
            VoiceType.WAVENET,
            VoiceType.NEURAL,
            VoiceType.STANDARD,
            VoiceType.JOURNEY,  # does not support speaking rate adjustments
        ]
        self.provider_preferences: Dict[str, VoiceProvider] = {}
        self.voice_overrides: Dict[str, Dict[str, str]] = {}
        self.loaded_languages: Set[str] = set()  # Track which languages we've loaded

    def set_voice_override(self, language_code: str, gender: str, voice_name: str):
        """Set a voice override for a specific language and gender.

        Args:
            language_code (str): Language code in format "fr-FR"
            gender (str): "MALE" or "FEMALE"
            voice_name (str): Name of the voice as it appears in self.voices[language_code]

        Example:
            voice_manager.set_voice_override("fr-FR", "MALE", "fr-FR-DenisNeural")
        """
        # Initialize nested dict if needed
        if language_code not in self.voice_overrides:
            self.voice_overrides[language_code] = {}

        # Set the override
        self.voice_overrides[language_code][gender.upper()] = voice_name
        print(f"setting voice override: {voice_name}")

    def _lazy_load_voices(self, language_code: str):
        """Only load voices for a language if we haven't already"""
        if language_code not in self.loaded_languages:
            try:
                self._load_google_voices(language_code=language_code)
            except Exception as e:
                print(f"Warning: Failed to load Google voices: {e}")

            try:
                self._load_azure_voices(locale=language_code)
            except Exception as e:
                print(f"Warning: Failed to load Azure voices: {e}")

            self.loaded_languages.add(language_code)

    def _load_google_voices(self, language_code: str):
        """Load Google voices with proper error handling"""
        try:
            from google.cloud import texttospeech

            language_object = Language.get(language_code)
            client = texttospeech.TextToSpeechClient()
            response = client.list_voices(language_code=language_code)

            for voice in response.voices:
                voice_type = VoiceType.STANDARD
                if "Neural2" in voice.name:
                    voice_type = VoiceType.NEURAL2
                elif "Studio" in voice.name:
                    voice_type = VoiceType.STUDIO
                elif "Wavenet" in voice.name:
                    voice_type = VoiceType.WAVENET
                elif "Journey" in voice.name:
                    voice_type = VoiceType.JOURNEY

                voice_info = VoiceInfo(
                    name=voice.name,
                    provider=VoiceProvider.GOOGLE,
                    voice_type=voice_type,
                    gender=voice.ssml_gender.name,
                    language_code=language_code,
                    country_code=language_object.territory,
                    voice_id=voice.name,
                )

                if language_code in self.voices:
                    self.voices[language_code].append(voice_info)
                else:
                    self.voices[language_code] = [voice_info]

        except Exception as e:
            print(f"Warning: Unable to initialize Google TTS: {e}")

    def _load_azure_voices(self, locale: str):
        """Load Azure voices with proper error handling. locale is the language_code"""
        try:

            language_object = Language.get(locale)
            speech_key = os.getenv("AZURE_API_KEY")
            if not speech_key:
                print("Warning: AZURE_API_KEY not found in environment variables")
                return

            service_region = os.getenv("AZURE_REGION", "eastus")
            speech_config = speechsdk.SpeechConfig(
                subscription=speech_key, region=service_region
            )
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config
            )

            result = speech_synthesizer.get_voices_async(locale=locale).get()
            for voice in result.voices:
                voice_type = (
                    VoiceType.NEURAL
                    if voice.voice_type._name_ == "OnlineNeural"
                    else VoiceType.STANDARD
                )

                voice_info = VoiceInfo(
                    name=voice.local_name,
                    provider=VoiceProvider.AZURE,
                    voice_type=voice_type,
                    gender=voice.gender._name_.upper(),
                    language_code=voice.locale,
                    country_code=language_object.territory,
                    voice_id=voice.short_name,
                )

                if locale in self.voices:
                    self.voices[locale].append(voice_info)
                else:
                    self.voices[locale] = [voice_info]

        except Exception as e:
            print(f"Warning: Unable to initialize Azure TTS: {e}")

    def get_voice(
        self,
        language_code: str,
        gender: str = "FEMALE",
    ) -> Optional[VoiceInfo]:
        """Get best available voice with fallback options.

        Will first check for override voices set via set_voice_override().
        If no override found, will select best voice based on ranking.

        Args:
            language_code (str): Language code in format "fr-FR"
            gender (str): "MALE" or "FEMALE"

        Returns:
            VoiceInfo object or None if no voice found
        """
        self._lazy_load_voices(language_code)

        # Check for override first
        gender = gender.upper()
        if (
            language_code in self.voice_overrides
            and gender in self.voice_overrides[language_code]
        ):
            override_voice_name = self.voice_overrides[language_code][gender]
            # Find the voice in our loaded voices
            if language_code in self.voices:
                for voice in self.voices[language_code]:
                    if voice.name == override_voice_name:
                        print(f"Using override voice: {override_voice_name}")
                        return voice

        # If no voices available at all, return dummy
        if not self.voices:
            return VoiceInfo(
                name="dummy",
                provider=VoiceProvider.NONE,
                voice_type=VoiceType.NONE,
                gender=gender,
                language_code=language_code,
                country_code="dummy",
                voice_id="dummy",
            )

        # Rest of original logic
        available_voices = self.voices.get(language_code, [])
        if not available_voices:
            return None

        gender_voices = [v for v in available_voices if v.gender == gender]
        if not gender_voices:
            gender_voices = available_voices

        preferred_provider = self.provider_preferences.get(language_code)
        if preferred_provider:
            provider_voices = [
                v for v in gender_voices if v.provider == preferred_provider
            ]
            if provider_voices:
                gender_voices = provider_voices

        sorted_voices = sorted(
            gender_voices,
            key=lambda x: (
                self.voice_type_ranking.index(x.voice_type)
                if x.voice_type in self.voice_type_ranking
                else len(self.voice_type_ranking)
            ),
        )

        return sorted_voices[0] if sorted_voices else None


class ConfigLoader:
    def __init__(self, config_file="config.json"):
        """Initialize with explicit object attributes"""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(self.script_dir, config_file)
        self.config = SimpleNamespace()
        self._last_load_time = 0
        self._file_modified_time = 0
        self.voice_manager = VoiceManager()  # No immediate voice loading
        self.time_api_last_called = 0  # New attribute for API rate limiting
        self.API_DELAY_SECONDS = 20  # Configurable delay between API calls
        self._load_config()

    def update_api_timestamp(self):
        """Update the timestamp of the last API call"""
        self.time_api_last_called = time.time()

    def get_time_since_last_api_call(self):
        """Get the number of seconds since the last API call"""
        return time.time() - self.time_api_last_called

    def _validate_language_code(
        self, language_code: str, field_name: str
    ) -> Tuple[str, str, str]:
        """Validates language code with better error handling"""
        if not language_code:
            raise ValueError(f"{field_name} must be specified")

        try:
            language_object = Language.get(language_code)

            if not language_object.is_valid():
                raise ValueError(f"{field_name} is not parsing as a valid language.")

        except LanguageTagError:
            raise LanguageTagError(
                f"{field_name} must be in format 'language-COUNTRY' (e.g., 'fr-FR')"
            )

        return (
            f"{language_object}",  # fr-FR
            language_object.language,  # fr
            language_object.display_name()
            .split("(")[0]
            .strip(),  # French (France) -> French
            language_object.territory_name(),  # France
        )

    def _load_config(self):
        """Load config with fallback values"""
        try:
            if not os.path.exists(self.config_file):
                print(f"Warning: Config file not found at {self.config_file}")
                config_dict = {
                    "TARGET_LANGUAGE_CODE": "fr-FR",  # Default values
                    "SOURCE_LANGUAGE_CODE": "en-US",
                }
            else:
                with open(self.config_file, "r") as f:
                    config_dict = json.load(f)

            # Validate and store language codes
            target_code, target_alpha2, target_name, target_country_name = (
                self._validate_language_code(
                    config_dict.get("TARGET_LANGUAGE_CODE"), "TARGET_LANGUAGE_CODE"
                )
            )
            (
                source_code,
                source_alpha2,
                source_name,
                source_country_name,
            ) = self._validate_language_code(
                config_dict.get("SOURCE_LANGUAGE_CODE"), "SOURCE_LANGUAGE_CODE"
            )

            # Update config with validated values
            config_dict.update(
                {
                    "TARGET_LANGUAGE_CODE": target_code,
                    "TARGET_LANGUAGE_ALPHA2": target_alpha2,
                    "TARGET_LANGUAGE_NAME": target_name,
                    "TARGET_COUNTRY_NAME": target_country_name,
                    "SOURCE_LANGUAGE_CODE": source_code,
                    "SOURCE_LANGUAGE_ALPHA2": source_alpha2,
                    "SOURCE_LANGUAGE_NAME": source_name,
                    "SOURCE_COUNTRY_NAME": source_country_name,
                }
            )

            self.config = SimpleNamespace(**config_dict)
            self._last_load_time = time.time()
            self._file_modified_time = (
                os.path.getmtime(self.config_file)
                if os.path.exists(self.config_file)
                else 0
            )

            if (
                hasattr(self.config, "SOURCE_LANGUAGE_MALE_VOICE")
                and self.config.SOURCE_LANGUAGE_MALE_VOICE
            ):
                self.voice_manager.set_voice_override(
                    self.config.SOURCE_LANGUAGE_CODE,
                    "MALE",
                    self.config.SOURCE_LANGUAGE_MALE_VOICE,
                )

            if (
                hasattr(self.config, "TARGET_LANGUAGE_FEMALE_VOICE")
                and self.config.TARGET_LANGUAGE_FEMALE_VOICE
            ):
                self.voice_manager.set_voice_override(
                    self.config.TARGET_LANGUAGE_CODE,
                    "FEMALE",
                    self.config.TARGET_LANGUAGE_FEMALE_VOICE,
                )

            if (
                hasattr(self.config, "TARGET_LANGUAGE_MALE_VOICE")
                and self.config.TARGET_LANGUAGE_MALE_VOICE
            ):
                self.voice_manager.set_voice_override(
                    self.config.TARGET_LANGUAGE_CODE,
                    "MALE",
                    self.config.TARGET_LANGUAGE_MALE_VOICE,
                )

        except Exception as e:
            print(f"Error loading config: {e}")
            # Set fallback values
            self.config = SimpleNamespace(
                TARGET_LANGUAGE_CODE="fr-FR",
                TARGET_LANGUAGE_ALPHA2="fr",
                TARGET_LANGUAGE_NAME="French",
                SOURCE_LANGUAGE_CODE="en-US",
                SOURCE_LANGUAGE_ALPHA2="en",
                SOURCE_LANGUAGE_NAME="English",
            )

    def get_voice_models(self):
        """Get voice models with fallback values"""
        try:
            source_voice = self.voice_manager.get_voice(
                self.config.SOURCE_LANGUAGE_CODE, gender="MALE"
            )
            target_voice_female = self.voice_manager.get_voice(
                self.config.TARGET_LANGUAGE_CODE, gender="FEMALE"
            )
            target_voice_male = self.voice_manager.get_voice(
                self.config.TARGET_LANGUAGE_CODE, gender="MALE"
            )

            return (source_voice, target_voice_female, target_voice_male)
        except Exception as e:
            print(f"Warning: Error getting voice models: {e}")

    def _check_reload(self):
        """Check if config file has been modified"""
        try:
            if os.path.exists(self.config_file):
                current_mtime = os.path.getmtime(self.config_file)
                if current_mtime > self._file_modified_time:
                    print("Config file has been modified. Reloading...")
                    self._load_config()
        except Exception as e:
            print(f"Error checking config reload: {e}")

    def __getattr__(self, name):
        """Delegate attribute access to config object after checking reload"""
        # First check if it's a direct instance attribute
        if name in self.__dict__:
            return self.__dict__[name]

        # If not, check the config namespace
        self._check_reload()
        return getattr(self.config, name)


# Create singleton instance
config = ConfigLoader()  # Delegate to config object
