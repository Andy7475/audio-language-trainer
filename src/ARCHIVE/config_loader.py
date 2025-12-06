import json
import os
import time
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Dict

from langcodes import Language, LanguageTagError


class VoiceProvider(Enum):
    GOOGLE = "google"
    AZURE = "azure"
    ELEVENLABS = "elevenlabs"


@dataclass
class VoiceInfo:
    name: str
    provider: VoiceProvider
    voice_id: str
    language_code: str


class VoiceManager:
    """Manages voice selection based on preferred_voices.json configuration."""

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.preferred_voices_file = os.path.join(
            self.script_dir, "preferred_voices.json"
        )
        self.preferred_voices = self._load_preferred_voices()
        self.elevenlabs_client = None

    def _load_preferred_voices(self) -> Dict:
        """Load preferred voices configuration."""
        if not os.path.exists(self.preferred_voices_file):
            raise FileNotFoundError(
                f"Preferred voices file not found at {self.preferred_voices_file}"
            )

        with open(self.preferred_voices_file, "r") as f:
            return json.load(f)

    def _init_elevenlabs_client(self):
        """Initialize the ElevenLabs client if not already done."""
        if self.elevenlabs_client is None:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                raise ValueError(
                    "ELEVENLABS_API_KEY not found in environment variables"
                )

            try:
                from elevenlabs import ElevenLabs

                self.elevenlabs_client = ElevenLabs(api_key=api_key)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize ElevenLabs client: {e}")

    def get_voice(self, language_code: str, gender: str, enum_type: str) -> VoiceInfo:
        """Get voice configuration for specified language, gender and enum type.

        Args:
            language_code: Language code (e.g. "fr-FR")
            gender: "male" or "female"
            enum_type: "phrases" or "stories"

        Returns:
            VoiceInfo object with voice configuration

        Raises:
            KeyError: If voice configuration not found
            ValueError: If provider is invalid
        """
        try:
            voice_config = self.preferred_voices[language_code][enum_type][
                gender.lower()
            ]
        except KeyError:
            raise KeyError(
                f"No voice configuration found for {language_code} {enum_type} {gender}. Preferred voices: {self.preferred_voices}"
            )

        provider = voice_config["provider"]
        voice_id = voice_config["voice_id"]

        try:
            provider_enum = VoiceProvider(provider)
        except ValueError:
            raise ValueError(f"Invalid provider: {provider}")

        return VoiceInfo(
            name=voice_id,  # Using voice_id as name for simplicity
            provider=provider_enum,
            voice_id=voice_id,
            language_code=language_code,
        )


class ConfigLoader:
    def __init__(self, config_file="config.json"):
        """Initialize with explicit object attributes"""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(self.script_dir, config_file)
        self.config = SimpleNamespace()
        self._last_load_time = 0
        self._file_modified_time = 0
        self.voice_manager = VoiceManager()
        self.time_api_last_called = 0
        self.API_DELAY_SECONDS = 20
        self._load_config()

    def update_api_timestamp(self):
        """Update the timestamp of the last API call"""
        self.time_api_last_called = time.time()

    def get_time_since_last_api_call(self):
        """Get the number of seconds since the last API call"""
        return time.time() - self.time_api_last_called

    def _validate_language_code(
        self, language_code: str, field_name: str
    ) -> tuple[str, str, str, str]:
        """Validates language code with better error handling and returns language details.

        Args:
            language_code: Language code to validate (e.g. "fr-FR")
            field_name: Name of the field being validated (for error messages)

        Returns:
            Tuple of (language_code, alpha2_code, language_name, country_name)

        Raises:
            ValueError: If language code is empty or invalid
            LanguageTagError: If language code format is incorrect
        """
        if not language_code:
            raise ValueError(f"{field_name} must be specified")

        try:
            language_object = Language.get(language_code)
            if not language_object.is_valid():
                raise ValueError(f"{field_name} is not parsing as a valid language.")

            # Extract components
            language_code = str(language_object)
            alpha2_code = language_object.language
            language_name = language_object.language_name()
            country_name = (
                language_object.territory_name() if language_object.territory else ""
            )

            return language_code, alpha2_code, language_name, country_name

        except LanguageTagError:
            raise LanguageTagError(
                f"{field_name} must be in format 'language-COUNTRY' (e.g., 'fr-FR')"
            )

    def _load_config(self):
        """Load config with fallback values"""
        try:
            if not os.path.exists(self.config_file):
                raise FileNotFoundError(f"Config file not found at {self.config_file}")
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

            # Refresh preferred voices in voice manager
            self.voice_manager.preferred_voices = (
                self.voice_manager._load_preferred_voices()
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    def get_voice_models(self, enum_type: str = "phrases"):
        """
        Get voice models for specified enum type.

        Args:
            enum_type: "phrases" or "stories"

        Returns:
            Tuple of (source_voice_female, source_voice_male, target_voice_female, target_voice_male)

        Raises:
            KeyError: If voice configuration not found
        """
        source_voice_female = self.voice_manager.get_voice(
            self.config.SOURCE_LANGUAGE_CODE, "female", enum_type
        )
        source_voice_male = self.voice_manager.get_voice(
            self.config.SOURCE_LANGUAGE_CODE, "male", enum_type
        )
        target_voice_female = self.voice_manager.get_voice(
            self.config.TARGET_LANGUAGE_CODE, "female", enum_type
        )
        target_voice_male = self.voice_manager.get_voice(
            self.config.TARGET_LANGUAGE_CODE, "male", enum_type
        )

        return (
            source_voice_female,
            source_voice_male,
            target_voice_female,
            target_voice_male,
        )

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
        self._check_reload()
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.config, name)


# Create singleton instance
config = ConfigLoader()
