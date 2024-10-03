import json
from types import SimpleNamespace
import os
import time
from typing import Dict, Optional, List
import pycountry

# Assume these imports are available


class ConfigLoader:
    def __init__(self, config_file="config.json"):
        self.config_file = self._find_config_file(config_file)
        self.config = SimpleNamespace()
        self._last_load_time = 0
        self._file_modified_time = 0
        self.english_voice_models = {}
        self.target_language_voice_models = {}
        self._load_config()

    def get_language_name(self) -> str:
        language_name = pycountry.languages.get(alpha_2=self.config.TARGET_LANGUAGE)
        if language_name:
            return language_name.name.capitalize()
        else:
            raise AttributeError(
                "Invalid TARGET_LANGUAGE code in config, use an Alpha-2 like 'en'"
            )

    def _find_config_file(self, config_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, config_file)

        if os.path.isfile(full_path):
            print(f"Found config file at: {full_path}")
            return full_path
        else:
            print(f"Could not find {config_file} in {script_dir}")
            raise FileNotFoundError(f"Could not find {config_file}")

    def _load_config(self):
        try:
            with open(self.config_file, "r") as f:
                config_dict = json.load(f)
            self.config = SimpleNamespace(**config_dict)
            self._last_load_time = time.time()
            self._file_modified_time = os.path.getmtime(self.config_file)
            self.language_name = self.get_language_name()
            print(
                f"Language name: {self.language_name} determined from code {self.config.TARGET_LANGUAGE}"
            )
            print(f"Successfully loaded config from: {self.config_file}")
            self._update_voice_models()
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Initializing with default values.")
            self.config = SimpleNamespace(
                TARGET_LANGUAGE="en",
                COUNTRY_CODE_ENGLISH="GB",
                COUNTRY_CODE_TARGET_LANGUAGE=None,
                USE_CHEAP_VOICE_MODELS=True,
            )

    def _check_reload(self):
        try:
            if os.path.getmtime(self.config_file) > self._file_modified_time:
                print("Config file has been modified. Reloading...")
                self._load_config()
        except Exception as e:
            print(f"Error checking config reload: {e}")

    def _update_voice_models(self):
        self.english_voice_models = self._get_optimal_voice_models(
            "en", self.config.COUNTRY_CODE_ENGLISH, self.config.USE_CHEAP_VOICE_MODELS
        )
        self.target_language_voice_models = self._get_optimal_voice_models(
            self.config.TARGET_LANGUAGE,
            self.config.COUNTRY_CODE_TARGET_LANGUAGE,
            self.config.USE_CHEAP_VOICE_MODELS,
        )

    def _get_optimal_voice_models(
        self,
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
        from google.cloud import texttospeech
        from google.cloud import translate_v2 as translate

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
        voices = tts_client.list_voices(language_code=language_code).voices

        if not voices:
            raise ValueError(f"No voices found for language code '{language_code}'")

        # Determine the best language code
        available_codes = set(sum((list(v.language_codes) for v in voices), []))
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

            if (
                len(gender_voices) == 0
            ):  # it means there is no gender voice available for that gender
                gender_voices = voices

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
            if not voices_to_choose:
                raise ValueError(
                    f"No voice found for within select_best_voice function and gender: {gender}, within this voice list: {voices}"
                )
            return voices_to_choose[0]

        best_male_voice = select_best_voice(voices, texttospeech.SsmlVoiceGender.MALE)
        best_female_voice = select_best_voice(
            voices, texttospeech.SsmlVoiceGender.FEMALE
        )

        return {
            "language_code": best_language_code,
            "male_voice": (best_male_voice.name if best_male_voice else None),
            "female_voice": (best_female_voice.name if best_female_voice else None),
        }

    def __getattr__(self, name):
        self._check_reload()
        return getattr(self.config, name)

    def get_voice_models(self):
        """Returns the voice models to use"""
        if not self.english_voice_models:
            self.english_voice_models = self._get_optimal_voice_models(
                "en",
                self.config.COUNTRY_CODE_ENGLISH,
                self.config.USE_CHEAP_VOICE_MODELS,
            )
        if not self.target_language_voice_models:
            self.target_language_voice_models = self._get_optimal_voice_models(
                self.config.TARGET_LANGUAGE,
                self.config.COUNTRY_CODE_TARGET_LANGUAGE,
                self.config.USE_CHEAP_VOICE_MODELS,
            )
        return self.english_voice_models, self.target_language_voice_models


config = ConfigLoader()
