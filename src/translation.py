from google.cloud import translate_v2 as translate
from src.config_loader import config


def translate_from_english(text, target_language: str = None) -> str:
    """translates text into the target_language, returns translated text"""
    if target_language is None:
        target_language = config.TARGET_LANGUAGE

    result = translate.Client().translate(
        text, target_language=target_language, source_language="en"
    )
    return result["translatedText"]
