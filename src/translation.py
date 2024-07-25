import copy
from typing import Dict, List, Tuple
from google.cloud import translate_v2 as translate
from src.config_loader import config
from tqdm import tqdm


def translate_from_english(text, target_language: str = None) -> str:
    """translates text into the target_language, returns translated text"""
    if target_language is None:
        target_language = config.TARGET_LANGUAGE

    result = translate.Client().translate(
        text, target_language=target_language, source_language="en"
    )
    return result["translatedText"]


def translate_dialogue(dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """translates the 'text' part of the dialogue, keeping the speaker parts
    there"""
    translated_dialogue = copy.deepcopy(dialogue)
    for utterance in tqdm(translated_dialogue):
        text = utterance["text"]
        translated_text = translate_from_english(text)
        utterance["text"] = translated_text

    return translated_dialogue


def translate_phrases(corrected_phrases: List[str]) -> List[Tuple[str, str]]:
    """translates a list of english phrases and returns a tuple of english, target_language
    phrases back as this is an easier format to pass into audio generation, and to manually inspect
    """

    translated_phrases = []
    for phrase in tqdm(corrected_phrases):
        translated_phrase = translate_from_english(phrase)
        translated_phrases.append((phrase, translated_phrase))

    return translated_phrases
