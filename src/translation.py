import copy
from typing import Dict, List, Tuple, Union
from google.cloud import translate_v2 as translate
from src.config_loader import config
from tqdm import tqdm


def batch_translate(texts, batch_size=128):
    """Translate texts in batches."""
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = translate.Client().translate(
            batch, target_language=config.TARGET_LANGUAGE, source_language="en"
        )
        translated_texts.extend([item["translatedText"] for item in result])
    return translated_texts


def translate_from_english(
    text: Union[List[str], str], target_language: str = None
) -> List[str]:
    """translates text into the target_language, returns translated text. text can be a str or List[str]"""
    if target_language is None:
        target_language = config.TARGET_LANGUAGE

    result = translate.Client().translate(
        text, target_language=target_language, source_language="en"
    )

    if isinstance(result, list):
        result = [item["translatedText"] for item in result]
    else:
        result = [result["translatedText"]]
    return result


def translate_dialogue(dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """translates the 'text' part of the dialogue, keeping the speaker parts"""
    texts = [utterance["text"] for utterance in dialogue]
    translated_texts = batch_translate(texts)

    translated_dialogue = []
    for utterance, translated_text in zip(dialogue, translated_texts):
        translated_utterance = utterance.copy()
        translated_utterance["text"] = translated_text
        translated_dialogue.append(translated_utterance)

    return translated_dialogue


def translate_phrases(corrected_phrases: List[str]) -> List[Tuple[str, str]]:
    """translates a list of english phrases and returns a tuple of english, target_language
    phrases back as this is an easier format to pass into audio generation, and to manually inspect
    """
    translated_phrases = batch_translate(corrected_phrases)
    return list(zip(corrected_phrases, translated_phrases))
