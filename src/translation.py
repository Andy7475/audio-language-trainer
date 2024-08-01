import copy
from typing import Dict, List, Tuple, Union
from google.cloud import translate_v2 as translate
from src.config_loader import config
from tqdm import tqdm


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
    """translates the 'text' part of the dialogue, keeping the speaker parts
    there"""

    list_english_utterances = [utterance["text"] for utterance in dialogue]
    translated_utterances = translate_from_english(list_english_utterances)

    translated_dialogue = copy.deepcopy(dialogue)

    for dialogue_part, translation in zip(translated_dialogue, translated_utterances):
        dialogue_part["text"] = translation

    return translated_dialogue


def translate_phrases(corrected_phrases: List[str]) -> List[Tuple[str, str]]:
    """translates a list of english phrases and returns a tuple of english, target_language
    phrases back as this is an easier format to pass into audio generation, and to manually inspect
    """

    translated_phrases = zip(
        corrected_phrases, translate_from_english(corrected_phrases)
    )

    return list(translated_phrases)
