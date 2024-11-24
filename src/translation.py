import copy
from typing import Dict, List, Tuple, Union
from google.cloud import translate_v2 as translate
from src.config_loader import config
from tqdm import tqdm
from google.cloud import language_v1
from google.api_core import retry


def batch_translate(texts, batch_size=128):
    """Translate texts in batches."""
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = translate.Client().translate(
            batch, target_language=config.TARGET_LANGUAGE_ALPHA2, source_language="en"
        )
        translated_texts.extend([item["translatedText"] for item in result])
    return translated_texts


def translate_from_english(
    text: Union[List[str], str], target_language: str = None
) -> List[str]:
    """translates text into the target_language, returns translated text. text can be a str or List[str]"""
    if target_language is None:
        target_language = config.TARGET_LANGUAGE_ALPHA2

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


def tokenize_text(
    text: str, language_code: str = config.TARGET_LANGUAGE_ALPHA2
) -> List[str]:
    """
    Tokenize text in a way that's practical for:
    1. Adding SSML breaks between words
    2. Looking up words in Wiktionary

    For space-separated languages: Simply split on spaces
    For CJK languages: Use API but merge very short tokens that likely belong together
    For Thai: Use API but accept larger chunks if tokenization fails

    Args:
        text: Text to tokenize
        language_code: Two-letter or three-letter, language code (e.g. 'en', 'ja')

    Returns:
        List of tokens suitable for TTS breaks and Wiktionary lookups
    """
    if not text:
        return []

    # For languages that use spaces
    if " " in text and not any(
        ord(char) > 0x4E00 for char in text
    ):  # Quick check for CJK
        return [token for token in text.split() if token]

    # For languages needing API
    try:
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT,
            language=language_code,
        )

        response = client.analyze_syntax(
            request={
                "document": document,
                "encoding_type": language_v1.EncodingType.UTF8,
            }
        )

        tokens = [token.text.content for token in response.tokens]

        # Post-process tokens for CJK languages
        if language_code in ("zh", "ja", "ko", "cmn", "yue", "wuu", "hak", "nan"):
            # Merge single characters that likely belong together
            merged_tokens = []
            current_token = ""

            for token in tokens:
                if len(token) == 1 and len(current_token) < 3:
                    current_token += token
                else:
                    if current_token:
                        merged_tokens.append(current_token)
                    current_token = token

            if current_token:
                merged_tokens.append(current_token)

            return merged_tokens

        return tokens if tokens else text.split()

    except Exception as e:
        print(f"API Tokenization failed: {str(e)}")
        # Fallback: split on spaces if present, otherwise return whole text as one token
        return text.split() if " " in text else [text]
