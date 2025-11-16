import json
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from google.cloud import language_v1

from src.config_loader import config
from src.connections.gcloud_auth import get_translate_client
from src.llm_tools.review_translations import review_batch_translations
from src.llm_tools.review_story_translations import review_story_dialogue


def review_translations_with_anthropic(
    phrase_pairs: List[Dict[str, str]],
    target_language: str = None,
    model: str = None,
) -> List[Dict[str, Any]]:
    """
    Use Anthropic API to review and improve translations using the tool interface.

    Args:
        phrase_pairs: List of dictionaries containing {'english': 'phrase', 'translation': 'current translation'}
        target_language: The target language for translations (defaults to config.TARGET_LANGUAGE_NAME)
        model: Anthropic model to use

    Returns:
        List of dictionaries with {'english': str, 'translation': str, 'modified': bool}
    """
    # Default to config target language if not specified
    if target_language is None:
        target_language = config.TARGET_LANGUAGE_NAME.lower()

    if model is None:
        model = config.ANTHROPIC_MODEL_NAME

    # Use the new llm_tools module
    return review_batch_translations(
        phrase_pairs=phrase_pairs,
        target_language_name=target_language,
        source_language_name=config.SOURCE_LANGUAGE_NAME,
        model=model
    )


def batch_translate(texts, batch_size=128):
    """Translate texts in batches."""
    translate_client = get_translate_client()
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = translate_client.translate(
            batch,
            target_language=config.TARGET_LANGUAGE_ALPHA2,
            source_language="en",
            format_="text",
        )
        translated_texts.extend([item["translatedText"] for item in result])
    return translated_texts


def translate_from_english(
    text: Union[List[str], str], target_language: str = None
) -> List[str]:
    """translates text into the target_language, returns translated text. text can be a str or List[str]"""
    if target_language is None:
        target_language = config.TARGET_LANGUAGE_ALPHA2

    translate_client = get_translate_client()
    result = translate_client.translate(
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


def tokenize_text(text: str, language_code: Optional[str] = None) -> List[str]:
    """
    Tokenize text using language-appropriate methods.

    For space-separated languages: Simply split on spaces
    For other languages: Use Google Cloud Natural Language API

    Args:
        text: Text to tokenize
        language_code: Two-letter language code (e.g. 'en', 'ja', defaults to current config language)

    Returns:
        List of tokens suitable for TTS breaks and Wiktionary lookups
    """
    # Evaluate language_code at runtime to pick up config changes
    if language_code is None:
        language_code = config.TARGET_LANGUAGE_CODE

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
        return tokens if tokens else text.split()

    except Exception as e:
        print(f"API Tokenization failed: {str(e)}")
        # Fallback: split on spaces if present, otherwise return whole text as one token
        return text.split() if " " in text else [text]


def review_translated_phrases_batch(
    translated_phrases: Dict[str, Dict[str, str]],
    target_language: str = None,
    batch_size: int = 20,
    model: str = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Wrapper function to batch review translations from the translated_phrases format.

    Args:
        translated_phrases: Dictionary with phrase_keys containing {'english': 'phrase', target_language: 'translation'}
        target_language: The target language for translations (defaults to config.TARGET_LANGUAGE_NAME)
        batch_size: Number of phrases to process in each batch (default: 20)
        model: Anthropic model to use

    Returns:
        Dictionary with same structure but with 'modified' field added to each translation
    """
    if target_language is None:
        target_language = config.TARGET_LANGUAGE_NAME.lower()

    if model is None:
        model = config.ANTHROPIC_MODEL_NAME
    # Convert to list of (phrase_key, phrase_data) tuples for batching
    phrase_items = list(translated_phrases.items())
    result = {}

    # Process in batches
    for i in range(0, len(phrase_items), batch_size):
        batch_items = phrase_items[i : i + batch_size]

        # Convert batch to phrase_pairs format
        phrase_pairs = []
        batch_keys = []

        for phrase_key, phrase_data in batch_items:
            phrase_pairs.append(
                {
                    "english": phrase_data["english"],
                    "translation": phrase_data[target_language],
                }
            )
            batch_keys.append(phrase_key)

        print(
            f"Processing batch {i//batch_size + 1} with {len(phrase_pairs)} phrases..."
        )

        # Call the original function
        reviewed_batch = review_translations_with_anthropic(
            phrase_pairs=phrase_pairs, target_language=target_language, model=model
        )

        # Convert results back to original format
        for j, reviewed_translation in enumerate(reviewed_batch):
            phrase_key = batch_keys[j]
            result[phrase_key] = {
                "english": reviewed_translation["english"],
                target_language: reviewed_translation["translation"],
                "modified": reviewed_translation["modified"],
            }

    return result


def review_story_dialogue_translations(
    story_dialogue: Dict[str, Dict[str, Any]],
    target_language: str = None,
    model: str = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Review and improve translations for a story's dialogue using Anthropic API.

    Args:
        story_dialogue: Dictionary where each key is a story part containing dialogue and translated_dialogue
        target_language: The target language for translations (defaults to config.TARGET_LANGUAGE_NAME)
        model: Anthropic model to use
        verbose: If True, print out old and new translations when changes are made

    Returns:
        Dictionary with same structure but with improved translations
    """
    # Default to config target language if not specified
    if target_language is None:
        target_language = config.TARGET_LANGUAGE_NAME.lower()
    if model is None:
        model = config.ANTHROPIC_MODEL_NAME

    # Flatten the dialogue for processing
    flattened_dialogue = []
    part_lengths = {}  # Keep track of how many utterances are in each part

    for part_name, part_data in story_dialogue.items():
        if "dialogue" in part_data and "translated_dialogue" in part_data:
            part_lengths[part_name] = len(part_data["dialogue"])
            for eng, trans in zip(
                part_data["dialogue"], part_data["translated_dialogue"]
            ):
                # Validate speaker
                if eng["speaker"] not in ["Sam", "Alex"]:
                    raise ValueError(
                        f"Invalid speaker: {eng['speaker']}. Must be either 'Sam' or 'Alex'"
                    )

                flattened_dialogue.append(
                    {
                        "speaker": eng["speaker"],
                        "text": eng["text"],
                        "translation": trans["text"],
                        "modified": False,
                    }
                )

    if not flattened_dialogue:
        print("No dialogue found to review")
        return story_dialogue

    # Use the new llm_tools module
    try:
        reviewed_translations = review_story_dialogue(
            flattened_dialogue=flattened_dialogue,
            target_language_name=target_language,
            source_language_name=config.SOURCE_LANGUAGE_NAME,
            model=model
        )

        # Reconstruct the story structure
        result = {}
        current_index = 0

        for part_name, length in part_lengths.items():
            if part_name in story_dialogue:
                # Get the slice of reviewed translations for this part
                part_reviewed = reviewed_translations[
                    current_index : current_index + length
                ]

                # Create new part data
                result[part_name] = {
                    "dialogue": story_dialogue[part_name]["dialogue"],
                    "translated_dialogue": [
                        {
                            "speaker": item["speaker"],
                            "text": item["translation"],
                        }
                        for item in part_reviewed
                    ],
                }

                # Print modified translations if verbose is True
                if verbose:
                    for i, (old, reviewed) in enumerate(
                        zip(
                            story_dialogue[part_name]["translated_dialogue"],
                            part_reviewed,
                        )
                    ):
                        if (
                            reviewed["modified"]
                            and old["text"] != reviewed["translation"]
                        ):
                            print(
                                f"English: {story_dialogue[part_name]['dialogue'][i]['text']}"
                            )
                            print(f"Old: {old['text']}")
                            print(f"New: {reviewed['translation']}")
                            print()

            current_index += length

        return result

    except Exception as e:
        print(f"API call error: {e}")
        return story_dialogue
