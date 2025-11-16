from typing import List, Optional, Union

import langcodes

from src.connections.gcloud_auth import get_translate_client
from src.llm_tools.review_translation import refine_translation
from src.models import BCP47Language, get_language_code



def translate_with_google_translate(
    text: Union[str, List[str]],
    target_language: Union[str, BCP47Language],
    source_language: Union[str, BCP47Language] = "en",
    batch_size: int = 128
) -> Union[str, List[str]]:
    """
    Translate text using Google Translate API.

    Supports both single strings and lists of strings, with automatic batching
    for large lists.

    Args:
        text: The text to translate (string or list of strings)
        target_language: BCP47Language object or 2-letter language code for the target language
        source_language: BCP47Language object or 2-letter language code for source language (default: "en" for English)
        batch_size: Number of texts to process per batch (default: 128)

    Returns:
        str if input is str, List[str] if input is List[str]

    Raises:
        RuntimeError: If translation fails
    """
    try:
        translate_client = get_translate_client()

        # Extract language codes
        # Google Translate API uses 2-letter codes
        target_code = get_language_code(target_language)
        source_code = get_language_code(source_language)

        # Normalize input to list and track if input was a single string
        is_single_string = isinstance(text, str)
        texts = [text] if is_single_string else text

        # Process in batches
        translated_texts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = translate_client.translate(
                batch,
                target_language=target_code,
                source_language=source_code,
                format_="text",
            )
            translated_texts.extend([item["translatedText"] for item in result])

        # Return in the same format as input
        return translated_texts[0] if is_single_string else translated_texts

    except Exception as e:
        raise RuntimeError(f"Failed to translate text with Google Translate: {e}")


def refine_translation_with_anthropic(
    source_phrase: str,
    initial_translation: str,
    target_language: Union[str, BCP47Language],
    source_language: Union[str, BCP47Language] = None,
    model: Optional[str] = None
) -> str:
    """
    Refine a translation using Anthropic's Claude API.

    Args:
        source_phrase: The original source language phrase
        initial_translation: The initial Google Translate translation
        target_language: BCP47Language object or language name string for the target language
        source_language: BCP47Language object or language name string for the source language (default: English)
        model: Anthropic model to use (default: claude-sonnet-4-20250514)

    Returns:
        str: The refined translation text

    Raises:
        RuntimeError: If refinement fails
    """
    if model is None:
        model = "claude-sonnet-4-20250514"

    # Get target language name (e.g., "French" from "fr-FR")
    if isinstance(target_language, str) and len(target_language) == 2:
        # It's a language code, map it to language name
        target_language_name = langcodes.get(target_language).display_name()
    elif isinstance(target_language, str):
        # It's already a language name
        target_language_name = target_language
    else:
        # It's a BCP47Language object
        target_language_name = target_language.display_name()

    # Use the llm_tools module
    # Note: source_language parameter is accepted for future compatibility
    # but the current llm_tools.refine_translation expects english_phrase
    return refine_translation(
        english_phrase=source_phrase,
        initial_translation=initial_translation,
        target_language_name=target_language_name,
        model=model
    )

