"""Text processing utilities for TTS."""

import html
import re
from typing import List


def clean_tts_text(text: str) -> str:
    """
    Clean and prepare text for TTS processing by decoding HTML entities.

    Args:
        text: Input text that may contain HTML entities or special characters

    Returns:
        Cleaned text ready for TTS processing
    """
    return html.unescape(text)


def tokenize_text(text: str, language_code: str) -> List[str]:
    """
    Split text into tokens (words) for language-aware processing.

    Uses basic word splitting that works across languages. For more sophisticated
    tokenization, consider integrating with NLP libraries.

    Args:
        text: Text to tokenize
        language_code: BCP47 language code (e.g., "fr-FR", "en-GB")

    Returns:
        List of word tokens
    """
    # Remove extra whitespace and split on word boundaries
    # This handles most languages that use spaces between words
    text = text.strip()

    # Split on whitespace and punctuation, keeping track of significant tokens
    # Use regex to split on whitespace while preserving structure
    tokens = re.split(r'\s+', text)

    # Filter out empty tokens
    tokens = [token.strip() for token in tokens if token.strip()]

    return tokens
