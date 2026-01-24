"""Text processing utilities for TTS."""

import html


def clean_tts_text(text: str) -> str:
    """
    Clean and prepare text for TTS processing by decoding HTML entities.

    Args:
        text: Input text that may contain HTML entities or special characters

    Returns:
        Cleaned text ready for TTS processing
    """
    return html.unescape(text)
