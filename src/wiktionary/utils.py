"""Utility functions for Wiktionary operations.

This module contains helper functions used across the wiktionary package.
"""

import string
import unicodedata
from typing import Optional

from bs4 import BeautifulSoup


def clean_word_for_lookup(word: str) -> str:
    """Clean a word for Wiktionary lookup by removing leading and trailing punctuation.

    This function provides a global solution for handling punctuation in multiple languages,
    including Spanish inverted punctuation marks (¿, ¡), French quotation marks (« »),
    smart quotes, and other Unicode punctuation.

    Args:
        word: The word to clean

    Returns:
        Cleaned word with leading and trailing punctuation removed

    Examples:
        >>> clean_word_for_lookup("¿Cómo")
        'Cómo'
        >>> clean_word_for_lookup("¡Hola!")
        'Hola'
        >>> clean_word_for_lookup("«word»")
        'word'
        >>> clean_word_for_lookup("Qu'est-ce")  # Keeps internal apostrophe
        "Qu'est-ce"
    """
    if not word:
        return word

    # Define comprehensive punctuation set
    # Standard ASCII punctuation
    ascii_punct = set(string.punctuation)

    # Add Unicode punctuation commonly used in various languages
    unicode_punct = {
        # Spanish inverted punctuation
        "¿",
        "¡",
        # French/European quotation marks
        "«",
        "»",
        "‹",
        "›",
        # Smart quotes (using Unicode escape sequences)
        "\u201c",
        "\u201d",
        "\u2018",
        "\u2019",  # " " ' '
        # Ellipsis
        "\u2026",  # …
        # Dashes
        "\u2013",
        "\u2014",  # – —
        # Additional quotation marks
        "\u201e",
        "\u201c",
        "\u201a",
        "\u2018",  # „ " ‚ '
        # Mathematical and other symbols that might appear
        "\u00b0",
        "\u2032",
        "\u2033",  # ° ′ ″
    }

    # Get all Unicode punctuation categories for comprehensive coverage
    # This catches punctuation we might not have explicitly listed
    unicode_categories = {"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps"}

    def is_punctuation(char):
        """Check if a character is punctuation using multiple methods"""
        return (
            char in ascii_punct
            or char in unicode_punct
            or unicodedata.category(char) in unicode_categories
        )

    # Strip punctuation from both ends
    cleaned = word

    # Remove leading punctuation
    while cleaned and is_punctuation(cleaned[0]):
        cleaned = cleaned[1:]

    # Remove trailing punctuation
    while cleaned and is_punctuation(cleaned[-1]):
        cleaned = cleaned[:-1]

    return cleaned


def get_wiktionary_language_name(language_name: str) -> str:
    """Map standard language names to Wiktionary header names.

    Wiktionary sometimes uses different names for languages than what
    langcodes returns. This function handles those mappings.

    Args:
        language_name: Standard language name (e.g., "Mandarin Chinese")

    Returns:
        str: Wiktionary-compatible language name

    Example:
        >>> get_wiktionary_language_name("Mandarin Chinese")
        'Chinese'
        >>> get_wiktionary_language_name("Modern Greek")
        'Greek'
    """
    language_name = language_name.capitalize()
    wiktionary_mapping = {
        "Mandarin Chinese": "Chinese",
        "Modern Greek": "Greek",
        "Standard Arabic": "Arabic",
        "Brazilian Portuguese": "Portuguese",
        "European Portuguese": "Portuguese",
        # Add more mappings as discovered
    }
    return wiktionary_mapping.get(language_name, language_name)


def find_language_section(soup: BeautifulSoup, language_name: str) -> Optional[str]:
    """Find the language section in a Wiktionary page.

    Tries different strategies to locate the correct language section header.

    Args:
        soup: BeautifulSoup object of the Wiktionary page
        language_name: Language name to search for

    Returns:
        Section anchor name if found, None otherwise

    Example:
        >>> from bs4 import BeautifulSoup
        >>> html = '<h2 id="French"><span>French</span></h2>'
        >>> soup = BeautifulSoup(html, 'html.parser')
        >>> find_language_section(soup, "French")
        'French'
    """
    # Try exact match with mapping
    wiktionary_name = get_wiktionary_language_name(language_name)
    if section := soup.find("h2", {"id": wiktionary_name}):
        return wiktionary_name

    # Try words in reverse order (longest to shortest)
    # This helps with multi-word language names like "Modern Greek"
    words = language_name.split()
    for i in range(len(words), 0, -1):
        partial_name = " ".join(words[:i])
        if section := soup.find("h2", {"id": partial_name}):
            return partial_name

    # If still not found, try individual words
    for word in words:
        if section := soup.find("h2", {"id": word}):
            return word

    return None
