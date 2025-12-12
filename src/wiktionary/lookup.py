"""Wiktionary web lookup functionality.

This module handles fetching dictionary entries from Wiktionary's web interface.
"""

import urllib.parse
from typing import Optional, Literal

import requests
from bs4 import BeautifulSoup

from src.wiktionary.models import WiktionaryEntry
from src.wiktionary.utils import (
    clean_word_for_lookup,
    get_wiktionary_language_name,
    find_language_section,
)


USER_AGENT = {
    "User-Agent": "Mozilla/5.0 (compatible; WiktionaryBot/1.0; +https://github.com/Andy7475/audio-language-trainer)"
}


def fetch_wiktionary_entry(
    token: str,
    language_code: str,
    timeout: int = 10,
) -> WiktionaryEntry:
    """Fetch a wiktionary entry from the web.

    Tries multiple lookup strategies:
    1. Lowercase version
    2. Capitalized version (for languages like German)
    3. Original case

    Args:
        token: The word/token to look up
        language_code: ISO 639-1 language code (e.g., 'en', 'fr', 'de')
        timeout: Request timeout in seconds (default: 10)

    Returns:
        WiktionaryEntry: Entry with exists=True if found, exists=False otherwise

    Example:
        >>> entry = fetch_wiktionary_entry("Haus", "de")  # German noun
        >>> print(entry.exists)
        True
        >>> print(entry.lookup_variant)
        'capitalized'
    """
    # Clean the token for lookup
    clean_token = clean_word_for_lookup(token)
    if not clean_token:
        return _create_not_found_entry(token, language_code)

    token_lower = clean_token.lower()

    # Get the language name for Wiktionary section lookup
    language_name = _get_language_name_from_code(language_code)

    # Try lowercase first
    entry = _try_wiktionary_lookup(
        token_lower,
        language_code,
        language_name,
        variant="lowercase",
        timeout=timeout,
    )
    if entry.exists:
        return entry

    # Try capitalized (for German nouns, etc.)
    token_cap = clean_token.capitalize()
    if token_cap != token_lower:
        entry = _try_wiktionary_lookup(
            token_cap,
            language_code,
            language_name,
            variant="capitalized",
            timeout=timeout,
        )
        if entry.exists:
            return entry

    # Try original case if different from both
    if clean_token not in [token_lower, token_cap]:
        entry = _try_wiktionary_lookup(
            clean_token,
            language_code,
            language_name,
            variant="original",
            timeout=timeout,
        )
        if entry.exists:
            return entry

    # Not found with any variant
    return _create_not_found_entry(token_lower, language_code)


def _try_wiktionary_lookup(
    lookup_word: str,
    language_code: str,
    language_name: str,
    variant: Literal["lowercase", "capitalized", "original"],
    timeout: int,
) -> WiktionaryEntry:
    """Try to look up a word on Wiktionary with a specific variant.

    Args:
        lookup_word: The exact word to look up
        language_code: ISO 639-1 code
        language_name: Full language name for section matching
        variant: Which variant this lookup represents
        timeout: Request timeout

    Returns:
        WiktionaryEntry: Entry with exists=True if found
    """
    try:
        encoded_word = urllib.parse.quote(lookup_word)
        url = f"https://en.wiktionary.org/wiki/{encoded_word}"

        response = requests.get(url, headers=USER_AGENT, timeout=timeout)

        if response.status_code != 200:
            return _create_not_found_entry(lookup_word.lower(), language_code)

        soup = BeautifulSoup(response.content, "html.parser")
        section_name = find_language_section(soup, language_name)

        if section_name:
            return WiktionaryEntry(
                token=lookup_word.lower(),
                language_code=language_code,
                exists=True,
                url=url,
                section_anchor=f"#{section_name}",
                lookup_variant=variant,
            )

        return _create_not_found_entry(lookup_word.lower(), language_code)

    except requests.RequestException as e:
        print(f"Wiktionary lookup failed for '{lookup_word}': {e}")
        return _create_not_found_entry(lookup_word.lower(), language_code)


def _create_not_found_entry(
    token: str,
    language_code: str,
) -> WiktionaryEntry:
    """Create a WiktionaryEntry for a word not found on Wiktionary.

    Args:
        token: The lowercase token
        language_code: ISO 639-1 code

    Returns:
        WiktionaryEntry with exists=False
    """
    return WiktionaryEntry(
        token=token.lower(),
        language_code=language_code,
        exists=False,
        url=None,
        section_anchor=None,
        lookup_variant=None,
    )


def _get_language_name_from_code(language_code: str) -> str:
    """Get the full language name from ISO 639-1 code.

    Args:
        language_code: ISO 639-1 code like 'en', 'fr', 'de'

    Returns:
        str: Language name suitable for Wiktionary section lookup

    Example:
        >>> _get_language_name_from_code("en")
        'English'
        >>> _get_language_name_from_code("fr")
        'French'
    """
    # Use langcodes to get the display name
    import langcodes

    try:
        lang = langcodes.get(language_code)
        language_name = lang.display_name("en")

        # Apply Wiktionary-specific mappings
        return get_wiktionary_language_name(language_name)

    except Exception:
        # Fallback: capitalize the code
        return language_code.capitalize()
