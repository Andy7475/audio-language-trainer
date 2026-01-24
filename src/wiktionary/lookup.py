"""Wiktionary  lookup functionality.

This module handles fetching dictionary entries from a local datbase.
"""

from typing import List
import string

# Database location (save in the wiktionary path)
from src.connections.wiktionary import get_wiktionary_db
from src.utils import normalize_lang_code_for_wiktionary


def get_wiktionary_urls(words: List[str], lang_code: str) -> List[str]:
    """Get a list of text hyperlinks

    Args:
        words (list[str]): tokens in translation model
        lang_code (str):
    """
    return [_find_wiktionary_url_from_token(word, lang_code) for word in words]


def _find_wiktionary_url_from_token(word: str, lang_code: str) -> str:
    """We try a few case variants if the original token does not work"""

    def _is_link(url: str) -> bool:
        return url.startswith("<a href")

    cases_to_try = [word, word.lower(), word.title(), word.upper()]
    punc_strip = [word.strip(string.punctuation) for word in cases_to_try]
    words_to_try = cases_to_try + punc_strip
    for search_word in words_to_try:
        url = get_wiktionary_url(
            search_word=search_word, original_word=word, lang_code=lang_code
        )
        if _is_link(url):
            return url
    return word


def get_wiktionary_url(search_word: str, original_word: str, lang_code: str):
    """
    Get Wiktionary URL for a word in a specific language.
    Returns an HTML link if the entry exists, otherwise returns the word as-is.

    Args:
        word: The word to look up
        lang_code: Language code (e.g., 'en', 'nl', 'de')

    Returns:
        str: HTML <a> tag if entry exists, otherwise the word
    """

    normalized_lang_code = normalize_lang_code_for_wiktionary(lang_code)
    _conn = get_wiktionary_db()

    cursor = _conn.cursor()
    cursor.execute(
        "SELECT url FROM entries WHERE word=? AND lang_code=?",
        (search_word, normalized_lang_code),
    )
    result = cursor.fetchone()

    if result:
        url = result[0]
        return f'<a href="{url}" target="_blank">{original_word}</a>'
    else:
        return original_word
