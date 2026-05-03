"""Wiktionary lookup functionality.

This module handles fetching dictionary entries from a local database.
"""

from typing import List, Optional
import string

from connections.wiktionary import get_wiktionary_db
from utils import normalize_lang_code_for_wiktionary

# Wiktionary pos values that represent real content words (excludes names,
# soft-redirects, romanizations, characters, etc.)
CONTENT_WORD_POS = ["verb", "noun", "adj"]


def get_wiktionary_urls(
    words: List[str],
    lang_code: str,
    pos: Optional[str] = None,
    pos_list: Optional[List[str]] = None,
) -> List[str]:
    """Get a list of text hyperlinks.

    Args:
        words: Tokens to look up.
        lang_code: BCP-47 language code.
        pos: Exact part of speech to match (e.g. 'verb'). Takes priority over pos_list.
        pos_list: Accept any of these pos values (e.g. ['verb', 'noun', 'adj']).
    """
    return [_find_wiktionary_url_from_token(word, lang_code, pos, pos_list) for word in words]


def _find_wiktionary_url_from_token(
    word: str,
    lang_code: str,
    pos: Optional[str] = None,
    pos_list: Optional[List[str]] = None,
) -> str:
    """Try a few case variants until one returns a wiktionary link."""

    def _is_link(url: str) -> bool:
        return url.startswith("<a href")

    cases_to_try = [word, word.lower(), word.title(), word.upper()]
    punc_strip = [w.strip(string.punctuation) for w in cases_to_try]
    for search_word in cases_to_try + punc_strip:
        url = get_wiktionary_url(
            search_word=search_word,
            original_word=word,
            lang_code=lang_code,
            pos=pos,
            pos_list=pos_list,
        )
        if _is_link(url):
            return url
    return word


def get_wiktionary_url(
    search_word: str,
    original_word: str,
    lang_code: str,
    pos: Optional[str] = None,
    pos_list: Optional[List[str]] = None,
) -> str:
    """Get a Wiktionary URL for a word, optionally filtered by part of speech.

    Args:
        search_word: The word to look up (may be a case variant).
        original_word: The original token to use as link text.
        lang_code: Language code (e.g. 'en', 'nl', 'de').
        pos: Exact pos to require (e.g. 'verb'). Takes priority over pos_list.
        pos_list: Accept any of these pos values (e.g. ['verb', 'noun', 'adj']).
                  Ignored when pos is set.

    Returns:
        HTML <a> tag if a matching entry exists, otherwise the original word.
    """
    normalized_lang_code = normalize_lang_code_for_wiktionary(lang_code)
    cursor = get_wiktionary_db().cursor()

    if pos:
        cursor.execute(
            "SELECT url FROM entries WHERE word=? AND lang_code=? AND pos=?",
            (search_word, normalized_lang_code, pos),
        )
    elif pos_list:
        placeholders = ",".join("?" * len(pos_list))
        cursor.execute(
            f"SELECT url FROM entries WHERE word=? AND lang_code=? AND pos IN ({placeholders})",
            (search_word, normalized_lang_code, *pos_list),
        )
    else:
        cursor.execute(
            "SELECT url FROM entries WHERE word=? AND lang_code=?",
            (search_word, normalized_lang_code),
        )

    result = cursor.fetchone()
    if result:
        return f'<a href="{result[0]}" target="_blank">{original_word}</a>'
    return original_word
