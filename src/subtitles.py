import pysrt
import re
from typing import List, Optional
from nlp import get_text_tokens
import string
from wiktionary.lookup import get_wiktionary_urls, CONTENT_WORD_POS


def read_subtitles(file_path: str) -> pysrt.SubRipFile:
    """
    Read an srt file and return a pysrt SubRipFile object.

    Args:
        file_path: Path to the .srt file.

    Returns:
        pysrt.SubRipFile object containing the parsed subtitles.
    """
    return pysrt.open(file_path)


def remove_cc_info(text: str) -> str:
    """Remove Closed Captioning info enclosed in square brackets."""
    return re.sub(r"\[.*?\]", "", text)


def process_subtitles(
    subs: pysrt.SubRipFile, language_code: str, split_on_space: bool = False
) -> List[str]:
    """
    Process subtitles to remove CC info (in square brackets),
    remove duplicates, and return unique phrases.

    Args:
        subs: pysrt.SubRipFile object

    Returns:
        List of unique string phrases
    """
    unique_phrases = []
    seen = set()

    for sub in subs:
        # Get text
        text = sub.text

        # Remove CC info in square brackets
        text = remove_cc_info(text)

        # Remove newlines and extra spaces
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()

        # Strip leading speaker-change hyphens (e.g. "- Flytta" or "-Flytta")
        text = re.sub(r"^-\s*", "", text).strip()

        if text and text not in seen:
            seen.add(text)
            unique_phrases.append(text)

    return unique_phrases


def strip_punctuation(token: str) -> str:
    """Strip punctuation from the start and end of a token."""
    return token.strip(string.punctuation + " ")


def filter_wiktionary_words(
    words: List[str],
    language_code: str,
    pos_list: Optional[List[str]] = None,
) -> List[str]:
    """Filter out words not found in Wiktionary.

    Args:
        words: Tokens to filter.
        language_code: BCP-47 language code.
        pos_list: If provided, only accept words that exist in Wiktionary with
                  one of these pos values. Defaults to None (accept any pos).
    """
    urls = get_wiktionary_urls(words, language_code, pos_list=pos_list)
    return [
        word
        for word, url_or_word in zip(words, urls)
        if url_or_word.startswith("<a href")
    ]


def get_subtitle_tokens(
    tokens: List[str],
    language_code: str,
    min_length: int = 3,
    to_lower: bool = True,
    pos_list: Optional[List[str]] = CONTENT_WORD_POS,
) -> List[str]:
    """Strip punctuation, filter short tokens, lowercase, and verify against Wiktionary content words."""
    tokens = [strip_punctuation(t) for t in tokens]
    valid_tokens = [t for t in tokens if len(t) >= min_length]
    if to_lower:
        valid_tokens = [t.lower() for t in valid_tokens]
    unique_tokens = list(set(valid_tokens))
    return sorted(
        filter_wiktionary_words(unique_tokens, language_code, pos_list=pos_list)
    )
