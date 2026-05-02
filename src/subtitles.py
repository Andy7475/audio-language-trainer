import pysrt
import re
from typing import List
from nlp import get_text_tokens
import string
from wiktionary.lookup import get_wiktionary_urls


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
        words = get_text_tokens(text, language_code, split_on_space=split_on_space)
        text = " ".join(strip_punctuation(word) for word in words)

        if text and text not in seen:
            seen.add(text)
            unique_phrases.append(text)

    return unique_phrases


def strip_punctuation(token: str) -> str:
    """Strip punctuation from the start and end of a token."""
    return token.strip(string.punctuation + " ")


def filter_wiktionary_words(words: List[str], language_code: str) -> List[str]:
    """
    Filter out words that are not found in Wiktionary.
    This helps remove names, non-words, or typos.
    """
    valid_words = []
    # get_wiktionary_urls returns an <a href=...> string if found, otherwise the original word
    urls = get_wiktionary_urls(words, language_code)
    for word, url_or_word in zip(words, urls):
        if url_or_word.startswith("<a href"):
            valid_words.append(word)
    return valid_words


def get_subtitle_tokens(
    phrases: List[str], language_code: str, to_lower: bool = True
) -> List[str]:
    """
    Get a list of valid Wiktionary tokens from a list of subtitle phrases.
    Uses basic tokenization and Wiktionary filtering, suitable for languages
    without robust NL API support (like Swedish).
    """
    # Combine phrases
    combined_text = " ".join(phrases)

    # Get tokens FIRST so we can preserve internal punctuation
    tokens = get_text_tokens(combined_text, language_code, split_on_space=True)

    # Strip punctuation from boundaries
    tokens = [strip_punctuation(t) for t in tokens]

    # Optional lowercase
    if to_lower:
        tokens = [t.lower() for t in tokens]

    # Remove empty strings
    tokens = [t for t in tokens if t.strip()]

    # Remove duplicates but keep unique set
    unique_tokens = list(set(tokens))

    # Filter by Wiktionary
    valid_tokens = filter_wiktionary_words(unique_tokens, language_code)

    return valid_tokens
