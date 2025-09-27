from src.translation import tokenize_text, translate_from_english
import requests
from bs4 import BeautifulSoup
import urllib.parse
import string
import unicodedata
from typing import List, Optional, Union, Tuple, Dict
from src.config_loader import config
from tqdm import tqdm


def purge_word_link_cache() -> dict:
    """
    Load the word_link_cache from GCS, remove all keys whose value is not a hyperlink, and return the cleaned dictionary.
    """
    from src.gcs_storage import get_wiktionary_cache_path, read_from_gcs

    bucket = config.GCS_PRIVATE_BUCKET
    cache_path = get_wiktionary_cache_path()
    word_link_cache = read_from_gcs(bucket, cache_path)
    # Only keep items where the value is a hyperlink (starts with <a href)
    cleaned_cache = {
        k: v
        for k, v in word_link_cache.items()
        if isinstance(v, str) and v.strip().startswith("<a href")
    }
    return cleaned_cache


def clean_word_for_lookup(word: str) -> str:
    """
    Clean a word for Wiktionary lookup by removing leading and trailing punctuation.

    This function provides a global solution for handling punctuation in multiple languages,
    including Spanish inverted punctuation marks (¿, ¡), French quotation marks (« »),
    smart quotes, and other Unicode punctuation.

    Args:
        word: The word to clean

    Returns:
        Cleaned word with leading and trailing punctuation removed

    Examples:
        >>> clean_word_for_lookup("¿Cómo")
        "Cómo"
        >>> clean_word_for_lookup("¡Hola!")
        "Hola"
        >>> clean_word_for_lookup("«word»")
        "word"
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


def generate_wiktionary_links(
    phrase: str,
    language_name: str = None,
    language_code: str = None,
    word_link_cache: Optional[Dict[str, str]] = None,
    return_cache: bool = False,
) -> Union[str, Tuple[str, Dict[str, str]]]:
    """
    Generate Wiktionary links for a given phrase in the specified language.
    Returns a string of <a href="...">...</a> HTML links for each word in the phrase. Space separated.
    If no link is found, the original word is returned within the string.

    Args:
        phrase: The phrase to generate links for
        language_name: The name of the language (defaults to config.TARGET_LANGUAGE_NAME)
        language_code: The language code (defaults to config.TARGET_LANGUAGE_CODE)
        word_link_cache: Optional dictionary mapping words to their HTML link strings
        return_cache: Whether to return the updated cache along with the links

    Returns:
        If return_cache is False: HTML string with Wiktionary links
        If return_cache is True: Tuple of (HTML string, updated cache dictionary)
    """
    if language_name is None:
        language_name = config.TARGET_LANGUAGE_NAME
    if language_code is None:
        language_code = config.TARGET_LANGUAGE_CODE

    # Initialize cache if not provided
    if word_link_cache is None:
        word_link_cache = {}
    else:
        # Make a copy to avoid modifying the original
        word_link_cache = word_link_cache.copy()

    words = tokenize_text(text=phrase, language_code=language_code)
    links: List[str] = []

    user_agent = {
        "User-Agent": "Mozilla/5.0 (compatible; WiktionaryBot/1.0; +https://github.com/Andy7475/audio-language-trainer)"
    }
    for word in words:
        # Use the new global cleaning function instead of just rstrip
        clean_word = clean_word_for_lookup(word)
        if not clean_word:
            links.append(word)
            continue

        # Use the word as is for cache lookup (preserves case)
        if clean_word in word_link_cache:
            links.append(word_link_cache[clean_word])
            continue

        # No cache hit, try to create wiktionary link
        try:
            # Try lowercase first (works for most languages)
            lookup_word = clean_word.lower()
            encoded_word = urllib.parse.quote(lookup_word)
            url = f"https://en.wiktionary.org/wiki/{encoded_word}"
            response = requests.get(url, headers=user_agent)
            found_section = False

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                if section_name := find_language_section(soup, language_name):
                    link = f'<a href="{url}#{section_name}" target="_blank" rel="noopener">{word}</a>'
                    links.append(link)
                    # Cache with the original case
                    word_link_cache[clean_word] = link.replace(word, clean_word)
                    found_section = True
            else:
                print(f"Failed to retrieve {url}")
                # print(f"Response text: {response.content}")

            # If lowercase didn't work, try capitalized (for languages like German)
            if not found_section:
                lookup_word_cap = clean_word.capitalize()
                if (
                    lookup_word_cap != lookup_word
                ):  # Only try if different from lowercase
                    encoded_word_cap = urllib.parse.quote(lookup_word_cap)
                    url_cap = f"https://en.wiktionary.org/wiki/{encoded_word_cap}"

                    response_cap = requests.get(url_cap, headers=user_agent)
                    if response_cap.status_code == 200:

                        soup_cap = BeautifulSoup(response_cap.content, "html.parser")
                        if section_name := find_language_section(
                            soup_cap, language_name
                        ):
                            link = f'<a href="{url_cap}#{section_name}" target="_blank" rel="noopener">{word}</a>'
                            links.append(link)
                            # Cache with the original case
                            word_link_cache[clean_word] = link.replace(word, clean_word)
                            found_section = True

            # If neither worked, fall back to original word
            if not found_section:
                links.append(word)
                # Cache the failure case too (as the word itself)
                word_link_cache[clean_word] = clean_word

        except requests.RequestException:
            links.append(word)
            # Cache the failure case
            word_link_cache[clean_word] = clean_word

    if return_cache:
        return " ".join(links), word_link_cache
    else:
        return " ".join(links)


def add_wiktionary_links(
    phrase_translations: dict, word_link_cache: dict = {}, overwrite: bool = False
) -> Tuple[dict, dict]:
    """
    Add wiktionary links to the phrase translations dictionary, using a cache to avoid
    duplicate queries for common words.

    Args:
        phrase_translations: Dictionary of phrase translations
        overwrite: Whether to overwrite existing wiktionary_links
        word_link_cache: Dictionary to cache word links - obtained from GCS
        collections/LM1000/translations/{config.TARGET_LANGUAGE_NAME.lower()}_wiktionary_cache.json"
    """

    for phrase_key in tqdm(phrase_translations, desc="Adding wiktionary links"):
        if "wiktionary_links" not in phrase_translations[phrase_key] or overwrite:
            phrase = phrase_translations[phrase_key][
                config.TARGET_LANGUAGE_NAME.lower()
            ]

            # Generate links and update cache
            links, word_link_cache = generate_wiktionary_links(
                phrase, word_link_cache=word_link_cache, return_cache=True
            )

            phrase_translations[phrase_key]["wiktionary_links"] = links

    return phrase_translations, word_link_cache


def generate_wiktionary_links_non_english(
    phrase: str, native_language_code: str = "uk"
) -> str:
    """
    Generate Wiktionary links for native speakers of other languages learning English.
    Similar format to the original generate_wiktionary_links function.

    Args:
        phrase: The English phrase to generate links for
        native_language_code: The two-letter language code (e.g., 'uk' for Ukrainian)

    Returns:
        HTML string with Wiktionary links in the native language
    """
    words = phrase.split()
    links: List[str] = []

    # Get translation of "English" in target language
    try:
        english_in_target = translate_from_english("English", native_language_code)
        if isinstance(english_in_target, list):
            english_in_target = english_in_target[0]
        english_in_target = english_in_target.capitalize()
    except Exception:
        # Fallback to "English" if translation fails
        english_in_target = "English"

    user_agent = {
        "User-Agent": "Mozilla/5.0 (compatible; WiktionaryBot/1.0; +https://github.com/Andy7475/audio-language-trainer)"
    }
    for word in words:
        clean_word = "".join(char for char in word if char.isalnum())
        if clean_word:
            # Lowercase the word for URL and search, but keep original for display
            lowercase_word = clean_word.lower()
            # URL encode the lowercase word to handle non-ASCII characters
            encoded_word = urllib.parse.quote(lowercase_word)
            # First try native language Wiktionary
            native_url = (
                f"https://{native_language_code}.wiktionary.org/wiki/{encoded_word}"
            )

            try:
                response = requests.get(native_url, headers=user_agent)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    # Look for the English section using h2 tag
                    language_section = None
                    for heading_level in range(1, 7):
                        if soup.find(f"h{heading_level}", {"id": english_in_target}):
                            language_section = True
                            break

                    if language_section:
                        # If found, create a link with the anchor to the specific language section
                        link = f'<a href="{native_url}#{english_in_target}">{word}</a>'
                        links.append(link)
                    else:
                        # If not found in native Wiktionary, try English Wiktionary
                        english_url = f"https://en.wiktionary.org/wiki/{encoded_word}"
                        link = f'<a href="{english_url}#English">{word}</a>'
                        links.append(link)
                else:
                    # If native Wiktionary fails, use English Wiktionary
                    english_url = f"https://en.wiktionary.org/wiki/{encoded_word}"
                    link = f'<a href="{english_url}#English">{word}</a>'
                    links.append(link)
            except requests.RequestException:
                # If request fails, add without link
                links.append(word)
        else:
            links.append(word)

    return " ".join(links)


def get_wiktionary_language_name(language_name: str) -> str:
    """Map standard language names to Wiktionary header names"""

    language_name = language_name.capitalize()
    wiktionary_mapping = {
        "Mandarin Chinese": "Chinese",
        "Modern Greek": "Greek",
        "Standard Arabic": "Arabic",
        "Brazilian Portuguese": "Portuguese",
        # Add more mappings as discovered
    }
    return wiktionary_mapping.get(language_name, language_name)


def find_language_section(soup: BeautifulSoup, language_name: str) -> Optional[str]:
    """Try different strategies to find the language section"""
    # Try exact match with mapping
    wiktionary_name = get_wiktionary_language_name(language_name)
    if section := soup.find("h2", {"id": wiktionary_name}):
        return wiktionary_name

    # Try words in reverse order (longest to shortest)
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
