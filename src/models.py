from typing import Annotated, Union

import langcodes
from langcodes import Language
from pydantic import PlainSerializer, BeforeValidator


def _validate_language_tag(tag: str) -> Language:
    """Validate and standardize a language tag using langcodes."""

    language = langcodes.get(tag)
    if language.is_valid() and language.territory:
        return language
    else:
        raise ValueError(f"Invalid language tag: {tag}")


# Your custom type - that's it!
BCP47Language = Annotated[
    Language,
    BeforeValidator(_validate_language_tag),
    PlainSerializer(lambda x: x.to_tag(), return_type=str, when_used="always"),
]


def get_language_code(language: Union[str, BCP47Language]) -> str:
    """
    Extract the language code from a language parameter.

    Handles both language code strings and BCP47Language objects,
    returning the 2-letter language code suitable for API calls.

    Args:
        language: Either a 2-letter language code string or a BCP47Language object

    Returns:
        str: The 2-letter language code (e.g., "en", "fr", "ja")

    Examples:
        >>> get_language_code("en")
        "en"
        >>> get_language_code(BCP47Language.get("fr-FR"))
        "fr"
    """
    if isinstance(language, str):
        return language
    else:
        # It's a BCP47Language (Language object)
        return language.language
