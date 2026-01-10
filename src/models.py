from typing import Annotated, Union

from langcodes import Language
from pydantic import PlainSerializer, BeforeValidator


def _validate_language_tag(tag: str) -> Language:
    """Validate and standardize a language tag using langcodes."""

    language = Language.get(tag)
    if language.is_valid() and language.territory:
        return language
    else:
        raise ValueError(f"Invalid language tag: {tag}")


BCP47Language = Annotated[
    Language,
    BeforeValidator(_validate_language_tag),
    PlainSerializer(lambda x: x.to_tag(), return_type=str, when_used="always"),
]


def get_language(language: Union[str, Language]) -> Language:
    """
    Convert language parameter to Language (Language) object.

    Normalizes language input - accepts either a Language object or a string
    language tag (e.g., "fr-FR") and returns a Language object.

    Args:
        language: Either a Language object or a string language tag (e.g., "fr-FR")

    Returns:
        Language: Normalized language object

    Examples:
        >>> lang = get_language("fr-FR")  # Returns Language object
        >>> lang = get_language(langcodes.get("fr-FR"))  # Returns same object
        >>> lang.to_tag()
        'fr-FR'
    """
    if isinstance(language, str):
        return Language.get(language)
    return language
