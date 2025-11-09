import langcodes
from langcodes import Language
from pydantic import BaseModel, Field, PlainSerializer, BeforeValidator
from typing import List, Optional, Literal, Annotated

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
    PlainSerializer(lambda x: x.to_tag(), return_type=str, when_used="always")
]