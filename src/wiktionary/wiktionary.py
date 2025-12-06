"""Wiktionary functionality for caching and generating dictionary links."""

from typing import Dict

from pydantic import BaseModel, Field


class WiktionaryCache(BaseModel):
    """Pydantic model representing a Wiktionary cache entry in Firestore.

    This model corresponds to the wiktionary_cache collection schema defined in firestore.md.
    Each entry caches Wiktionary lookup results to avoid repeated API calls.
    """

    word: str = Field(..., description="The word being cached")
    language: str = Field(..., description="BCP-47 language code")
    valid: bool = Field(..., description="Whether Wiktionary entry exists")
    wiktionary_urls: Dict[str, str] = Field(
        ...,
        description="URLs for different Wiktionary language editions (source language)",
    )
    checked: str = Field(
        ..., description="When cache was last verified (ISO timestamp)"
    )

    def get_cache_id(self) -> str:
        """Generate the document ID for this cache entry.

        Returns:
            str: Document ID in format: {language}_{word}

        Example:
            >>> cache = WiktionaryCache(word="magasin", language="fr", ...)
            >>> cache.get_cache_id()
            'fr_magasin'
        """
        return f"{self.language}_{self.word}"
