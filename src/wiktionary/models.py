"""Pydantic models for Wiktionary caching in Firestore.

This module defines the data models for storing Wiktionary lookup results
in Firestore, keyed by token and language code.
"""

from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, Field


class WiktionaryEntry(BaseModel):
    """Represents a single Wiktionary entry for a token in a specific language.

    Firestore structure:
        Collection: wiktionary
        Document ID: {lowercase_token}  (e.g., "hello", "bonjour")
        Subcollection: languages
        Document ID: {language_code}  (e.g., "en", "fr", "ja")

    This two-level structure allows efficient querying:
    - Get all language entries for a token: wiktionary/{token}/languages/*
    - Get specific language entry: wiktionary/{token}/languages/{lang_code}
    - Handles 1MB Firestore document limit by separating languages
    """

    token: str = Field(
        ...,
        description="The lowercase token (word) being cached",
    )
    language_code: str = Field(
        ...,
        description="ISO 639-1 language code (e.g., 'en', 'fr', 'ja', 'de')",
    )
    exists: bool = Field(
        ...,
        description="Whether a valid Wiktionary entry exists for this token",
    )
    url: Optional[str] = Field(
        default=None,
        description="Full Wiktionary URL if entry exists, None otherwise",
    )
    section_anchor: Optional[str] = Field(
        default=None,
        description="HTML anchor to the language section (e.g., '#French')",
    )
    lookup_variant: Optional[Literal["lowercase", "capitalized", "original"]] = Field(
        default=None,
        description="Which variant was used to find the entry (for German capitalized nouns, etc.)",
    )

    def get_html_link(self, display_text: str) -> str:
        """Generate HTML link for this token.

        Args:
            display_text: The text to display in the link (preserves original casing/punctuation)

        Returns:
            HTML anchor tag if entry exists, plain text otherwise

        Example:
            >>> entry = WiktionaryEntry(
            ...     token="hello",
            ...     language_code="en",
            ...     exists=True,
            ...     url="https://en.wiktionary.org/wiki/hello",
            ...     section_anchor="#English"
            ... )
            >>> entry.get_html_link("Hello")
            '<a href="https://en.wiktionary.org/wiki/hello#English" target="_blank" rel="noopener">Hello</a>'
        """
        if not self.exists or not self.url:
            return display_text

        full_url = self.url
        if self.section_anchor:
            full_url = f"{self.url}{self.section_anchor}"

        return f'<a href="{full_url}" target="_blank" rel="noopener">{display_text}</a>'

    def to_firestore_dict(self) -> dict:
        """Convert to dictionary suitable for Firestore storage.

        Converts datetime to ISO string for Firestore compatibility.
        """
        data = self.model_dump()
        return data

    @classmethod
    def from_firestore_dict(cls, data: dict) -> "WiktionaryEntry":
        """Create instance from Firestore document data.

        Converts ISO string back to datetime.
        """
        return cls(**data)


def get_firestore_path(token: str, language_code: str) -> tuple[str, str]:
    """Get Firestore collection and document path for a wiktionary entry.

    Args:
        token: The word/token to look up (will be lowercased)
        language_code: ISO 639-1 language code (e.g., 'en', 'fr')

    Returns:
        tuple: (collection_path, document_id)
            - collection_path: Path to the wiktionary collection
            - document_id: Composite document ID

    Example:
        >>> get_firestore_path("Hello", "en")
        ('wiktionary', 'en_hello')
        >>> get_firestore_path("Magasin", "fr")
        ('wiktionary', 'fr_magasin')
    """
    token_lower = token.lower()
    document_id = f"{language_code}_{token_lower}"

    return "wiktionary", document_id
