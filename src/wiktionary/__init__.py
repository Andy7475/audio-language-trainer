"""Wiktionary integration for language learning materials.

This package provides Firestore-based caching and web lookup for Wiktionary entries.
Key features:
- Language code-based storage (not territory, e.g., 'en' not 'en-GB')
- Flat Firestore structure: wiktionary/{lang_code}_{token}
- Batch operations for efficient lookups
- Token-preserving HTML link generation
"""

from src.cache import (
    get_wiktionary_entry,
    save_wiktionary_entry,
    batch_get_wiktionary_entries,
    batch_save_wiktionary_entries,
    get_or_fetch_wiktionary_entry,
    batch_get_or_fetch_wiktionary_entries,
)

from src.lookup import fetch_wiktionary_entry

from src.models import WiktionaryEntry, get_firestore_path

from src.utils import (
    clean_word_for_lookup,
    get_wiktionary_language_name,
    find_language_section,
)


__all__ = [
    # Cache operations
    "get_wiktionary_entry",
    "save_wiktionary_entry",
    "batch_get_wiktionary_entries",
    "batch_save_wiktionary_entries",
    "get_or_fetch_wiktionary_entry",
    "batch_get_or_fetch_wiktionary_entries",
    # Web lookup
    "fetch_wiktionary_entry",
    # Models
    "WiktionaryEntry",
    "get_firestore_path",
    # Utilities
    "clean_word_for_lookup",
    "get_wiktionary_language_name",
    "find_language_section",
]
