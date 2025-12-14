"""Firestore-based caching for Wiktionary lookups.

This module provides functions to:
- Query the wiktionary cache in Firestore
- Store new wiktionary entries
- Batch operations for multiple tokens
- Extract language codes from BCP47Language objects
"""

from typing import List, Optional, Dict
from datetime import datetime, timedelta

from src.connections.gcloud_auth import get_firestore_client
from src.wiktionary.models import WiktionaryEntry, get_firestore_path


def get_wiktionary_entry(
    token: str,
    language_code: str,
    database_name: str = "firephrases",
) -> Optional[WiktionaryEntry]:
    """Retrieve a wiktionary entry from Firestore cache.

    Args:
        token: The word/token to look up (case-insensitive)
        language_code: ISO 639-1 language code (e.g., 'en', 'fr')
        database_name: Firestore database name (default: "firephrases")

    Returns:
        WiktionaryEntry if found in cache, None otherwise

    Example:
        >>> entry = get_wiktionary_entry("hello", "en")
        >>> if entry and entry.exists:
        ...     print(entry.url)
    """
    client = get_firestore_client(database_name)

    collection_path, document_id = get_firestore_path(token, language_code)
    doc_ref = client.document(collection_path, document_id)
    doc = doc_ref.get()

    if not doc.exists:
        return None

    return WiktionaryEntry.from_firestore_dict(doc.to_dict())


def save_wiktionary_entry(
    entry: WiktionaryEntry,
    database_name: str = "firephrases",
) -> None:
    """Save a wiktionary entry to Firestore cache.

    Args:
        entry: WiktionaryEntry to save
        database_name: Firestore database name (default: "firephrases")

    Example:
        >>> entry = WiktionaryEntry(
        ...     token="hello",
        ...     language_code="en",
        ...     exists=True,
        ...     url="https://en.wiktionary.org/wiki/hello",
        ...     section_anchor="#English"
        ... )
        >>> save_wiktionary_entry(entry)
    """
    client = get_firestore_client(database_name)

    collection_path, document_id = get_firestore_path(entry.token, entry.language_code)
    doc_ref = client.document(collection_path, document_id)
    doc_ref.set(entry.to_firestore_dict())


def batch_get_wiktionary_entries(
    tokens: List[str],
    language_code: str,
    database_name: str = "firephrases",
) -> Dict[str, Optional[WiktionaryEntry]]:
    """Retrieve multiple wiktionary entries in a single batch operation.

    Args:
        tokens: List of words/tokens to look up (case-insensitive)
        language_code: ISO 639-1 language code (e.g., 'en', 'fr')
        database_name: Firestore database name (default: "firephrases")

    Returns:
        Dictionary mapping lowercase tokens to their WiktionaryEntry (or None if not cached)

    Example:
        >>> entries = batch_get_wiktionary_entries(["hello", "world", "foo"], "en")
        >>> for token, entry in entries.items():
        ...     if entry and entry.exists:
        ...         print(f"{token}: {entry.url}")
    """
    client = get_firestore_client(database_name)

    # Build document references for batch get
    doc_refs = []
    token_map = {}  # Map doc_ref to token

    for token in tokens:
        token_lower = token.lower()
        collection_path, document_id = get_firestore_path(token_lower, language_code)
        doc_ref = client.document(collection_path, document_id)
        doc_refs.append(doc_ref)
        token_map[doc_ref.path] = token_lower

    # Firestore limits batch gets to 500 documents
    # If we have more, split into chunks
    MAX_BATCH_SIZE = 500
    results = {}

    for i in range(0, len(doc_refs), MAX_BATCH_SIZE):
        batch_refs = doc_refs[i:i + MAX_BATCH_SIZE]
        docs = client.get_all(batch_refs)

        for doc in docs:
            token_lower = token_map[doc.reference.path]
            if doc.exists:
                results[token_lower] = WiktionaryEntry.from_firestore_dict(doc.to_dict())
            else:
                results[token_lower] = None

    return results


def batch_save_wiktionary_entries(
    entries: List[WiktionaryEntry],
    database_name: str = "firephrases",
) -> None:
    """Save multiple wiktionary entries in a batch operation.

    Args:
        entries: List of WiktionaryEntry objects to save
        database_name: Firestore database name (default: "firephrases")

    Example:
        >>> entries = [
        ...     WiktionaryEntry(token="hello", language_code="en", exists=True, ...),
        ...     WiktionaryEntry(token="world", language_code="en", exists=True, ...)
        ... ]
        >>> batch_save_wiktionary_entries(entries)
    """
    if not entries:
        return

    client = get_firestore_client(database_name)

    # Firestore limits batch writes to 500 operations
    MAX_BATCH_SIZE = 500

    for i in range(0, len(entries), MAX_BATCH_SIZE):
        batch_entries = entries[i:i + MAX_BATCH_SIZE]
        batch = client.batch()

        for entry in batch_entries:
            collection_path, document_id = get_firestore_path(
                entry.token, entry.language_code
            )
            doc_ref = client.document(collection_path, document_id)
            batch.set(doc_ref, entry.to_firestore_dict())

        batch.commit()


def get_or_fetch_wiktionary_entry(
    token: str,
    language_code: str,
    force_refresh: bool = False,
    max_age_days: int = 90,
    database_name: str = "firephrases",
) -> WiktionaryEntry:
    """Get wiktionary entry from cache or fetch from web if not cached.

    This is the main convenience function for getting wiktionary entries.
    It handles caching logic automatically.

    Args:
        token: The word/token to look up
        language_code: ISO 639-1 language code (e.g., 'en', 'fr')
        force_refresh: Force a fresh lookup even if cached (default: False)
        max_age_days: Refresh if cache older than this (default: 90 days)
        database_name: Firestore database name (default: "firephrases")

    Returns:
        WiktionaryEntry (either from cache or freshly fetched)

    Example:
        >>> entry = get_or_fetch_wiktionary_entry("bonjour", "fr")
        >>> html_link = entry.get_html_link("Bonjour!")
    """
    from src.wiktionary.lookup import fetch_wiktionary_entry

    # Check cache first (unless force refresh)
    if not force_refresh:
        cached = get_wiktionary_entry(token, language_code, database_name)
        if cached:
            return cached

    # Fetch from web
    entry = fetch_wiktionary_entry(token, language_code)

    # Save to cache
    save_wiktionary_entry(entry, database_name)

    return entry


def batch_get_or_fetch_wiktionary_entries(
    tokens: List[str],
    language_code: str,
    force_refresh: bool = False,
    max_age_days: int = 90,
    database_name: str = "firephrases",
) -> Dict[str, WiktionaryEntry]:
    """Batch version of get_or_fetch_wiktionary_entry.

    Efficiently retrieves multiple entries, fetching only uncached ones from web.

    Args:
        tokens: List of words/tokens to look up
        language_code: ISO 639-1 language code
        force_refresh: Force fresh lookups even if cached
        max_age_days: Refresh if cache older than this
        database_name: Firestore database name

    Returns:
        Dictionary mapping lowercase tokens to their WiktionaryEntry

    Example:
        >>> tokens = ["bonjour", "merci", "au revoir"]
        >>> entries = batch_get_or_fetch_wiktionary_entries(tokens, "fr")
        >>> for token, entry in entries.items():
        ...     print(entry.get_html_link(token))
    """
    from src.wiktionary.lookup import fetch_wiktionary_entry

    results = {}

    # Get cached entries
    if not force_refresh:
        cached = batch_get_wiktionary_entries(tokens, language_code, database_name)
        for token, entry in cached.items():
            if entry:
                results[token] = entry

    # Identify tokens that need fetching
    tokens_to_fetch = [
        token for token in tokens
        if token.lower() not in results
    ]

    # Fetch missing entries
    newly_fetched = []
    for token in tokens_to_fetch:
        entry = fetch_wiktionary_entry(token, language_code)
        results[token.lower()] = entry
        newly_fetched.append(entry)

    # Save newly fetched entries in batch
    if newly_fetched:
        batch_save_wiktionary_entries(newly_fetched, database_name)

    return results
