import hashlib
from typing import List, Optional


def normalize_tags(tags: Optional[str | List[str]]) -> List[str]:
    """Normalize Anki tags input into a unique, order-preserving list.

    Accepts a single tag string, a list of tags, or None. Falsy tags (e.g.
    empty strings) are dropped, and duplicates are removed while keeping
    first-seen order (plain list(set(...)) would lose that order).

    Args:
        tags: Single tag string, list of tags, or None.

    Returns:
        List[str]: Unique tags in first-seen order.
    """
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = [tags]

    seen: set[str] = set()
    unique_tags: List[str] = []
    for tag in tags:
        if tag and tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    return unique_tags


def generate_phrase_hash(english_text: str) -> str:
    """Generate a unique hash for this phrase based on English text.

    Returns:
        str: Phrase hash in format: {slug}_{hash_suffix}

    Example:
        >>> phrase = Phrase(english="She runs to the store daily", ...)
        >>> phrase.generate_phrase_hash()
        'she_runs_to_the_store_daily_a3f8d2'
    """
    # Hash the ORIGINAL phrase to capture punctuation differences
    hash_suffix = hashlib.sha256(english_text.encode()).hexdigest()[:6]

    # Create URL-safe slug from lowercase version
    normalized = english_text.lower().strip()
    # Keep only alphanumeric and convert spaces to underscores
    slug = "".join(c if c.isalnum() or c == " " else "" for c in normalized)
    slug = slug.replace(" ", "_")[:50]

    return f"{slug}_{hash_suffix}"


def generate_deck_name(collection: str, deck: str) -> str:
    return f"{collection}-{deck}"


def generate_note_guid(phrase_key: str, source_tag: str, target_tag: str) -> int:
    """Generate a deterministic Anki note guid for a phrase/language pair.

    Deterministic replacement for the note-guid formula that used to live
    inline in anki_tools.py as `_string_to_large_int(f"{phrase.key}_...")`,
    which relied on Python's randomized-per-process builtin hash() and so
    produced a different guid every run. Takes the three components
    separately (rather than a pre-joined string) so every call site formats
    them identically by construction.

    Args:
        phrase_key: The phrase's Firestore key (see generate_phrase_hash).
        source_tag: BCP-47 tag of the source language (e.g. 'en-GB').
        target_tag: BCP-47 tag of the target language (e.g. 'sv-SE').

    Returns:
        int: Deterministic guid, positive and less than 10**10 (matches the
            range of the formula it replaces).

    Example:
        >>> generate_note_guid("hello_world_abc123", "en-GB", "fr-FR")
        4837291056
    """
    text = f"{phrase_key}_{source_tag}_{target_tag}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    return int(digest[:10], 16) % (10**10)


ANKI_MANAGED_TAG_PREFIX = "fs::"


def to_anki_tags(firestore_tags: List[str]) -> List[str]:
    """Convert Firestore tag values into the fs::-prefixed form written to Anki.

    Firestore's own stored tag values are left unprefixed; the prefix is
    applied only at the Anki-write boundary so an Anki sync can safely
    overwrite exactly the tags it manages (anything starting with
    ANKI_MANAGED_TAG_PREFIX) while leaving Anki-native tags (e.g. 'marked',
    'leech') or any hand-added tag completely untouched.

    Args:
        firestore_tags: Tag values as stored in Translation.tags.

    Returns:
        List[str]: Same tags, each prefixed with ANKI_MANAGED_TAG_PREFIX.

    Example:
        >>> to_anki_tags(["SURVIVAL", "Pack01"])
        ['fs::SURVIVAL', 'fs::Pack01']
    """
    return [f"{ANKI_MANAGED_TAG_PREFIX}{tag}" for tag in firestore_tags]
