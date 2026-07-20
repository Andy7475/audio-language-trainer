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
