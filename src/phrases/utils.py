import hashlib


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

def generate_deck_name(collection:str, deck:str)->str:
    return f"{collection}-{deck}"