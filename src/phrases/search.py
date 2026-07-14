"""Querying and searching phrases in the database."""

from typing import Callable, Dict, List, Optional, Set, Tuple

from phrases.utils import generate_phrase_hash
from phrases.phrase_model import Phrase, Translation
from connections.gcloud_auth import get_firestore_client
from models import get_language
from google.cloud.firestore_v1 import FieldFilter, DocumentReference
from langcodes import Language


def get_phrases_by_collection(
    collection_name: str, deck: str | None = None
) -> List[Phrase]:
    """Retrieve phrases belonging to a specific collection or collections.
        collection_name: Name of the collection to filter phrases
    Returns:
        List of Phrases.

    """

    db = get_firestore_client()
    phrases_ref = db.collection("phrases")

    collection_filter = FieldFilter("collection", "==", collection_name)
    query = phrases_ref.where(filter=collection_filter)
    if deck:
        deck_filter = FieldFilter("deck", "==", deck)
        query = query.where(filter=deck_filter)
    results = query.get()

    phrases = []
    for doc in results:
        phrase_data = doc.to_dict()
        phrase = Phrase.model_validate(phrase_data)
        phrase.translations = _get_translations(doc.reference)
        phrases.append(phrase)

    return phrases


def get_verbs_from_phrases(phrases: List[Phrase]) -> List[str]:
    """Get verbs covered by a list of phrases"""

    covered_verbs = set()
    for p in phrases:
        covered_verbs = covered_verbs | set(p.verbs)

    return sorted(list(covered_verbs))


def get_vocab_from_phrases(phrases: List[Phrase]) -> List[str]:
    """Get verbs covered by a list of phrases"""

    covered_vocab = set()
    for p in phrases:
        covered_vocab = covered_vocab | set(p.vocab)

    return sorted(list(covered_vocab))


def _get_translations(document_ref: DocumentReference) -> dict[str, Translation]:
    """Helper function to get translations for a phrase document.

    Args:
        document_ref: Firestore DocumentReference for the phrase.

    Returns: List of translations."""

    TRANSLATIONS = {}
    translation_stream = document_ref.collection("translations").stream()
    for translation_doc in translation_stream:
        translation = Translation.model_validate(translation_doc.to_dict())
        TRANSLATIONS[translation.language.to_tag()] = translation
    return TRANSLATIONS


def _get_phrase_coverage(
    phrase: Phrase, target_verbs: Set[str], target_vocab: Set[str]
) -> float:
    """Return the fraction of a phrase's lemmas that appear in the target sets.

    Returns 0.0 if the phrase has no verbs or vocab (rather than raising),
    so the greedy loop can safely skip it.
    """
    phrase_verbs = set(phrase.verbs)
    phrase_vocab = set(phrase.vocab)
    num_phrase_items = len(phrase_verbs) + len(phrase_vocab)
    if num_phrase_items == 0:
        return 0.0
    matching = len(phrase_verbs & target_verbs) + len(phrase_vocab & target_vocab)
    return matching / num_phrase_items


def _translation_coverage(
    translation: Translation,
    remaining_verbs: Set[str],
    remaining_vocab: Set[str],
) -> int:
    """Return the count of uncovered target items that *translation* covers.

    Works on a Translation object so the greedy algorithm can operate on
    target-language verbs/vocab rather than the English phrase fields.
    """
    tv = set(translation.verbs) if translation.verbs else set()
    tw = set(translation.vocab) if translation.vocab else set()
    return len(tv & remaining_verbs) + len(tw & remaining_vocab)


def _to_lower(tokens: Set[str]) -> Set[str]:
    """Convert a list of tokens to lowercase."""
    return set([t.lower() for t in tokens])


def _greedy_set_cover(
    candidates: List[Phrase],
    target_verbs: Set[str],
    target_vocab: Set[str],
    get_verbs_fn: Callable[[Phrase], List[str]],
    get_vocab_fn: Callable[[Phrase], List[str]],
) -> List[Phrase]:
    """Greedy set-cover: iteratively pick the phrase that covers the most
    remaining target items until all are covered or no progress is possible.

    Args:
        candidates:   Pool of phrases to choose from (not mutated).
        target_verbs: Verb lemmas that must be covered.
        target_vocab: Vocab lemmas that must be covered.
        get_verbs_fn: Callable that extracts the verb list from a Phrase.
        get_vocab_fn: Callable that extracts the vocab list from a Phrase.

    Returns:
        Minimum (greedy-approximate) list of Phrases providing coverage.
    """
    if not (target_verbs or target_vocab):
        return []

    remaining_verbs = _to_lower(set(target_verbs))
    remaining_vocab = _to_lower(set(target_vocab))
    pool = list(candidates)  # never mutate the caller's list
    selected: List[Phrase] = []

    while remaining_verbs or remaining_vocab:
        best_phrase: Optional[Phrase] = None
        best_count = 0

        for phrase in pool:
            pv = _to_lower(set(get_verbs_fn(phrase)))
            pw = _to_lower(set(get_vocab_fn(phrase)))
            count = len(pv & remaining_verbs) + len(pw & remaining_vocab)
            if count > best_count:
                best_phrase = phrase
                best_count = count  # ← was missing in the old implementation

        if best_phrase is None or best_count == 0:
            print(
                f"Warning: cannot achieve full coverage. "
                f"Missing verbs: {remaining_verbs}, vocab: {remaining_vocab}"
            )
            break

        selected.append(best_phrase)
        pool.remove(best_phrase)
        # Must lowercase before subtracting: remaining sets are already lowercased
        # but tokens/verbs from phrases may have mixed case (e.g. "Ska" at sentence start).
        remaining_verbs -= _to_lower(set(get_verbs_fn(best_phrase)))
        remaining_vocab -= _to_lower(set(get_vocab_fn(best_phrase)))

    return selected


def find_minimum_coverage_phrases(
    phrases: List[Phrase],
    target_verbs: Set[str],
    target_vocab: Set[str],
) -> List[Phrase]:
    """Find the minimum set of phrases that cover all target verbs and vocabulary.

    Uses a greedy set cover algorithm that iteratively selects the phrase that
    covers the most uncovered items.  Delegates to ``_greedy_set_cover``.

    Args:
        phrases:      List of phrases to search through (not mutated).
        target_verbs: Set of English verb lemmas to cover.
        target_vocab: Set of English vocab lemmas to cover.

    Returns:
        List of phrases providing coverage (may be incomplete if the pool
        cannot cover all requested items).
    """
    return _greedy_set_cover(
        candidates=phrases,
        target_verbs=target_verbs,
        target_vocab=target_vocab,
        get_verbs_fn=lambda p: p.verbs,
        get_vocab_fn=lambda p: p.vocab,
    )


def find_phrases_by_token_coverage(
    phrases: List[Phrase],
    target_tokens: Set[str],
    language: Language | str,
    min_coverage_ratio: float = 0.5,
) -> List[Tuple[Phrase, float, Set[str]]]:
    """Find phrases containing target tokens in a specific language.

    Returns phrases ranked by coverage ratio (how many target tokens they contain).
    Useful for finding phrases that teach specific vocabulary in the target language.

    Args:
        phrases: List of phrases to search through
        target_tokens: Set of tokens to search for (e.g., {'ouvrir', 'porte', 'plaît'})
        language: BCP47 language tag to search in (e.g., 'fr-FR', 'ja-JP')
        min_coverage_ratio: Minimum ratio of tokens that must be covered (0.0 to 1.0).
                           Default 0.5 means phrase must contain at least 50% of target tokens.

    Returns:
        List of tuples: (phrase, coverage_ratio, matched_tokens)
        Sorted by coverage ratio (highest first), then by number of matched tokens.

    Example:
        >>> phrases = get_phrases_by_collection("WarmUp150")
        >>> # Find French phrases with these tokens
        >>> tokens = {'ouvrir', 'porte', 'plaît'}
        >>> results = find_phrases_by_token_coverage(phrases, tokens, 'fr-FR')
        >>> for phrase, ratio, matched in results[:5]:
        >>>     print(f"{ratio:.0%} coverage: {phrase.english} -> {matched}")
    """
    # Normalize language to BCP47Language
    if isinstance(language, str):
        language = get_language(language)

    language_tag = language.to_tag()

    # Results: (phrase, coverage_ratio, matched_tokens)
    results: List[Tuple[Phrase, float, Set[str]]] = []

    for phrase in phrases:
        # Check if phrase has translation for this language
        if language_tag not in phrase.translations:
            continue

        translation = phrase.translations[language_tag]

        # Find intersection of phrase tokens and target tokens
        # Convert to lowercase for case-insensitive matching
        phrase_tokens = {token.lower() for token in translation.tokens}
        target_tokens_lower = {token.lower() for token in target_tokens}
        matched_tokens = phrase_tokens & target_tokens_lower

        if not matched_tokens:
            continue

        # Calculate coverage ratio
        coverage_ratio = len(matched_tokens) / len(target_tokens_lower)

        # Filter by minimum coverage ratio
        if coverage_ratio >= min_coverage_ratio:
            results.append((phrase, coverage_ratio, matched_tokens))

    # Sort by coverage ratio (descending), then by number of matched tokens
    results.sort(key=lambda x: (x[1], len(x[2])), reverse=True)

    return results


# ---------------------------------------------------------------------------
# Translation-level vocab coverage (target-language aware)
# ---------------------------------------------------------------------------


def _load_phrases_with_translation(
    language_tag: str,
    collection: Optional[str] = None,
    deck: Optional[str] = None,
    database_name: str = "firephrases",
) -> List[Phrase]:
    """Load phrases that have a populated target-language translation.

    Streams the ``phrases`` collection (optionally filtered by collection/deck),
    then for each phrase does a **single direct get()** on
    ``translations/{language_tag}`` — avoiding loading all other language
    translations.  Only phrases whose translation has at least one verb or
    vocab entry are returned.

    Args:
        language_tag:    Exact Firestore document key, e.g. ``'sv-SE'``.
        collection:      Optional collection filter (e.g. ``'WarmUp150'``).
        deck:            Optional deck filter (e.g. ``'Pack01'``).
        database_name:   Firestore database name.

    Returns:
        List of Phrase objects, each with only the target translation populated.
    """
    db = get_firestore_client(database_name)
    query = db.collection("phrases")
    if collection:
        query = query.where(filter=FieldFilter("collection", "==", collection))
    if deck:
        query = query.where(filter=FieldFilter("deck", "==", deck))

    phrases: List[Phrase] = []
    for phrase_doc in query.stream():
        phrase_data = phrase_doc.to_dict()

        # Skip story phrases (phrases that have a story_title_hash)
        if phrase_data.get("story_title_hash"):
            continue

        t_doc = (
            phrase_doc.reference.collection("translations").document(language_tag).get()
        )
        if not t_doc.exists:
            continue

        t_data = t_doc.to_dict()
        translation = Translation.model_validate(t_data)

        # Only include phrases whose translation has usable NLP data
        if not translation.verbs and not translation.vocab:
            continue

        phrase = Phrase.model_validate(phrase_data)
        phrase.translations = {language_tag: translation}
        phrases.append(phrase)

    return phrases


def find_phrases_by_vocab_dict(
    vocab_dict: Dict[str, List[str]],
    language: Language | str,
    collection: Optional[str] = None,
    deck: Optional[str] = None,
    database_name: str = "firephrases",
) -> Tuple[List[Phrase], Dict[str, List[str]]]:
    """Find the minimum set of phrases whose target-language translation covers
    every verb and vocab word in *vocab_dict*.

    Args:
        vocab_dict:    Dict with keys ``'verbs'`` and/or ``'vocab'``, each a
                       list of lemma strings in the target language.
        language:      Target language — ``Language`` object or BCP-47 string
                       (e.g. ``'sv-SE'``).
        collection:    Optional — restrict search to this Firestore collection.
        deck:          Optional — restrict search to this deck.
        database_name: Firestore database (default: ``'firephrases'``).

    Returns:
        A tuple of:
        - ``phrases``: minimum (greedy-approximate) list of ``Phrase`` objects
          providing coverage.  Each phrase has only the target translation
          populated.
        - ``missing``: dict with keys ``'verbs'`` and ``'vocab'`` listing any
          words from *vocab_dict* that no phrase in the pool could cover
          (sorted alphabetically).  Both lists are empty on full coverage.

    Example:
        >>> phrases, missing = find_phrases_by_vocab_dict(
        ...     vocab_dict={"verbs": ["springa", "äta"], "vocab": ["hund", "bil"]},
        ...     language="sv-SE",
        ...     collection="WarmUp150",
        ... )
        >>> print(f"{len(phrases)} phrases selected, missing: {missing}")
        >>> for p in phrases:
        ...     print(p.english, p.translations["sv-SE"].text)
    """
    lang = get_language(language)
    language_tag = lang.to_tag()

    target_verbs: Set[str] = set(vocab_dict.get("verbs", []))
    target_vocab: Set[str] = set(vocab_dict.get("vocab", []))

    if not target_verbs and not target_vocab:
        return [], {"verbs": [], "vocab": []}

    candidates = _load_phrases_with_translation(
        language_tag=language_tag,
        collection=collection,
        deck=deck,
        database_name=database_name,
    )

    # we add tokens and vocab to the vocab checker to minimise too many returns due to lemma issues.
    selected = _greedy_set_cover(
        candidates=candidates,
        target_verbs=target_verbs,
        target_vocab=target_vocab,
        get_verbs_fn=lambda p: (
            p.translations[language_tag].verbs + p.translations[language_tag].tokens
            or []
        ),
        get_vocab_fn=lambda p: (
            p.translations[language_tag].vocab + p.translations[language_tag].tokens
            or []
        ),
    )

    # Compute what the selected phrases collectively cover.
    # Use same logic as the greedy algo (tokens supplement lemmas) and lowercase
    # to match case-insensitive coverage detection.
    covered_verbs: Set[str] = set()
    covered_vocab: Set[str] = set()
    for p in selected:
        t = p.translations[language_tag]
        covered_verbs |= _to_lower(set((t.verbs or []) + (t.tokens or [])))
        covered_vocab |= _to_lower(set((t.vocab or []) + (t.tokens or [])))

    missing: Dict[str, List[str]] = {
        "verbs": sorted(v for v in target_verbs if v.lower() not in covered_verbs),
        "vocab": sorted(v for v in target_vocab if v.lower() not in covered_vocab),
    }

    return selected, missing


def get_all_phrases(
    database_name: str = "firephrases",
    limit: Optional[int] = None,
) -> List[Phrase]:
    """Retrieve all phrases from the database.

    Args:
        database_name: Name of the Firestore database (default: "firephrases")
        limit: Maximum number of phrases to retrieve (default: None = all phrases)

    Returns:
        List of all Phrases with their translations

    Example:
        >>> all_phrases = get_all_phrases()
        >>> print(f"Total phrases in database: {len(all_phrases)}")
    """
    db = get_firestore_client(database_name)
    phrases_ref = db.collection("phrases")

    # Apply limit if specified
    if limit:
        query = phrases_ref.limit(limit)
    else:
        query = phrases_ref

    results = query.stream()

    phrases = []
    for doc in results:
        phrase_data = doc.to_dict()
        phrase = Phrase.model_validate(phrase_data)
        phrase.translations = _get_translations(doc.reference)
        phrases.append(phrase)

    return phrases


def get_phrase(
    phrase_hash: str, database_name: str = "firephrases"
) -> Optional[Phrase]:
    """Fetch a phrase from Firestore by its hash, including all translations.

    Fetches the phrase document from `phrases/{phrase_hash}` and all translations from
    the `phrases/{phrase_hash}/translations` subcollection.

    Args:
        phrase_hash: The phrase hash (document ID)
        database_name: Name of the Firestore database (default: "firephrases")

    Returns:
        Optional[Phrase]: The phrase object with translations if found, None otherwise

    Raises:
        RuntimeError: If Firestore query fails
    """
    client = get_firestore_client(database_name)
    doc_ref = client.collection("phrases").document(phrase_hash)
    doc = doc_ref.get()

    if not doc.exists:
        raise ValueError(f"Phrase with hash {phrase_hash} not found in Firestore")

    # Get phrase data
    phrase_data = doc.to_dict()

    core_phrase_references = {
        "key": phrase_hash,
        "firestore_document_ref": doc_ref,
    }

    # Fetch all translations from subcollection
    translations_docs = doc_ref.collection("translations").stream()
    translations = {}
    for translated_doc in translations_docs:
        translation_data = translated_doc.to_dict()
        translation = Translation.model_validate(translation_data)
        # Use language tag as the key
        translations[translation.language.to_tag()] = translation

    # Add translations to phrase data
    phrase_data["translations"] = translations
    phrase_data.update(core_phrase_references)
    return Phrase.model_validate(phrase_data)


def get_phrase_by_english(
    english_phrase: str, database_name: str = "firephrases"
) -> Optional[Phrase]:
    """Fetch a phrase from Firestore using its English text.

    Convenience wrapper that generates the phrase hash from English text and fetches
    the corresponding phrase document with all translations.

    Args:
        english_phrase: The English phrase text (e.g., "She runs to the store daily")
        database_name: Name of the Firestore database (default: "firephrases")

    Returns:
        Optional[Phrase]: The phrase object with translations if found, None otherwise

    Raises:
        RuntimeError: If Firestore query fails
    """
    phrase_hash = generate_phrase_hash(english_phrase)
    return get_phrase(phrase_hash, database_name=database_name)


def add_tags_to_translations(
    phrases: List[Phrase],
    language: Language | str,
    tags: List[str] | str,
) -> List[DocumentReference]:
    """Add Anki tags to the target language translation of a list of phrases.

    Updates both the local Phrase objects and their corresponding Firestore
    Translation documents via a targeted ``.update()``. This is efficient as
    it only updates the tags field.

    Args:
        phrases: List of Phrase objects to update.
        language: Target language (e.g. 'sv-SE' or a Language object).
        tags: Single tag string or list of tags to add.

    Returns:
        List of updated Firestore DocumentReferences.

    Example:
        >>> phrases, missing = find_phrases_by_vocab_dict(vocab, "sv-SE")
        >>> updated_refs = add_tags_to_translations(phrases, "sv-SE", "media::film::bron")
    """
    lang = get_language(language)
    language_tag = lang.to_tag()

    updated_refs: List[DocumentReference] = []

    for phrase in phrases:
        if language_tag not in phrase.translations:
            continue

        translation = phrase.translations[language_tag]

        # Merge new tags while avoiding duplicates and preserving order
        new_tags = translation.add_tags(tags)

        if new_tags:
            # Get the DocumentReference (using the private helper if missing)
            doc_ref = translation.firestore_document_ref
            if not doc_ref:
                doc_ref = translation._get_firestore_document_reference()

            # Targeted update to avoid overwriting other fields unnecessarily
            doc_ref.update({"tags": translation.tags})
            updated_refs.append(doc_ref)

    return updated_refs


def find_phrases_by_tag(
    tag: str,
    language: Language | str,
    collection: Optional[str] = None,
    deck: Optional[str] = None,
    database_name: str = "firephrases",
) -> List[Phrase]:
    """Find all phrases whose target-language translation contains a specific tag.

    Uses a fast Firestore collection group query to directly find translation
    documents containing the tag, then fetches their parent phrases.
    Returned Phrase objects have all translations fully populated (which is
    required for subsequent Anki deck generation).

    Args:
        tag: The tag to search for (e.g., 'media::film::bron').
        language: The target language (e.g., 'sv-SE').
        collection: Optional collection filter.
        deck: Optional deck filter.
        database_name: Firestore database name.

    Returns:
        List of Phrase objects that have the tag.
    """
    lang = get_language(language)
    language_tag = lang.to_tag()

    db = get_firestore_client(database_name)

    # Fast collection group query on the 'tags' array
    translations_query = db.collection_group("translations").where(
        filter=FieldFilter("tags", "array_contains", tag)
    )

    matched_phrases: List[Phrase] = []

    for t_doc in translations_query.stream():
        # Ensure we only process translations for the requested language
        if t_doc.id != language_tag:
            continue

        # The parent of a translation doc is the 'translations' subcollection,
        # and its parent is the phrase document
        phrase_ref = t_doc.reference.parent.parent
        if not phrase_ref:
            continue

        phrase_doc = phrase_ref.get()
        if not phrase_doc.exists:
            continue

        phrase_data = phrase_doc.to_dict()

        # Apply the optional collection/deck filters in Python
        # (Faster than trying to do composite queries across parent/child)
        if collection and phrase_data.get("collection") != collection:
            continue
        if deck and phrase_data.get("deck") != deck:
            continue

        phrase = Phrase.model_validate(phrase_data)
        # Populate all translations so it can be passed directly to create_anki_deck
        phrase.translations = _get_translations(phrase_ref)
        matched_phrases.append(phrase)

    return matched_phrases
