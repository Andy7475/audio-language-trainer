"""Querying and searching phrases in the database."""

from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

from phrases.utils import generate_phrase_hash, normalize_tags
from phrases.phrase_model import Phrase, Translation
from connections.gcloud_auth import get_firestore_client
from models import get_language
from google.cloud.firestore_v1 import FieldFilter, DocumentReference
from langcodes import Language
from nlp import extract_token_lemma_pos, VERB_POS_TAGS, VOCAB_POS_TAGS
from wiktionary.lookup import word_in_wiktionary


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

    return _cover_vocab_dict(candidates, target_verbs, target_vocab, language_tag)


def _cover_vocab_dict(
    candidates: List[Phrase],
    target_verbs: Set[str],
    target_vocab: Set[str],
    language_tag: str,
) -> Tuple[List[Phrase], Dict[str, List[str]]]:
    """Greedy-cover target_verbs/target_vocab using an already-loaded candidate pool.

    Extracted from ``find_phrases_by_vocab_dict`` so callers that already have
    a loaded candidate pool (e.g. ``add_tags_from_text``) can reuse the same
    coverage + missing-word computation without a second Firestore fetch.

    Args:
        candidates:   Phrases with the target-language translation populated
                      (see ``_load_phrases_with_translation``).
        target_verbs: Verb lemmas/tokens to cover.
        target_vocab: Vocab lemmas/tokens to cover.
        language_tag: BCP-47 tag used to key into each phrase's translations.

    Returns:
        Same shape as ``find_phrases_by_vocab_dict``: (selected phrases, missing dict).
    """
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


def _passes_wiktionary_check(
    lemma: str, bucket: Literal["verbs", "vocab"], language_code: str
) -> bool:
    """Check whether a lemma is a real dictionary word, to filter out NLP
    artefacts and disfluencies (e.g. 'uh', '-flytta') before they pollute a
    vocab_dict.

    Mirrors ``nlp.get_verbs_and_vocab``'s ``filter_by_wiktionary`` logic: the
    verb bucket is checked against pos='verb'; the vocab bucket is accepted if
    it matches noun, adj, or adv.

    Args:
        lemma: Dictionary-form word to check.
        bucket: Which POS bucket this word was classified into.
        language_code: Two-letter language code (e.g. 'sv').
    """
    if bucket == "verbs":
        return word_in_wiktionary(lemma, language_code, pos="verb")
    return (
        word_in_wiktionary(lemma, language_code, pos="noun")
        or word_in_wiktionary(lemma, language_code, pos="adj")
        or word_in_wiktionary(lemma, language_code, pos="adv")
    )


def _extract_text_vocab_dict(
    text: str,
    language: Language,
    candidate_tokens: Set[str],
) -> Tuple[Dict[str, List[str]], List[str]]:
    """Turn story text into a vocab_dict of coverage targets, plus ignored tokens.

    Extracts (token, lemma, pos) triples for *text*, drops anything that isn't
    a verb/noun/adj/adv or has no Wiktionary entry (disfluencies, acronyms,
    NLP artefacts), and for each surviving word prefers the native inflected
    token as the match target if some candidate phrase already contains it
    verbatim — falling back to the lemma otherwise. This lets coverage
    matching reinforce the exact form seen in the text when possible.

    Args:
        text: Source text (e.g. a story) to extract vocabulary from.
        language: Target language of *text*.
        candidate_tokens: Lowercased tokens already present across the
            candidate phrase pool (see ``_load_phrases_with_translation``),
            used to decide token-vs-lemma preference.

    Returns:
        Tuple of (vocab_dict with 'verbs'/'vocab' keys, sorted unique ignored tokens).
    """
    language_code = language.language
    assert language_code is not None, f"Language has no language subtag: {language!r}"

    triples = extract_token_lemma_pos(text, language_code)

    target_verbs: Set[str] = set()
    target_vocab: Set[str] = set()
    ignored_tokens: Set[str] = set()

    for token_text, lemma, pos in triples:
        if pos in VERB_POS_TAGS:
            bucket: Literal["verbs", "vocab"] = "verbs"
        elif pos in VOCAB_POS_TAGS:
            bucket = "vocab"
        else:
            ignored_tokens.add(token_text.lower())
            continue

        if not lemma or not _passes_wiktionary_check(lemma, bucket, language_code):
            ignored_tokens.add(token_text.lower())
            continue

        target = token_text.lower() if token_text.lower() in candidate_tokens else lemma
        (target_verbs if bucket == "verbs" else target_vocab).add(target)

    vocab_dict = {"verbs": sorted(target_verbs), "vocab": sorted(target_vocab)}
    return vocab_dict, sorted(ignored_tokens)


def add_tags_from_text(
    text: str,
    language: Language | str,
    tags: List[str] | str,
    collection: Optional[str] = None,
    deck: Optional[str] = None,
    database_name: str = "firephrases",
) -> Tuple[List[Phrase], Dict[str, List[str]]]:
    """Tag the minimum set of existing phrases needed to understand *text*.

    Extracts the vocabulary from *text* (e.g. a story or subtitle transcript),
    finds the minimum (greedy-approximate) set of existing phrases whose
    target-language translation already covers it, and adds *tags* to those
    phrases' Translation documents in Firestore — so they can later be pulled
    into a themed Anki deck via ``find_phrases_by_tag``.

    Words with no covering phrase are returned as a ``missing`` vocab_dict
    (same shape as ``find_phrases_by_vocab_dict``'s ``missing``) so new
    phrases can be generated for them separately, e.g. via
    ``generate_phrases_from_vocab_dict``.

    Args:
        text: Source text in the target language.
        language: Target language of *text* (e.g. 'sv-SE' or a Language object).
        tags: Single tag string or list of tags to add (e.g. 'media::film::bron').
        collection: Optional — restrict the candidate phrase pool to this collection.
        deck: Optional — restrict the candidate phrase pool to this deck.
        database_name: Firestore database (default: 'firephrases').

    Returns:
        Tuple of (tagged phrases, missing vocab_dict).

    Example:
        >>> phrases, missing = add_tags_from_text(
        ...     "Han sprang till affären och köpte ett äpple.",
        ...     language="sv-SE",
        ...     tags="media::film::bron",
        ... )
        >>> print(f"{len(phrases)} phrases tagged, missing: {missing}")
    """
    lang = get_language(language)
    language_tag = lang.to_tag()

    candidates = _load_phrases_with_translation(
        language_tag=language_tag,
        collection=collection,
        deck=deck,
        database_name=database_name,
    )

    candidate_tokens: Set[str] = set()
    for phrase in candidates:
        candidate_tokens |= _to_lower(set(phrase.translations[language_tag].tokens))

    vocab_dict, ignored_tokens = _extract_text_vocab_dict(text, lang, candidate_tokens)
    target_verbs = set(vocab_dict["verbs"])
    target_vocab = set(vocab_dict["vocab"])

    selected, missing = _cover_vocab_dict(candidates, target_verbs, target_vocab, language_tag)

    if selected:
        add_tags_to_translations(selected, lang, tags)

    total_targets = len(target_verbs) + len(target_vocab)
    total_missing = len(missing["verbs"]) + len(missing["vocab"])
    covered_count = total_targets - total_missing

    print(
        f"add_tags_from_text: {covered_count}/{total_targets} words covered by "
        f"{len(selected)} phrase(s); {len(ignored_tokens)} tokens ignored "
        f"(not a content word or no Wiktionary entry); {total_missing} words missing "
        f"(no matching phrase found)"
    )
    if ignored_tokens:
        print(f"  Ignored tokens: {ignored_tokens}")
    if total_missing:
        print(f"  Missing verbs: {missing['verbs']}")
        print(f"  Missing vocab: {missing['vocab']}")

    return selected, missing


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


def delete_tag_from_firestore(
    tag: List[str] | str,
    language: Language | str | None = None,
    database_name: str = "firephrases",
) -> List[DocumentReference]:
    """Remove tag(s) from Translation documents in Firestore.

    Tags are a property of a Translation, not a Phrase, so this operates
    directly on translation documents via the same fast collection-group
    query pattern as ``find_phrases_by_tag`` — no need to load full Phrase
    objects. Useful for cleaning up tags added by mistake (e.g. while
    experimenting with ``add_tags_from_text``/``add_tags_to_translations``).

    Args:
        tag: Single tag string or list of tags to remove.
        language: If given, only remove from this language's translations
            (e.g. 'sv-SE'). If None (default), removes from every language's
            translations that have it.
        database_name: Firestore database name.

    Returns:
        List of updated Firestore DocumentReferences (translations that
        actually had one of the tags removed).

    Example:
        >>> delete_tag_from_firestore("media::test::story")  # every language
        >>> delete_tag_from_firestore(["media::test::story"], language="sv-SE")
    """
    tags_to_remove = set(normalize_tags(tag))
    if not tags_to_remove:
        return []

    language_tag = get_language(language).to_tag() if language is not None else None

    db = get_firestore_client(database_name)

    # Query once per tag (Firestore only supports one array_contains clause
    # per query) and dedupe by path, since a translation may carry more than
    # one of the tags being removed.
    docs_by_path: Dict[str, Tuple[Any, List[str]]] = {}
    for t in tags_to_remove:
        translations_query = db.collection_group("translations").where(
            filter=FieldFilter("tags", "array_contains", t)
        )
        for t_doc in translations_query.stream():
            if language_tag is not None and t_doc.id != language_tag:
                continue
            docs_by_path[t_doc.reference.path] = (
                t_doc.reference,
                (t_doc.to_dict() or {}).get("tags", []),
            )

    updated_refs: List[DocumentReference] = []
    for doc_ref, existing_tags in docs_by_path.values():
        remaining_tags = [t for t in existing_tags if t not in tags_to_remove]
        if len(remaining_tags) != len(existing_tags):
            doc_ref.update({"tags": remaining_tags})
            updated_refs.append(doc_ref)

    return updated_refs
