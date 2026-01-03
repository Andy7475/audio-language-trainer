"""Querying and searching phrases in the database."""

from typing import List, Optional, Set, Dict, Tuple

from src.phrases.phrase_model import Phrase, Translation
from src.connections.gcloud_auth import get_firestore_client
from src.models import BCP47Language, get_language
from google.cloud.firestore_v1 import FieldFilter, DocumentReference

def get_phrases_by_collection(collection_name: str) -> List[Phrase]:
    """Retrieve phrases belonging to a specific collection or collections.
        collection_name: Name of the collection to filter phrases   
    Returns:
        List of Phrases.

    """

    db = get_firestore_client()
    phrases_ref = db.collection("phrases")

    results: Set[DocumentReference] = set()

    collection_filter = FieldFilter("collections", "equal_to", collection_name)
    query = phrases_ref.where(filter=collection_filter)
    results = query.get()

    phrases = []
    for doc in results:
        phrase_data = doc.to_dict()
        phrase = Phrase.model_validate(phrase_data)
        phrase.translations = _get_translations(doc.reference)
        phrases.append(phrase)

    return phrases

def _get_translations(document_ref:DocumentReference) -> dict[str, Translation]:
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


def _get_phrase_coverage(phrase: Phrase, target_verbs: Set[str], target_vocab: Set[str]) -> float:
    """Helper function to get which target items a phrase covers.

    Args:
        phrase: Phrase object
        target_verbs: Set of verb lemmas to cover
        target_vocab: Set of vocabulary lemmas to cover

    Returns:
        percentage coverage of target items covered by the phrase
    """

    phrase_verbs = set(phrase.verbs)
    phrase_vocab = set(phrase.vocab)
    num_phrase_items = len(phrase_verbs) + len(phrase_vocab)
    if num_phrase_items == 0:
        raise ValueError("Phrase has no verbs or vocabulary items to cover.")
    matching_verbs = phrase_verbs & target_verbs
    matching_vocab = phrase_vocab & target_vocab

    num_matching_items = len(matching_verbs) + len(matching_vocab)
    return num_matching_items / num_phrase_items
    phrase_items = set(phrase.verbs) | set(phrase.vocab)
    target_items = target_verbs | target_vocab
    covered_items = phrase_items & target_items
    return covered_items

def find_minimum_coverage_phrases(
    phrases: List[Phrase],
    target_verbs: Set[str],
    target_vocab: Set[str],
) -> List[Phrase]:
    """Find the minimum set of phrases that cover all target verbs and vocabulary.

    Uses a greedy set cover algorithm that iteratively selects the phrase that covers
    the most uncovered items. This provides a good approximation (log n factor) of the
    optimal solution in polynomial time.

    Algorithm complexity: O(n * m) where n = number of phrases, m = size of target set
    For 5000 phrases and typical vocab sizes, this runs very quickly.

    Args:
        phrases: List of phrases to search through
        target_verbs: Set of verb lemmas to cover (e.g., {'run', 'eat', 'sleep'})
        target_vocab: Set of vocabulary lemmas to cover (e.g., {'store', 'daily', 'apple'})

    Returns:
        List of phrases that provide complete coverage of verbs and vocab.
        Returns empty list if complete coverage is impossible.

    Example:
        >>> phrases = get_phrases_by_collection("WarmUp150")
        >>> verbs = {'run', 'eat', 'sleep'}
        >>> vocab = {'store', 'daily', 'apple'}
        >>> selected = find_minimum_coverage_phrases(phrases, verbs, vocab)
        >>> print(f"Coverage achieved with {len(selected)} phrases")
    """
    if len(target_verbs | target_vocab) == 0:
        return []

    remaining_verbs = target_verbs.copy()
    remaining_vocab = target_vocab.copy()

    def _update_verbs(phrase: Phrase, remaining_verbs: Set[str]) -> Set[str]:
        phrase_verbs = set(phrase.verbs)
        return remaining_verbs - phrase_verbs
    
    def _update_vocab(phrase: Phrase, remaining_vocab: Set[str]) -> Set[str]:
        phrase_vocab = set(phrase.vocab)
        return remaining_vocab - phrase_vocab
    # Track selected phrases
    selected_phrases: List[Phrase] = []

    while (remaining_verbs or remaining_vocab):
        best_phrase = None
        best_coverage = 0.0

        for phrase in phrases:
            phrase_coverage = _get_phrase_coverage(phrase, remaining_verbs, remaining_vocab)
            if phrase_coverage == 1:
                best_phrase = phrase
                break
            if phrase_coverage > best_coverage:
                best_phrase = phrase

        if not best_phrase:
            print(f"Warning: Cannot achieve complete coverage. Missing verbs: {remaining_verbs}, vocab: {remaining_vocab}")
            break

        selected_phrases.append(best_phrase)
        remaining_verbs = _update_verbs(best_phrase, remaining_verbs)
        remaining_vocab = _update_vocab(best_phrase, remaining_vocab)
        phrases.remove(best_phrase)

    return selected_phrases


def find_phrases_by_token_coverage(
    phrases: List[Phrase],
    target_tokens: Set[str],
    language: BCP47Language | str,
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