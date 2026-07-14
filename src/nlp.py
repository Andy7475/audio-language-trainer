from typing import Dict, List, Optional, Tuple, Union

from langcodes import Language
from models import get_language
from connections.gcloud_auth import get_nlp_client
from google.cloud import language_v1
from google.api_core.exceptions import InvalidArgument
from logger import logger


# ---------------------------------------------------------------------------
# spaCy language model registry
# ---------------------------------------------------------------------------
# Maps BCP-47 language codes to their recommended spaCy model name.
# Add entries here as you onboard new languages.
# Models must be installed separately — see the walkthrough in the docs.
SPACY_LANGUAGE_MODELS: Dict[str, str] = {
    "sv": "sv_core_news_lg",  # Swedish  — latest 3.8.0
    # Examples for extending to other languages:
    # "de": "de_core_news_lg",
    # "fr": "fr_core_news_lg",
    # "es": "es_core_news_lg",
    "nl": "nl_core_news_lg",
    # "nb": "nb_core_news_lg",  # Norwegian Bokmål
    # "fi": "fi_core_news_lg",
    # "da": "da_core_news_lg",
    # "pl": "pl_core_news_lg",
}

# Internal cache: language_code -> loaded spaCy Language object
_spacy_model_cache: Dict[str, object] = {}

# Universal Dependencies POS tags (shared by spaCy and Google NLP) that count
# as verbs vs. learnable vocab, used by both this module and phrases/search.py.
VERB_POS_TAGS = ["VERB", "AUX"]
VOCAB_POS_TAGS = ["NOUN", "ADJ", "ADV"]


def _load_spacy_model(language_code: str) -> Optional[object]:
    """
    Lazily load and cache a spaCy model for a given language code.

    If the model is not yet installed it will be downloaded automatically
    via ``spacy.cli.download`` and then loaded.  The loaded model is cached
    so subsequent calls are free.

    Args:
        language_code: BCP-47 language code (e.g. 'sv', 'de').

    Returns:
        A loaded spaCy Language object, or None if unavailable.
    """
    if language_code in _spacy_model_cache:
        return _spacy_model_cache[language_code]

    model_name = SPACY_LANGUAGE_MODELS.get(language_code)
    if not model_name:
        logger.debug(
            "_load_spacy_model: no spaCy model registered for language '%s'",
            language_code,
        )
        return None

    try:
        import spacy  # noqa: PLC0415  (deferred import — optional dependency)

        try:
            nlp = spacy.load(model_name)
        except OSError:
            # Model not installed — download it automatically and retry.
            logger.info(
                "_load_spacy_model: model '%s' not found locally, downloading…",
                model_name,
            )
            try:
                spacy.cli.download(model_name)
            except SystemExit as exc:
                # spacy.cli.download calls sys.exit(0) on success, which raises
                # SystemExit(0).  Any non-zero exit code is a real failure.
                if exc.code != 0:
                    raise RuntimeError(
                        f"spacy.cli.download failed for '{model_name}' "
                        f"(exit code {exc.code})"
                    ) from exc
            nlp = spacy.load(model_name)

        _spacy_model_cache[language_code] = nlp
        logger.info(
            "_load_spacy_model: loaded spaCy model '%s' for language '%s'",
            model_name,
            language_code,
        )
        return nlp

    except RuntimeError as exc:
        logger.warning(
            "_load_spacy_model: could not download spaCy model '%s': %s",
            model_name,
            exc,
        )
        return None
    except ImportError:
        logger.warning(
            "_load_spacy_model: spaCy is not installed. "
            "Install it with: pip install spacy"
        )
        return None


# ---------------------------------------------------------------------------
# spaCy POS tag → normalised POS name
# ---------------------------------------------------------------------------
# spaCy uses Universal Dependencies POS tags (same labels as Google NLP),
# so no mapping is required.  Both APIs produce "VERB", "NOUN", "ADJ", etc.
# The one exception is that spaCy uses "AUX" for auxiliaries — same as Google.
# "PUNCT" is also the same.  No translation layer needed.


def _analyze_text_syntax_spacy(
    text: str, language_code: str
) -> Optional[List[Tuple[str, str, str]]]:
    """
    Analyse text with a spaCy model and return (token, lemma, pos) triples.

    This mirrors the output of the Google NLP path so callers are agnostic
    to which backend produced the result.

    Args:
        text: Text to analyse.
        language_code: BCP-47 language code.

    Returns:
        List of (token_text, lemma, pos) triples, or None if no model is available.
    """
    nlp = _load_spacy_model(language_code)
    if nlp is None:
        return None

    doc = nlp(text)
    return [(token.text, token.lemma_.lower(), token.pos_) for token in doc]


# ---------------------------------------------------------------------------
# Public API — Google NLP with spaCy fallback
# ---------------------------------------------------------------------------


def analyze_text_syntax(
    text: str, language_code: str = "en"
) -> language_v1.AnalyzeSyntaxResponse | None:
    """
    Analyze text syntax using Google Natural Language API.

    Args:
        text: Text to analyze
        language_code: BCP-47 language code (e.g., 'en', 'en-US')

    Returns:
        AnalyzeSyntaxResponse containing tokens with lemmas and POS tags,
        or None if the language is not supported by the Google API.
    """
    client = get_nlp_client()

    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT,
        language=language_code,
    )

    try:
        response = client.analyze_syntax(
            request={
                "document": document,
                "encoding_type": language_v1.EncodingType.UTF8,
            }
        )

        return response
    except InvalidArgument:
        # Raised by the API when the provided language is not supported
        logger.warning(
            "analyze_text_syntax: language not supported by Google NLP: %s",
            language_code,
        )
        return None


def _extract_token_lemma_pos_for_text(
    text: str, language_code: str
) -> List[Tuple[str, str, str]]:
    """
    Analyse a single text, preferring spaCy for registered languages and
    falling back to Google NLP otherwise (or if the spaCy model isn't
    installed).

    Args:
        text: Text to analyse.
        language_code: BCP-47 language code.

    Returns:
        List of (token_text, lemma, pos) triples, preserving the native
        surface form of each token alongside its lemma and POS tag.
    """
    use_spacy_first = language_code in SPACY_LANGUAGE_MODELS

    if use_spacy_first:
        spacy_results = _analyze_text_syntax_spacy(text, language_code)
        if spacy_results:
            return spacy_results
        # spaCy model not installed — fall through to Google NLP
        logger.warning(
            "_extract_token_lemma_pos_for_text: spaCy model not available for '%s', "
            "falling back to Google NLP",
            language_code,
        )

    response = analyze_text_syntax(text, language_code)
    if response is not None:
        triples: List[Tuple[str, str, str]] = []
        for token in response.tokens:
            pos_tag = language_v1.PartOfSpeech.Tag(token.part_of_speech.tag).name
            lemma = token.lemma.lower()
            triples.append((token.text.content, lemma, pos_tag))
        return triples

    logger.warning(
        "_extract_token_lemma_pos_for_text: no backend could process text "
        "for language '%s': %s",
        language_code,
        text,
    )
    return []


def extract_lemmas_and_pos(
    phrases: List[str], language_code: str = "en"
) -> List[Tuple[str, str]]:
    """
    Extract (lemma, POS) tuples from phrases.

    If the language is registered in SPACY_LANGUAGE_MODELS, spaCy is used
    as the primary backend (since we know Google NLP won't support it).
    For all other languages, Google NLP is tried first with spaCy as fallback.

    Args:
        phrases: List of phrases to analyze
        language_code: BCP-47 language code (default: 'en')

    Returns:
        List of (lemma, pos) tuples
    """
    vocab_set: List[Tuple[str, str]] = []
    for phrase in phrases:
        triples = _extract_token_lemma_pos_for_text(phrase, language_code)
        vocab_set.extend((lemma, pos) for _, lemma, pos in triples)
    return vocab_set


def extract_token_lemma_pos(
    text: str, language_code: str = "en"
) -> List[Tuple[str, str, str]]:
    """
    Extract (token_text, lemma, pos) triples from a single text, preserving
    the native surface form of each token (e.g. a past-tense verb form)
    rather than only its dictionary lemma.

    Useful when downstream matching should prefer an exact inflected-form
    match before falling back to lemma-based matching (see
    phrases.search.add_tags_from_text).

    Args:
        text: Text to analyze
        language_code: BCP-47 language code (default: 'en')

    Returns:
        List of (token_text, lemma, pos) triples, in order.
    """
    return _extract_token_lemma_pos_for_text(text, language_code)


def get_vocab_from_phrases(phrases: List[str], language_code: str = "en") -> List[str]:
    """
    Extract vocabulary words (non-verbs) from a list of phrases.

    Args:
        phrases: List of phrases to analyze
        language_code: BCP-47 language code (default: 'en')

    Returns:
        List of vocabulary words (non-verbs)
    """
    lemmas_and_pos = extract_lemmas_and_pos(phrases, language_code)
    if lemmas_and_pos:
        vocab = get_vocab_from_lemmas_and_pos(lemmas_and_pos)
    else:
        logger.warning(
            "get_vocab_from_phrases: no vocabulary found for phrases: %s",
            phrases,
        )
        vocab = []
    return vocab


def get_verbs_from_phrases(phrases: List[str], language_code: str = "en") -> List[str]:
    """
    Extract verbs from a list of phrases.

    Args:
        phrases: List of phrases to analyze
        language_code: BCP-47 language code (default: 'en')

    Returns:
        List of verbs
    """
    lemmas_and_pos = extract_lemmas_and_pos(phrases, language_code)
    verbs = get_verbs_from_lemmas_and_pos(lemmas_and_pos)
    return verbs


def get_verbs_and_vocab(
    phrases: List[str],
    language: Union[Language, str],
    filter_by_wiktionary: bool = False,
) -> Dict[str, List[str]]:
    """
    Extract verbs and vocab from a list of phrases, returning a dict.

    Accepts a ``langcodes.Language`` object or a plain BCP-47 string (e.g.
    ``'sv'``, ``'en-GB'``).  The language code passed to the NLP backends is
    always the two-letter alpha-2 form (e.g. ``'sv'``) so that both the spaCy
    registry and the Google NLP API receive a consistent key.

    Args:
        phrases:  List of phrase strings to analyse.
        language: Target language as a ``Language`` object or BCP-47 string.
        filter_by_wiktionary: If True, discard any lemma not found in the local
            Wiktionary database (verbs checked against pos='verb', vocab against
            pos='noun' or pos='adj').  Useful for noisy input like subtitle data
            where NLP artefacts (e.g. '-flytta') may appear as false positives.

    Returns:
        Dict with keys ``'verbs'`` and ``'vocab'``, each containing a
        deduplicated list of lemma strings.
    """
    lang_obj: Language = get_language(language)
    # Use the two-letter language code for both spaCy registry lookups and
    # Google NLP API calls (e.g. 'sv', not 'sv-SE').
    language_code: str = lang_obj.language

    lemmas_and_pos = extract_lemmas_and_pos(phrases, language_code)
    result = {
        "verbs": get_verbs_from_lemmas_and_pos(lemmas_and_pos),
        "vocab": get_vocab_from_lemmas_and_pos(lemmas_and_pos),
    }

    if filter_by_wiktionary:
        from wiktionary.lookup import word_in_wiktionary

        result["verbs"] = [
            w for w in result["verbs"] if word_in_wiktionary(w, language_code, pos="verb")
        ]
        result["vocab"] = [
            w
            for w in result["vocab"]
            if word_in_wiktionary(w, language_code, pos="noun")
            or word_in_wiktionary(w, language_code, pos="adj")
            or word_in_wiktionary(w, language_code, pos="adv")
        ]

    return result


def get_verbs_from_lemmas_and_pos(lemmas_and_pos: List[Tuple[str, str]]) -> list[str]:
    """Extract verbs from a set of (word, pos) tuples."""
    verbs = [word for word, pos in lemmas_and_pos if pos in VERB_POS_TAGS]
    return list(set(verbs))


def get_vocab_from_lemmas_and_pos(lemmas_and_pos: List[Tuple[str, str]]) -> list[str]:
    """Extract vocabulary words suitable for learning from a set of (word, pos) tuples."""
    vocab = [word for word, pos in lemmas_and_pos if pos in VOCAB_POS_TAGS]
    return list(set(vocab))


def get_tokens_from_lemmas_and_pos(lemmas_and_pos: List[Tuple[str, str]]) -> list[str]:
    """Extract tokens from a set of (word, pos) tuples."""
    tokens = [word for word, pos in lemmas_and_pos if pos not in ["PUNCT"]]
    return tokens


def get_text_tokens(
    text: str, language_code: str = "en", split_on_space: bool = False
) -> List[str]:
    """
    Tokenize text using language-appropriate methods.

    For space-separated languages: Simply split on spaces.
    Tries Google NLP API, then spaCy, then falls back to splitting on spaces.

    Args:
        text: Text to tokenize
        language_code: BCP-47 language code (default: 'en')

    Returns:
        List of tokens suitable for TTS breaks and Wiktionary lookups
    """
    if not text:
        return []

    if split_on_space:
        return text.split() if " " in text else [text]

    # If the language is registered in spaCy, use it as the primary tokeniser
    if language_code in SPACY_LANGUAGE_MODELS:
        nlp = _load_spacy_model(language_code)
        if nlp is not None:
            doc = nlp(text)
            return [token.text for token in doc if not token.is_space]
        # Model not installed — fall through to Google NLP
        logger.warning(
            "get_text_tokens: spaCy model not available for '%s', "
            "falling back to Google NLP",
            language_code,
        )

    response = analyze_text_syntax(text, language_code)
    if response:
        return [token.text.content for token in response.tokens]

    return text.split() if " " in text else [text]
