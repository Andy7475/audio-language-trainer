from typing import List, Tuple


from src.connections.gcloud_auth import get_nlp_client
from google.cloud import language_v1
from google.api_core.exceptions import InvalidArgument
from src.logger import logger


def analyze_text_syntax(
    text: str, language_code: str = "en"
) -> language_v1.AnalyzeSyntaxResponse | None:
    """
    Analyze text syntax using Google Natural Language API.

    Args:
        text: Text to analyze
        language_code: BCP-47 language code (e.g., 'en', 'en-US')

    Returns:
        AnalyzeSyntaxResponse containing tokens with lemmas and POS tags
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
    except InvalidArgument as e:
        # Raised by the API when the provided language is not supported
        logger.warning(
            "analyze_text_syntax: language not supported by Google NLP: %s",
            language_code,
        )
        return None


def extract_lemmas_and_pos(
    phrases: List[str], language_code: str = "en"
) -> List[Tuple[str, str]]:
    """
    Extract (lemma, POS) tuples from phrases using Google NLP API.

    Args:
        phrases: List of phrases to analyze
        language_code: BCP-47 language code (default: 'en')

    Returns:
        Set of (lemma, pos) tuples
    """
    vocab_set = []

    for phrase in phrases:
        response = analyze_text_syntax(phrase, language_code)
        if response:
            for token in response.tokens:
                # Get POS tag name (e.g., 'VERB', 'NOUN', 'ADJ')
                pos_tag = language_v1.PartOfSpeech.Tag(token.part_of_speech.tag).name

                lemma = token.lemma.lower()

                vocab_set.append((lemma, pos_tag))

        return vocab_set

    return vocab_set


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


def get_verbs_from_lemmas_and_pos(lemmas_and_pos: List[Tuple[str, str]]) -> list[str]:
    """Extract verbs from a set of (word, pos) tuples."""
    verbs = [word for word, pos in lemmas_and_pos if pos in ["VERB", "AUX"]]
    return list(set(verbs))


def get_vocab_from_lemmas_and_pos(lemmas_and_pos: List[Tuple[str, str]]) -> list[str]:
    """Extract vocab (non-verbs) from a set of (word, pos) tuples."""
    vocab = [
        word for word, pos in lemmas_and_pos if pos not in ["VERB", "AUX", "PUNCT"]
    ]
    return list(set(vocab))


def get_tokens_from_lemmas_and_pos(lemmas_and_pos: List[Tuple[str, str]]) -> list[str]:
    """Extract tokens from a set of (word, pos) tuples."""
    tokens = [word for word, pos in lemmas_and_pos if pos not in ["PUNCT"]]
    return tokens


def get_text_tokens(text: str, language_code: str = "en") -> List[str]:
    """
    Tokenize text using language-appropriate methods.

    For space-separated languages: Simply split on spaces
    For other languages: Use Google Cloud Natural Language API

    Args:
        text: Text to tokenize
        language_code: BCP-47 language code (default: 'en')

    Returns:
        List of tokens suitable for TTS breaks and Wiktionary lookups
    """
    if not text:
        return []

    response = analyze_text_syntax(text, language_code)
    if response:
        return [token.text.content for token in response.tokens]
    else:
        return text.split() if " " in text else [text]

