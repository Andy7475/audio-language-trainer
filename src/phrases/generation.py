"""Phrase generation module for language learners."""

from typing import Dict, List

from langcodes import Language

from llm_tools.verb_phrase_generation import generate_verb_phrases
from llm_tools.vocab_phrase_generation import generate_vocab_phrases
from models import get_language
from nlp import get_vocab_from_phrases
from logger import logger


def generate_phrases_from_vocab_dict(
    vocab_dict: Dict[str, List[str]],
    language: Language | str | None = None,
) -> List[str]:
    """Generate phrases from a vocabulary dictionary.

    Iterates through verbs first (present/past/future tenses), then generates
    descriptive phrases for remaining vocabulary words not already covered.

    Args:
        vocab_dict: Dictionary with keys 'verbs' and 'vocab' containing lists of words
        language: Target language for phrase generation (default: en-GB)

    Returns:
        List of generated phrase strings

    Raises:
        ValueError: If vocab_dict doesn't have required 'verbs' or 'vocab' keys
    """
    if "verbs" not in vocab_dict or "vocab" not in vocab_dict:
        raise ValueError("vocab_dict must contain 'verbs' and 'vocab' keys")

    language = get_language(language)

    logger.info(f"Starting verb phrase generation. {len(vocab_dict['verbs'])} verbs to process.")
    all_phrases = _generate_verb_phrases_batch(vocab_dict["verbs"], language=language)

    vocab_present_in_verb_phrases = get_vocab_from_phrases(all_phrases)
    remaining_vocab = _remove_words_from_list(vocab_dict["vocab"], vocab_present_in_verb_phrases)

    logger.info(f"Starting vocab phrase generation. {len(remaining_vocab)} vocab words to process.")
    vocab_phrases = _generate_vocab_phrases_batch(remaining_vocab, language=language)
    all_phrases.extend(vocab_phrases)

    return all_phrases


def _generate_verb_phrases_batch(
    verb_list: List[str],
    language: Language | None = None,
) -> List[str]:
    """Generate present/past/future phrases for each verb."""
    phrases = []

    for i, verb in enumerate(verb_list, 1):
        try:
            logger.info(f"  [{i}/{len(verb_list)}] Generating phrases for verb: '{verb}'")
            result = generate_verb_phrases(verb, language=language)
            for base_phrase in result.get("base_phrases", []):
                phrases.append(base_phrase["phrase"])
            for meaning_phrase in result.get("meaning_variations", []):
                phrases.append(meaning_phrase["phrase"])
        except Exception as e:
            logger.error(f"  Error generating phrases for verb '{verb}': {e}")

    return phrases


def _remove_words_from_list(word_list: List[str], words_to_remove: List[str]) -> List[str]:
    return [word for word in word_list if word not in words_to_remove]


def _generate_vocab_phrases_batch(
    vocab_list: List[str],
    batch_size: int = 20,
    language: Language | None = None,
) -> List[str]:
    """Generate descriptive phrases for vocabulary words in batches."""
    phrases = []
    remaining_words = vocab_list.copy()
    initial_count = len(vocab_list)

    while remaining_words:
        batch_words = remaining_words[:batch_size]
        context_words = remaining_words[batch_size : batch_size + 25]
        processed_so_far = initial_count - len(remaining_words)

        try:
            logger.info(
                f"  [{processed_so_far + 1}-{processed_so_far + len(batch_words)}/{initial_count}] "
                f"Generating phrases for {len(batch_words)} words"
            )
            result = generate_vocab_phrases(
                batch_words, context_words=context_words, language=language
            )

            for result_data in result.get("results", []):
                phrases.append(result_data["phrase"])

            additional_words = result.get("all_additional_words", [])
            remaining_words = remaining_words[len(batch_words):]
            remaining_words = _remove_words_from_list(remaining_words, additional_words)

        except Exception as e:
            logger.error(f"  Error generating phrases for batch: {e}")
            remaining_words = remaining_words[len(batch_words):]

    return phrases
