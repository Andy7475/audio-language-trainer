"""Phrase generation module for language learners.

This module provides phrase generation using an iterative approach based on verbs and vocabulary.
Instead of trying to "extinguish" a vocabulary list, we iterate through verbs and vocab items,
generating phrases and naturally tracking which words have been used.
"""

from typing import Dict, List, Tuple

from ..llm_tools.verb_phrase_generation import generate_verb_phrases
from ..llm_tools.vocab_phrase_generation import generate_vocab_phrases
from ..nlp import get_vocab_from_phrases
from ..logger import logger

def generate_phrases_from_vocab_dict(
    vocab_dict: Dict[str, List[str]],
    max_iterations: int = 10,
) -> Tuple[List[str], Dict[str, any]]:
    """Generate English phrases from a vocabulary dictionary using verb and vocab iteration.

    This method iterates through verbs to generate action-based phrases, then through vocabulary
    items to generate descriptive phrases, naturally tracking word usage without complex
    vocabulary "extinguishing" logic.

    Args:
        vocab_dict: Dictionary with keys 'verbs' and 'vocab' containing lists of words
        max_iterations: Maximum number of vocabulary iteration rounds (only applies to vocab generation)

    Returns:
        Tuple containing:
            - List of generated phrase strings
            - Dictionary with tracking information:
                {
                    "total_phrases": int,
                    "verb_phrases": int,
                    "vocab_phrases": int,
                    "verbs_processed": int,
                    "vocab_processed": int,
                    "words_used": List[str] (additional words extracted from phrases),
                    "errors": List[str] (any errors encountered during generation)
                }

    Raises:
        ValueError: If vocab_dict doesn't have required 'verbs' or 'vocab' keys
    """
    if "verbs" not in vocab_dict or "vocab" not in vocab_dict:
        raise ValueError("vocab_dict must contain 'verbs' and 'vocab' keys")

    all_phrases = []
    tracking_info = {
        "total_phrases": 0,
        "verb_phrases": 0,
        "vocab_phrases": 0,
        "verbs_processed": 0,
        "vocab_processed": 0,
        "words_used": [],
        "errors": [],
    }

    verb_list = vocab_dict["verbs"]
    vocab_list = vocab_dict["vocab"]

    # Generate phrases from verbs
    logger.info(f"Starting verb phrase generation. {len(verb_list)} verbs to process.")
    all_phrases, verb_tracking = _generate_verb_phrases_batch(verb_list)
    tracking_info["verb_phrases"] = len(all_phrases)
    tracking_info["verbs_processed"] = verb_tracking["verbs_processed"]
    tracking_info["words_used"].extend(verb_tracking["additional_words"])
    tracking_info["errors"].extend(verb_tracking["errors"])

    vocab_present_in_verb_phrases = get_vocab_from_phrases(all_phrases)
    vocab_list = _remove_words_from_list(vocab_list, vocab_present_in_verb_phrases)
    logger.info(
        f"\nStarting vocab phrase generation. {len(vocab_list)} vocab words to process."
    )
    vocab_phrases, vocab_tracking = _generate_vocab_phrases_batch(
        vocab_list, max_iterations=max_iterations
    )
    all_phrases.extend(vocab_phrases)
    tracking_info["vocab_phrases"] = len(vocab_phrases)
    tracking_info["vocab_processed"] = vocab_tracking["vocab_processed"]
    tracking_info["words_used"].extend(vocab_tracking["additional_words"])
    tracking_info["errors"].extend(vocab_tracking["errors"])

    # Finalize tracking
    tracking_info["total_phrases"] = len(all_phrases)
    tracking_info["words_used"] = list(set(tracking_info["words_used"]))  # Deduplicate

    return all_phrases, tracking_info


def _generate_verb_phrases_batch(
    verb_list: List[str],
) -> Tuple[List[str], Dict[str, any]]:
    """Process all verbs and generate phrases for each.

    Args:
        verb_list: List of verbs to generate phrases for

    Returns:
        Tuple of (phrases list, tracking dict)
    """
    phrases = []
    all_additional_words = []
    errors = []
    processed_count = 0

    for i, verb in enumerate(verb_list, 1):
        try:
            logger.info(f"  [{i}/{len(verb_list)}] Generating phrases for verb: '{verb}'")
            result = generate_verb_phrases(verb)

            # Extract all phrases
            for base_phrase in result.get("base_phrases", []):
                phrases.append(base_phrase["phrase"])
            for meaning_phrase in result.get("meaning_variations", []):
                phrases.append(meaning_phrase["phrase"])

            # Track additional words
            all_additional_words.extend(result.get("all_additional_words", []))
            processed_count += 1

        except Exception as e:
            error_msg = f"Error generating phrases for verb '{verb}': {str(e)}"
            logger.info(f"  ERROR: {error_msg}")
            errors.append(error_msg)

    return phrases, {
        "verbs_processed": processed_count,
        "additional_words": all_additional_words,
        "errors": errors,
    }


def _remove_words_from_list(
    word_list: List[str], words_to_remove: List[str]
) -> List[str]:
    """Remove specified words from a list.

    Args:
        word_list: Original list of words
        words_to_remove: List of words to remove from the original list

    Returns:
        New list with specified words removed
    """
    return [word for word in word_list if word not in words_to_remove]


def _generate_vocab_phrases_batch(
    vocab_list: List[str],
    max_iterations: int = 10,
    batch_size: int = 20,
) -> Tuple[List[str], Dict[str, any]]:
    """Process vocabulary items and generate phrases for each in batches.

    Groups vocabulary words into batches of N (default 10) and processes each batch
    in a single API call for efficiency. Words that appear as "additional words" in
    generated phrases are removed from the remaining vocabulary to avoid redundant
    processing. Optionally iterates multiple times through the vocabulary list.

    Args:
        vocab_list: List of vocabulary words
        max_iterations: Maximum number of passes through vocabulary
        batch_size: Number of words to process per API call (default 10)

    Returns:
        Tuple of (phrases list, tracking dict)
    """
    phrases = []
    all_additional_words = []
    errors = []
    processed_count = 0

    # Track remaining words to process (will be reduced as words are used in phrases)
    remaining_words = vocab_list.copy()
    initial_count = len(vocab_list)

    for iteration in range(1, max_iterations + 1):
        if not remaining_words:
            logger.info(f"  Iteration {iteration}: All vocabulary processed.")
            break

        logger.info(f"  Iteration {iteration}/{max_iterations}")

        # Always take from the front of remaining_words
        batch_num = 0
        while remaining_words:
            # Get current batch from the front
            batch_words = remaining_words[:batch_size]

            # Get context: next 25 words after this batch
            context_words = remaining_words[batch_size : batch_size + 25]

            # Calculate display info
            batch_num += 1
            processed_so_far = initial_count - len(remaining_words)
            batch_start_display = processed_so_far + 1
            batch_end_display = processed_so_far + len(batch_words)

            try:
                logger.info(
                    f"    [{batch_start_display}-{batch_end_display}/{initial_count}] Generating phrases for {len(batch_words)} words"
                )
                result = generate_vocab_phrases(
                    batch_words, context_words=context_words
                )

                # Extract phrases from results
                for result_data in result.get("results", []):
                    phrases.append(result_data["phrase"])

                # Track additional words from this batch
                batch_additional_words = result.get("all_additional_words", [])
                all_additional_words.extend(batch_additional_words)
                processed_count += len(batch_words)

                # Remove the processed batch words from remaining list
                remaining_words = remaining_words[len(batch_words) :]

                # Also remove words that were used in phrases from remaining list
                # This prevents processing words that have already appeared in generated phrases
                remaining_words = _remove_words_from_list(
                    remaining_words, batch_additional_words
                )

            except Exception as e:
                error_msg = f"Error generating phrases for batch {batch_start_display}-{batch_end_display}: {str(e)}"
                logger.info(f"    ERROR: {error_msg}")
                errors.append(error_msg)
                # Move past this batch even on error
                remaining_words = remaining_words[len(batch_words) :]

        # After completing iteration, prepare for next iteration if needed
        # remaining_words already has used words removed, so we'll process what's left
        if remaining_words:
            logger.info(
                f"  Completed iteration {iteration}. {len(remaining_words)} words remaining for next iteration."
            )

    return phrases, {
        "vocab_processed": processed_count,
        "additional_words": all_additional_words,
        "errors": errors,
    }
