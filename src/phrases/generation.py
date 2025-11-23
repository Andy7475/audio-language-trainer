"""Phrase generation module for language learners.

This module provides phrase generation using an iterative approach based on verbs and vocabulary.
Instead of trying to "extinguish" a vocabulary list, we iterate through verbs and vocab items,
generating phrases and naturally tracking which words have been used.
"""

from typing import Dict, List, Tuple

from src.llm_tools.verb_phrase_generation import generate_verb_phrases
from src.llm_tools.vocab_phrase_generation import generate_vocab_phrases


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
        "errors": []
    }

    verb_list = vocab_dict["verbs"]
    vocab_list = vocab_dict["vocab"]

    # Generate phrases from verbs
    print(f"Starting verb phrase generation. {len(verb_list)} verbs to process.")
    all_phrases, verb_tracking = _generate_verb_phrases_batch(verb_list)
    tracking_info["verb_phrases"] = len(all_phrases)
    tracking_info["verbs_processed"] = verb_tracking["verbs_processed"]
    tracking_info["words_used"].extend(verb_tracking["additional_words"])
    tracking_info["errors"].extend(verb_tracking["errors"])

    # Generate phrases from vocabulary items
    print(f"\nStarting vocab phrase generation. {len(vocab_list)} vocab words to process.")
    vocab_phrases, vocab_tracking = _generate_vocab_phrases_batch(
        vocab_list,
        max_iterations=max_iterations
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
            print(f"  [{i}/{len(verb_list)}] Generating phrases for verb: '{verb}'")
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
            print(f"  ERROR: {error_msg}")
            errors.append(error_msg)

    return phrases, {
        "verbs_processed": processed_count,
        "additional_words": all_additional_words,
        "errors": errors
    }


def _generate_vocab_phrases_batch(
    vocab_list: List[str],
    max_iterations: int = 10,
    batch_size: int = 10,
) -> Tuple[List[str], Dict[str, any]]:
    """Process vocabulary items and generate phrases for each in batches.

    Groups vocabulary words into batches of N (default 10) and processes each batch
    in a single API call for efficiency. Optionally iterates multiple times through
    the vocabulary list.

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
    vocab_index = 0
    vocab_len = len(vocab_list)

    for iteration in range(1, max_iterations + 1):
        if vocab_index >= vocab_len:
            print(f"  Iteration {iteration}: All vocabulary processed.")
            break

        print(f"  Iteration {iteration}/{max_iterations}")

        # Process words in batches of batch_size
        while vocab_index < vocab_len:
            batch_start = vocab_index
            batch_end = min(vocab_index + batch_size, vocab_len)
            batch_words = vocab_list[batch_start:batch_end]

            # Get context: next 25 words from current batch position
            context_start = batch_end
            context_end = min(context_start + 25, vocab_len)
            context_words = vocab_list[context_start:context_end]

            try:
                print(f"    [{batch_start+1}-{batch_end}/{vocab_len}] Generating phrases for {len(batch_words)} words")
                result = generate_vocab_phrases(batch_words, context_words=context_words)

                # Extract phrases from results
                for result_data in result.get("results", []):
                    phrases.append(result_data["phrase"])

                # Track additional words
                all_additional_words.extend(result.get("all_additional_words", []))
                processed_count += len(batch_words)

            except Exception as e:
                error_msg = f"Error generating phrases for batch {batch_start+1}-{batch_end}: {str(e)}"
                print(f"    ERROR: {error_msg}")
                errors.append(error_msg)

            vocab_index = batch_end

        # Reset index for next iteration if we haven't reached the end
        if iteration < max_iterations and vocab_index >= vocab_len:
            vocab_index = 0

    return phrases, {
        "vocab_processed": processed_count,
        "additional_words": all_additional_words,
        "errors": errors
    }


def get_additional_words_for_tracking(phrases: List[str]) -> List[str]:
    """Extract words from a list of phrases for vocabulary tracking.

    This is a utility function that can be used to manually extract words
    from phrases if needed. Words are extracted by splitting on spaces and
    converting to lowercase.

    Args:
        phrases: List of phrase strings

    Returns:
        List of unique words extracted from phrases
    """
    words = set()
    for phrase in phrases:
        # Split phrase and extract words, removing common articles and prepositions
        phrase_words = phrase.lower().split()
        for word in phrase_words:
            # Skip common articles and prepositions that don't add vocabulary value
            if word.strip(".,?!\"'") not in {"a", "an", "the", "of", "in", "on", "at", "to", "from", "with", "by"}:
                words.add(word.strip(".,?!\"'"))
    return sorted(list(words))
