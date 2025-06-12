import re
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spacy
from plotly.subplots import make_subplots
from tqdm import tqdm

from src.convert import clean_filename


def get_text_from_dialogue(dialogue: List[Dict[str, str]]) -> List[str]:
    """ignoring the speaker, just gets all the utterances from a dialogue and puts
    them in a single list"""

    phrases = []
    for utterance in dialogue:
        phrases.append(utterance["text"])
    return phrases


def plot_vocabulary_growth(phrases: List[str], window: int = 15) -> None:
    """
    Plot vocabulary growth with cumulative total, new words per phrase, rolling mean,
    and overall mean.

    Args:
        phrases: List of phrases to analyze
        window: Window size for rolling mean calculation
    """
    vocab = set()
    cumulative_counts = []
    new_words_per_phrase = []

    # Calculate new words and cumulative counts
    for phrase in phrases:
        current_words = set(phrase.lower().split())
        new_words = len(current_words - vocab)
        new_words_per_phrase.append(new_words)

        vocab.update(current_words)
        cumulative_counts.append(len(vocab))

    # Calculate rolling mean and overall mean
    df = pd.DataFrame({"new_words": new_words_per_phrase})
    rolling_mean = (
        df["new_words"].rolling(window=min(window, len(phrases)), min_periods=1).mean()
    )
    overall_mean = np.mean(new_words_per_phrase)

    # Create plot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add cumulative vocabulary trace
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(phrases) + 1)),
            y=cumulative_counts,
            name="Total Vocabulary",
            line=dict(color="blue"),
        ),
        secondary_y=False,
    )

    # Add rolling mean line
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(phrases) + 1)),
            y=rolling_mean,
            name="New Words per Phrase\n(rolling mean)",
            line=dict(color="red"),
        ),
        secondary_y=True,
    )

    # Add overall mean line
    fig.add_trace(
        go.Scatter(
            x=[1, len(phrases)],
            y=[overall_mean, overall_mean],
            name="Overall new words per phrase",
            line=dict(color="purple", dash="dot"),
        ),
        secondary_y=True,
    )

    # Update layout
    fig.update_layout(
        title="Vocabulary Growth Analysis",
        xaxis_title="Phrase Number",
        showlegend=True,
        # Add margin to ensure legend is visible
        margin=dict(r=150),
        # Improve legend formatting
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
    )

    # Update y-axes titles
    fig.update_yaxes(
        title_text="Total Unique Words", secondary_y=False, gridcolor="lightgrey"
    )
    fig.update_yaxes(
        title_text="New Words per Phrase", secondary_y=True, gridcolor="lightgrey"
    )

    # Show plot
    fig.show()


def load_spacy_model():
    """Load spaCy model with error handling."""
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_md")
        return spacy.load("en_core_web_md")


def remove_matching_words(phrases: list[str], original_set: set[str]) -> set[str]:
    """
    Remove items from original_set that match with any word in phrases,
    ignoring parenthetical text in original_set items.

    So if our original set had 'falling (over)' as an entry, it would be removed if our
    phrases had 'falling' in it.

    This is because falling (over) as a prompt to an LLM can give better context for phrase
    creation, but we won't have the (over) returned in any of our phrases.
    """
    updated_set = original_set.copy()

    for item in original_set:
        base_word = re.sub(r"\([^)]*\)\s*", "", item).strip()
        if base_word in phrases:
            updated_set.remove(item)

    return updated_set


def get_vocab_dictionary_from_phrases(
    english_phrases: List[str],
) -> Dict[str, List[str]]:
    """Processes the english phrases to extract a vocabulary dictionary with keys
    'verbs' and 'vocab'. This is so we can, for a given chunk of phrases we are learning (in
    flash cards), extract the vocab, and then re-use that vocab to create a story to
    listen to (practice long-form listening)

    Returns: vocab_dict: {'verbs' : ['try', 'care', ...], 'vocab' : ['really', 'hello', ...]}
    """

    vocab_pos_tuples = extract_vocab_and_pos(
        english_phrases
    )  # [('trying', 'VERB'), etc]
    vocab_dict = get_verb_and_vocab_lists(vocab_pos_tuples)
    return vocab_dict


def extract_vocab_and_pos(english_phrases: List[str]) -> List[Tuple[str, str]]:
    """Returns the (lemma and POS) for feeding into update_vocab_usage, as a list."""
    # Process vocabulary
    nlp = load_spacy_model()

    vocab_set = set()
    excluded_names = {"sam", "alex"}

    for phrase in english_phrases:
        doc = nlp(phrase)

        for token in doc:
            if (
                token.pos_ != "PUNCT"
                and token.ent_type_ != "PERSON"
                and token.text.lower() not in excluded_names
            ):
                vocab_set.add((token.lemma_.lower(), token.pos_))

    return vocab_set


def get_vocab_dict_from_dialogue(
    story_dict: Dict, limit_story_parts: list = None
) -> Dict[str, List[str]]:
    """
    For a given English dialogue story dictionary {'introduction' : {'dialogue' : [...] etc}, extracts the vocab used and places it into a dictionary
    with keys 'verbs' and 'vocab', which is our common format.
    Excludes punctuation, persons identified by spaCy, and the names 'sam' and 'alex'.
    """

    if limit_story_parts:
        story_parts_to_process = limit_story_parts
    else:
        story_parts_to_process = list(story_dict.keys())  # all of them

    english_phrases = []
    for story_part in story_parts_to_process:
        content = story_dict.get(story_part)
        if content:
            for utterance in content.get("dialogue"):
                english_phrases.append(utterance["text"])
        else:
            raise KeyError(f"We are missing story_part {story_part} in the dictionary")

    return get_vocab_dictionary_from_phrases(english_phrases)


def find_missing_vocabulary(vocab_dict_source: dict, vocab_dict_target: dict) -> dict:
    """Compare vocabulary between source flashcards and target story to find gaps in coverage.

    Identifies which words in the target story aren't covered by existing flashcards,
    helping determine what new flashcards need to be created.

    Args:
        vocab_dict_source: Dictionary with 'verbs' and 'vocab' lists from existing flashcards
        vocab_dict_target: Dictionary with 'verbs' and 'vocab' lists from target story

    Returns:
        Dictionary containing:
        - missing_vocab: Dictionary with 'verbs' and 'vocab' lists containing uncovered words
        - coverage_stats: Dictionary with percentage coverage statistics
    """
    # Convert to sets for set operations
    source_verbs = set(vocab_dict_source["verbs"])
    target_verbs = set(vocab_dict_target["verbs"])

    source_vocab = set(vocab_dict_source["vocab"])
    target_vocab = set(vocab_dict_target["vocab"])

    # Find words in target not covered by source
    uncovered_verbs = target_verbs - source_verbs
    uncovered_vocab = target_vocab - source_vocab

    # Calculate coverage percentages
    verb_coverage = (
        len(target_verbs - uncovered_verbs) / len(target_verbs) * 100
        if target_verbs
        else 100
    )
    vocab_coverage = (
        len(target_vocab - uncovered_vocab) / len(target_vocab) * 100
        if target_vocab
        else 100
    )

    # Print analysis
    print("=== VOCABULARY COVERAGE ANALYSIS ===")
    print(f"Target verbs covered by flashcards: {verb_coverage:.1f}%")
    print(f"Target vocabulary covered by flashcards: {vocab_coverage:.1f}%")

    if uncovered_verbs:
        print("\nVerbs needing new flashcards:")
        print(list(uncovered_verbs)[:5], "..." if len(uncovered_verbs) > 5 else "")

    if uncovered_vocab:
        print("\nVocabulary needing new flashcards:")
        print(list(uncovered_vocab)[:5], "..." if len(uncovered_vocab) > 5 else "")

    return {
        "missing_vocab": {
            "verbs": list(uncovered_verbs),
            "vocab": list(uncovered_vocab),
        },
        "coverage_stats": {
            "verb_coverage": verb_coverage,
            "vocab_coverage": vocab_coverage,
            "total_target_verbs": len(target_verbs),
            "total_target_vocab": len(target_vocab),
        },
    }


def process_phrase_vocabulary(phrase: str) -> tuple[set, set, set]:
    """Process a single phrase to extract verb and vocab matches

    Returns:
        tuple containing:
        - set of (word, pos) tuples for all words
        - set of verb matches
        - set of vocab matches
    """
    vocab_used = extract_vocab_and_pos([phrase])
    verb_matches = set()
    vocab_matches = set()

    for word, pos in vocab_used:
        if pos in ["VERB", "AUX"]:
            verb_matches.add(word)
        else:
            vocab_matches.add(word)

    return vocab_used, verb_matches, vocab_matches


def create_flashcard_index(flashcard_phrases: list[str]) -> dict:
    """Create indexes mapping words to the flashcards containing them.
    Also stores vocabulary data for each phrase to avoid redundant processing.

    Args:
        flashcard_phrases: List of phrases to index

    Returns:
        Dictionary containing:
        {
            "verb_index": {word: [flashcard_idx1, flashcard_idx2, ...]},
            "vocab_index": {word: [flashcard_idx1, flashcard_idx2, ...]},
            "word_counts": [
                {
                    "verb_count": int,
                    "vocab_count": int,
                    "words": list of (word, pos) tuples
                },
                ...
            ],
            "phrases": list of phrases,
            "phrase_vocab": {
                "phrase_key1": {
                    "verbs": [...],
                    "vocab": [...],
                    "all_words": [...]
                },
                ...
            }
        }
    """
    verb_index = {}  # word -> set of flashcard indices
    vocab_index = {}
    flashcard_word_counts = []
    phrase_vocab = {}  # phrase_key -> vocabulary data

    for idx, phrase in tqdm(
        enumerate(flashcard_phrases),
        desc="Indexing phrases...",
        total=len(flashcard_phrases),
    ):
        # Process phrase vocabulary once
        vocab_used, verb_matches, vocab_matches = process_phrase_vocabulary(phrase)

        # Store vocabulary data for this phrase
        phrase_key = clean_filename(phrase)
        phrase_vocab[phrase_key] = {
            "verbs": list(verb_matches),
            "vocab": list(vocab_matches),
            "all_words": list(vocab_used),
        }

        # Build indexes
        for word in verb_matches:
            if word not in verb_index:
                verb_index[word] = set()
            verb_index[word].add(idx)

        for word in vocab_matches:
            if word not in vocab_index:
                vocab_index[word] = set()
            vocab_index[word].add(idx)

        # Store word counts
        flashcard_word_counts.append(
            {
                "verb_count": len(verb_matches),
                "vocab_count": len(vocab_matches),
                "words": list(vocab_used),
            }
        )

    # Convert sets to lists for JSON
    for word in verb_index:
        verb_index[word] = list(verb_index[word])
    for word in vocab_index:
        vocab_index[word] = list(vocab_index[word])

    return {
        "verb_index": verb_index,
        "vocab_index": vocab_index,
        "word_counts": flashcard_word_counts,
        "phrases": flashcard_phrases,
        "phrase_vocab": phrase_vocab,
    }


def extract_substring_matches(
    new_phrases: List[str], target_phrases: Set[str]
) -> Set[str]:
    """Should find matches due to the presence of phrasal verbs etc
    in our target_phrases (original vocab set) as this might contain
    multiple words or lexical chunks like 'what's the time?'

    We are basically checking that the phrases we have generated (new_phrases) have successfully
    'ticked off' words or lexical chunks we are trying to generate (target_phrases) that come
    from our vocab dict.

    WIth the set of phrases we return, we will remove those from the to-do list so we steadily
    erode away the target phrases / chunks we need to create"""
    # Convert all new phrases to lowercase
    lowercase_phrases = [phrase.lower() for phrase in new_phrases]

    # Convert all target phrases to lowercase
    lowercase_targets = [target.lower() for target in target_phrases]

    matched_substrings = set()

    for target in lowercase_targets:
        for phrase in lowercase_phrases:
            # Check for exact whole word matches with word boundaries
            if target in phrase:
                matched_substrings.add(target)
                break

    return matched_substrings


def extract_spacy_lowercase_words(new_phrases: List[str]) -> Set[str]:
    # Ensure the spaCy model is loaded
    nlp = load_spacy_model()

    # Initialize an empty set to store unique lowercase words
    lowercase_words = set()

    # Process each phrase with spaCy
    for phrase in new_phrases:
        doc = nlp(phrase)

        # Add the lowercase version of each token's text to the set
        lowercase_words.update(token.text.lower() for token in doc)

    return lowercase_words


def get_verb_and_vocab_lists(used_words: Set[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Separate the input set of (word, POS) tuples into verb and vocabulary lists.

    Args:
    used_words (Set[Tuple[str, str]]): A set of tuples containing (word, POS)

    Returns:
    Dict[str, List[str]]: A dictionary with 'verbs' and 'vocab' lists
    """
    verb_list = []
    vocab_list = []

    for word, pos in used_words:
        if pos in ["VERB", "AUX"]:
            verb_list.append(word)
        else:
            vocab_list.append(word)

    return {"verbs": verb_list, "vocab": vocab_list}


def create_story_index(
    story_dialogues: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]],
) -> Dict[str, Dict]:
    """
    Create indexes mapping words to the stories containing them.
    Similar to create_flashcard_index but for story dialogues.

    Args:
        story_dialogues: Dictionary mapping story names to story parts, where each part contains a dialogue list
            Format: {
                "story_name": {
                    "part1": {
                        "dialogue": [
                            {"speaker": "Alex", "text": "..."},
                            {"speaker": "Sam", "text": "..."},
                            ...
                        ]
                    },
                    "part2": {
                        "dialogue": [...]
                    },
                    ...
                },
                ...
            }

    Returns:
        Dictionary containing:
        {
            "verb_index": {word: [story_name1, story_name2, ...]},
            "vocab_index": {word: [story_name1, story_name2, ...]},
            "word_counts": {story_name: {"verb_count": int, "vocab_count": int, "words": list}},
            "stories": list of story names,
            "story_vocab": {
                "story_name1": {
                    "verbs": [...],
                    "vocab": [...],
                    "all_words": [...]
                },
                ...
            }
        }
    """
    verb_index = defaultdict(set)  # word -> set of story names
    vocab_index = defaultdict(set)
    story_word_counts = {}
    story_vocab = {}  # story_name -> vocabulary data

    for story_name, story_parts in tqdm(
        story_dialogues.items(), desc="Indexing stories...", total=len(story_dialogues)
    ):
        # Process all parts of the story
        all_verbs = set()
        all_vocab = set()
        all_words = set()

        for part_name, part_data in story_parts.items():
            # Get dialogue from the part
            dialogue = part_data["dialogue"]
            # Get all text from dialogue
            phrases = get_text_from_dialogue(dialogue)

            # Process all phrases in this part
            for phrase in phrases:
                vocab_used, verb_matches, vocab_matches = process_phrase_vocabulary(
                    phrase
                )

                # Add to story's word sets
                all_verbs.update(verb_matches)
                all_vocab.update(vocab_matches)
                all_words.update(vocab_used)

                # Add to indexes
                for word in verb_matches:
                    verb_index[word].add(story_name)
                for word in vocab_matches:
                    vocab_index[word].add(story_name)

        # Store story's word counts
        story_word_counts[story_name] = {
            "verb_count": len(all_verbs),
            "vocab_count": len(all_vocab),
            "words": list(all_words),
        }

        # Store vocabulary data for this story
        story_vocab[story_name] = {
            "verbs": list(all_verbs),
            "vocab": list(all_vocab),
            "all_words": list(all_words),
        }

    # Convert sets to lists for JSON
    for word in verb_index:
        verb_index[word] = list(verb_index[word])
    for word in vocab_index:
        vocab_index[word] = list(vocab_index[word])

    return {
        "verb_index": dict(verb_index),
        "vocab_index": dict(vocab_index),
        "word_counts": story_word_counts,
        "stories": list(story_dialogues.keys()),
        "story_vocab": story_vocab,
    }


def determine_story_sequence(story_index: Dict[str, Dict]) -> List[str]:
    """
    Determine optimal story sequence based on vocabulary complexity.
    Stories with smaller vocabulary are placed earlier in the sequence.
    Verbs are weighted 2x more heavily than regular vocabulary.

    Args:
        story_index: Pre-built story index from GCS

    Returns:
        List[str]: Story names sorted from simplest to most complex vocabulary
    """
    # Calculate vocabulary complexity score for each story
    story_scores = []

    for story in story_index["stories"]:
        # Get vocabulary counts
        verb_count = len(story_index["story_vocab"][story]["verbs"])
        vocab_count = len(story_index["story_vocab"][story]["vocab"])

        # Calculate complexity score (verbs weighted 2x)
        complexity_score = (verb_count * 2) + vocab_count

        # Store story name and score
        story_scores.append((story, complexity_score, verb_count, vocab_count))

    # Sort stories by complexity score (ascending)
    story_scores.sort(key=lambda x: x[1])

    # Print story sequence with complexity details
    print("\nOptimized Story Sequence:")
    print("-" * 60)
    print(f"{'Story':<30} {'Score':<10} {'Verbs':<10} {'Vocab':<10}")
    print("-" * 60)
    for story, score, verbs, vocab in story_scores:
        print(f"{story:<30} {score:<10} {verbs:<10} {vocab:<10}")

    # Extract and return just the story names in sorted order
    return [story[0] for story in story_scores]


def assign_phrases_to_stories(
    story_index: Dict[str, Dict],
    phrase_index: Dict[str, Dict],
    max_phrases_per_story: int = 50,
    target_new_words_per_card: float = 2.0,
    story_sequence: List[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Assign phrases to stories in sequence, tracking global vocabulary acquisition
    to create an optimal learning progression.

    Args:
        story_index: Pre-built story index from GCS
        phrase_index: Pre-built phrase index containing vocabulary data
        max_phrases_per_story: Target maximum phrases per story
        target_new_words_per_card: Target average new words per flashcard
        story_sequence: Optional list of story names in desired learning sequence.
                      If None, will be determined automatically based on vocabulary size.

    Returns:
        Dictionary mapping story names to lists of assigned phrases with scores
    """
    # Generate story sequence if not provided
    if story_sequence is None:
        story_sequence = determine_story_sequence(story_index)

    # Get all phrases
    phrases = phrase_index["phrases"]
    phrase_keys = [clean_filename(p) for p in phrases]

    # Initialize assignments and tracking
    assignments = {story: [] for story in story_sequence}
    story_phrase_counts = {story: 0 for story in story_sequence}

    # Global tracking of accumulated vocabulary knowledge
    global_known_verbs = set()
    global_known_vocab = set()

    # Set of remaining phrases
    remaining_phrases = set(phrase_keys)

    # Process each story in sequence
    for story in story_sequence:
        if story not in story_index["stories"]:
            raise KeyError(f"Story {story} not found in story index")

        print(f"\nProcessing story: {story}")

        # Get story vocabulary
        story_verbs = set(story_index["story_vocab"][story]["verbs"])
        story_vocab = set(story_index["story_vocab"][story]["vocab"])

        # Calculate remaining vocabulary to learn for this story
        remaining_story_verbs = story_verbs - global_known_verbs
        remaining_story_vocab = story_vocab - global_known_vocab

        print(
            f"Story has {len(remaining_story_verbs)} new verbs and {len(remaining_story_vocab)} new vocabulary words to learn"
        )

        # Track vocabulary covered by this story's flashcards
        story_covered_verbs = set()
        story_covered_vocab = set()

        # Determine target number of phrases based on vocabulary needs
        total_new_words = len(remaining_story_verbs) + len(remaining_story_vocab)
        target_phrases = min(
            max_phrases_per_story,
            max(10, int(total_new_words / target_new_words_per_card)),
        )

        print(f"Target number of phrases for this story: {target_phrases}")

        # Assign phrases until we reach target or exhaust options
        assigned_count = 0

        while (
            assigned_count < target_phrases
            and remaining_phrases
            and (remaining_story_verbs or remaining_story_vocab)
        ):
            # Find best remaining phrase for this story
            best_phrase = None
            best_score = -1
            best_new_words = 0
            best_info = None

            for phrase_key in remaining_phrases:
                if phrase_key not in phrase_index["phrase_vocab"]:
                    continue

                # Get vocab from phrase index
                phrase_verbs = set(phrase_index["phrase_vocab"][phrase_key]["verbs"])
                phrase_vocab = set(phrase_index["phrase_vocab"][phrase_key]["vocab"])

                # Calculate new words this phrase would teach for this story
                new_story_verbs = phrase_verbs & remaining_story_verbs
                new_story_vocab = phrase_vocab & remaining_story_vocab

                # Calculate new words globally
                new_global_verbs = phrase_verbs - global_known_verbs
                new_global_vocab = phrase_vocab - global_known_vocab

                # Score formula: prioritize story relevance but consider global learning
                # Weight verbs higher than regular vocabulary
                story_score = (len(new_story_verbs) * 3) + len(new_story_vocab)
                global_score = (len(new_global_verbs) * 2) + len(new_global_vocab)

                # Combined score weights story relevance higher
                score = (story_score * 2) + global_score

                # Favor phrases with total new words close to target
                total_new = len(new_global_verbs) + len(new_global_vocab)
                distance_penalty = abs(total_new - target_new_words_per_card)
                adjusted_score = score - distance_penalty

                if adjusted_score > best_score:
                    best_phrase = phrase_key
                    best_score = adjusted_score
                    best_new_words = total_new
                    best_info = {
                        "phrase": phrases[phrase_keys.index(phrase_key)],
                        "score": adjusted_score,
                        "new_story_verbs": len(new_story_verbs),
                        "new_story_vocab": len(new_story_vocab),
                        "new_global_verbs": len(new_global_verbs),
                        "new_global_vocab": len(new_global_vocab),
                        "total_new_words": total_new,
                    }

            # If no good phrase found, consider breaking out
            if best_score <= 0:
                print(f"No more relevant phrases found for {story}")
                break

            # Add best phrase to assignments
            assignments[story].append(best_info)
            remaining_phrases.remove(best_phrase)
            assigned_count += 1

            # Update vocabulary tracking
            phrase_verbs = set(phrase_index["phrase_vocab"][best_phrase]["verbs"])
            phrase_vocab = set(phrase_index["phrase_vocab"][best_phrase]["vocab"])

            # Update story coverage
            story_covered_verbs.update(phrase_verbs & story_verbs)
            story_covered_vocab.update(phrase_vocab & story_vocab)

            # Update global knowledge
            global_known_verbs.update(phrase_verbs)
            global_known_vocab.update(phrase_vocab)

            # Update remaining vocabulary to learn
            remaining_story_verbs = story_verbs - global_known_verbs
            remaining_story_vocab = story_vocab - global_known_vocab

        # Update story phrase count
        story_phrase_counts[story] = assigned_count

        # Calculate coverage statistics for this story
        # Calculate story-specific coverage statistics
        verb_coverage = (
            len(story_covered_verbs) / len(story_verbs) if story_verbs else 1.0
        )
        vocab_coverage = (
            len(story_covered_vocab) / len(story_vocab) if story_vocab else 1.0
        )

        # Calculate how much of this story's vocabulary is already covered by global knowledge
        story_verb_coverage_by_global = (
            len(story_verbs & global_known_verbs) / len(story_verbs)
            if story_verbs
            else 1.0
        )
        story_vocab_coverage_by_global = (
            len(story_vocab & global_known_vocab) / len(story_vocab)
            if story_vocab
            else 1.0
        )
        story_total_coverage_by_global = (
            (
                len(story_verbs & global_known_verbs)
                + len(story_vocab & global_known_vocab)
            )
            / (len(story_verbs) + len(story_vocab))
            if (story_verbs or story_vocab)
            else 1.0
        )

        # Print coverage statistics
        print(f"Assigned {assigned_count} phrases to {story}")
        print(f"Story verb coverage from this story's flashcards: {verb_coverage:.1%}")
        print(
            f"Story vocab coverage from this story's flashcards: {vocab_coverage:.1%}"
        )
        print(f"Coverage of this story by global knowledge:")
        print(
            f"  Story verb coverage by global knowledge: {story_verb_coverage_by_global:.1%} ({len(story_verbs & global_known_verbs)}/{len(story_verbs)})"
        )
        print(
            f"  Story vocab coverage by global knowledge: {story_vocab_coverage_by_global:.1%} ({len(story_vocab & global_known_vocab)}/{len(story_vocab)})"
        )
        print(
            f"  Story total coverage by global knowledge: {story_total_coverage_by_global:.1%} ({len(story_verbs & global_known_verbs) + len(story_vocab & global_known_vocab)}/{len(story_verbs) + len(story_vocab)})"
        )

    # After processing all stories, assign any remaining flashcards
    if remaining_phrases:
        print(f"\nAssigning {len(remaining_phrases)} remaining phrases")

        # For remaining phrases, assign to most relevant story
        for phrase_key in list(remaining_phrases):
            if phrase_key not in phrase_index["phrase_vocab"]:
                remaining_phrases.remove(phrase_key)
                continue

            best_story = None
            best_relevance = -1

            for story in story_sequence:
                # Skip stories that are already at max
                if (
                    story_phrase_counts[story] >= max_phrases_per_story * 1.2
                ):  # Allow some overflow
                    continue

                # Get vocabulary
                phrase_verbs = set(phrase_index["phrase_vocab"][phrase_key]["verbs"])
                phrase_vocab = set(phrase_index["phrase_vocab"][phrase_key]["vocab"])

                story_verbs = set(story_index["story_vocab"][story]["verbs"])
                story_vocab = set(story_index["story_vocab"][story]["vocab"])

                # Calculate relevance score
                verb_overlap = len(phrase_verbs & story_verbs)
                vocab_overlap = len(phrase_vocab & story_vocab)

                relevance = (verb_overlap * 2) + vocab_overlap

                if relevance > best_relevance:
                    best_story = story
                    best_relevance = relevance

            # If we found a relevant story, assign the phrase
            if best_story and best_relevance > 0:
                # Get phrase info
                phrase_verbs = set(phrase_index["phrase_vocab"][phrase_key]["verbs"])
                phrase_vocab = set(phrase_index["phrase_vocab"][phrase_key]["vocab"])

                # Calculate new words relative to global knowledge
                new_global_verbs = phrase_verbs - global_known_verbs
                new_global_vocab = phrase_vocab - global_known_vocab

                # Create info dictionary
                info = {
                    "phrase": phrases[phrase_keys.index(phrase_key)],
                    "score": best_relevance,
                    "new_story_verbs": 0,  # Not calculated for remaining assignments
                    "new_story_vocab": 0,
                    "new_global_verbs": len(new_global_verbs),
                    "new_global_vocab": len(new_global_vocab),
                    "total_new_words": len(new_global_verbs) + len(new_global_vocab),
                }

                # Add to assignments
                assignments[best_story].append(info)
                story_phrase_counts[best_story] += 1

                # Update global knowledge
                global_known_verbs.update(phrase_verbs)
                global_known_vocab.update(phrase_vocab)

                # Remove from remaining
                remaining_phrases.remove(phrase_key)

    # Print assignment statistics
    print("\nFinal Assignment Statistics:")
    total_assigned = sum(story_phrase_counts.values())
    print(f"Total phrases: {len(phrases)}")
    print(f"Total assigned: {total_assigned}")
    print(f"Remaining unassigned: {len(remaining_phrases)}")

    print("\nPhrases per story:")
    for story in story_sequence:
        if story in story_phrase_counts:
            print(f"{story}: {story_phrase_counts[story]} phrases")

            # Calculate average new words per flashcard for this story
            if story_phrase_counts[story] > 0:
                total_new_words = sum(
                    item["total_new_words"] for item in assignments[story]
                )
                avg_new_words = total_new_words / story_phrase_counts[story]
                print(f"  Average new words per flashcard: {avg_new_words:.2f}")

    return assignments


def plot_vocabulary_growth_from_assignments(assignments: Dict[str, List[Dict[str, Any]]], window: int = 15) -> None:
    """
    Plot vocabulary growth using the assignments dictionary format.
    Uses the 'total_new_words' key for each phrase.

    Args:
        assignments: Dictionary mapping story names to lists of phrase dicts
        window: Window size for rolling mean calculation
    """
    # Flatten all phrase dicts in story order
    all_phrase_dicts = []
    for story in assignments:
        all_phrase_dicts.extend(assignments[story])

    # Use total_new_words for each phrase
    new_words_per_phrase = [d.get("total_new_words", 0) for d in all_phrase_dicts]
    cumulative_counts = []
    cumulative = 0
    for n in new_words_per_phrase:
        cumulative += n
        cumulative_counts.append(cumulative)

    # Calculate rolling mean and overall mean
    df = pd.DataFrame({"new_words": new_words_per_phrase})
    rolling_mean = (
        df["new_words"].rolling(window=min(window, len(new_words_per_phrase)), min_periods=1).mean()
    )
    overall_mean = np.mean(new_words_per_phrase)

    # Create plot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add cumulative vocabulary trace
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(new_words_per_phrase) + 1)),
            y=cumulative_counts,
            name="Total Vocabulary (cumulative sum)",
            line=dict(color="blue"),
        ),
        secondary_y=False,
    )

    # Add rolling mean line
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(new_words_per_phrase) + 1)),
            y=rolling_mean,
            name="New Words per Phrase\n(rolling mean)",
            line=dict(color="red"),
        ),
        secondary_y=True,
    )

    # Add overall mean line
    fig.add_trace(
        go.Scatter(
            x=[1, len(new_words_per_phrase)],
            y=[overall_mean, overall_mean],
            name="Overall new words per phrase",
            line=dict(color="purple", dash="dot"),
        ),
        secondary_y=True,
    )

    # Update layout
    fig.update_layout(
        title="Vocabulary Growth Analysis (Assignments)",
        xaxis_title="Phrase Number",
        showlegend=True,
        margin=dict(r=150),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
    )

    # Update y-axes titles
    fig.update_yaxes(
        title_text="Total Unique Words (Cumulative)", secondary_y=False, gridcolor="lightgrey"
    )
    fig.update_yaxes(
        title_text="New Words per Phrase", secondary_y=True, gridcolor="lightgrey"
    )

    # Show plot
    fig.show()
