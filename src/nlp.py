import re
import subprocess
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spacy
from plotly.subplots import make_subplots
from spacy.tokens import Token
from tqdm import tqdm


def plot_vocabulary_growth(phrases: List[str], window: int = 10) -> None:
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

    # Add new words per phrase bars
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(phrases) + 1)),
            y=new_words_per_phrase,
            name="New Words per Phrase",
            opacity=0.3,
            marker_color="green",
        ),
        secondary_y=True,
    )

    # Add rolling mean line
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(phrases) + 1)),
            y=rolling_mean,
            name=f"Rolling Mean (window={min(window, len(phrases))})",
            line=dict(color="red"),
        ),
        secondary_y=True,
    )

    # Add overall mean line
    fig.add_trace(
        go.Scatter(
            x=[1, len(phrases)],
            y=[overall_mean, overall_mean],
            name="Overall Mean",
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
    """Create indexes mapping words to the flashcards containing them."""
    verb_index = {}  # word -> set of flashcard indices
    vocab_index = {}
    flashcard_word_counts = []

    for idx, phrase in tqdm(
        enumerate(flashcard_phrases),
        desc="Indexes phrases...",
        total=len(flashcard_phrases),
    ):
        vocab_used, verb_matches, vocab_matches = process_phrase_vocabulary(phrase)

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
    }


def find_candidate_cards(
    remaining_verbs: set, remaining_vocab: set, flashcard_index: dict
) -> set:
    """Find all cards that contain any remaining words"""
    candidate_cards = set()

    # Get all cards containing remaining verbs
    for word in remaining_verbs:
        if word in flashcard_index["verb_index"]:
            candidate_cards.update(flashcard_index["verb_index"][word])

    # Get all cards containing remaining vocab
    for word in remaining_vocab:
        if word in flashcard_index["vocab_index"]:
            candidate_cards.update(flashcard_index["vocab_index"][word])

    return candidate_cards


def find_best_card(
    candidate_cards: set,
    remaining_verbs: set,
    remaining_vocab: set,
    flashcard_index: dict,
) -> tuple[int, dict]:
    """Find card with most matches from candidates

    Returns:
        tuple of (best_card_index, match_info)
        where match_info contains verb and vocab matches
    """
    best_card_idx = None
    best_score = 0
    best_matches = None

    for card_idx in candidate_cards:
        card_words = flashcard_index["word_counts"][card_idx]["words"]

        # Count matches with remaining words
        verb_matches = {
            word
            for word, pos in card_words
            if pos in ["VERB", "AUX"] and word in remaining_verbs
        }
        vocab_matches = {
            word
            for word, pos in card_words
            if pos not in ["VERB", "AUX"] and word in remaining_vocab
        }

        score = len(verb_matches) + len(vocab_matches)
        if score > best_score:
            best_score = score
            best_card_idx = card_idx
            best_matches = {"verbs": verb_matches, "vocab": vocab_matches}

    return best_card_idx, best_matches


def get_matching_flashcards_indexed(vocab_dict: dict, flashcard_index: dict) -> dict:
    """Find minimal set of flashcards that cover the vocabulary.

    Prioritizes cards that contain the most uncovered words to minimize
    the total number of cards needed.

    Args:
        vocab_dict: Dictionary with 'verbs' and 'vocab' lists to cover
        flashcard_index: Pre-computed index mapping words to flashcard indices

    Returns:
        Dictionary containing:
        - selected_cards: List of selected flashcards with match info
        - remaining_vocab: Words not found in any flashcard
    """
    remaining_verbs = set(vocab_dict["verbs"])
    remaining_vocab = set(vocab_dict["vocab"])
    selected_cards = []

    while remaining_verbs or remaining_vocab:
        # Find all cards containing any remaining words
        candidate_cards = find_candidate_cards(
            remaining_verbs, remaining_vocab, flashcard_index
        )
        if not candidate_cards:
            break

        # Find card with most matches
        best_card_idx, best_matches = find_best_card(
            candidate_cards, remaining_verbs, remaining_vocab, flashcard_index
        )
        if not best_matches:
            break

        # Add best card to selection
        selected_cards.append(
            {
                "phrase": flashcard_index["phrases"][best_card_idx],
                "new_matches": len(best_matches["verbs"]) + len(best_matches["vocab"]),
                "verb_matches": best_matches["verbs"],
                "vocab_matches": best_matches["vocab"],
            }
        )

        # Remove covered words
        remaining_verbs -= best_matches["verbs"]
        remaining_vocab -= best_matches["vocab"]

    return {
        "selected_cards": selected_cards,
        "remaining_vocab": {"verbs": remaining_verbs, "vocab": remaining_vocab},
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


def extract_content_words(phrase: str, nlp) -> Set[Tuple[str, str]]:
    """
    Extract content words (verbs and meaningful vocabulary) from a phrase.
    Returns set of (lemma, pos) tuples.
    """
    doc = nlp(phrase.lower())

    # Define parts of speech to exclude
    exclude_pos = {
        "DET",
        "PUNCT",
        "SPACE",
        "PART",
        "CCONJ",
        "SCONJ",
        "ADP",
        "PRON",
        "PROPN",
    }

    content_words = set()
    for token in doc:
        # Only include if:
        # 1. Not in excluded POS tags
        # 2. Not a stop word (unless it's a verb)
        if token.pos_ not in exclude_pos and (
            not token.is_stop or token.pos_ == "VERB"
        ):
            content_words.add((token.lemma_.lower(), token.pos_))

    return content_words


def check_vocab_match(
    phrase_words: Set[Tuple[str, str]], vocab_dict: Dict[str, List[str]]
) -> bool:
    """
    Check if the content words from the phrase match the vocabulary dictionary.
    Returns True if all content words are found in the vocabulary lists.
    """
    for lemma, pos in phrase_words:
        if pos in ["VERB", "AUX"]:
            if lemma not in vocab_dict.get("verbs", []):
                return False
        else:
            if lemma not in vocab_dict.get("vocab", []):
                return False
    return True


def phrase_matches_vocab(
    english_phrase: str, vocab_dictionary: Dict[str, List[str]]
) -> bool:
    """
    Check if a phrase only uses words from the provided vocabulary dictionary.

    Args:
        english_phrase: The English phrase to check
        vocab_dictionary: Dictionary with 'verbs' and 'vocab' lists

    Returns:
        bool: True if all content words in the phrase are found in the vocabulary lists
    """
    nlp = load_spacy_model()

    # Convert vocabulary lists to lowercase for matching
    vocab_dict = {
        "verbs": [v.lower() for v in vocab_dictionary.get("verbs", [])],
        "vocab": [v.lower() for v in vocab_dictionary.get("vocab", [])],
    }

    # Extract content words from the phrase
    phrase_words = extract_content_words(english_phrase, nlp)

    # Check if all content words are in the vocabulary
    return check_vocab_match(phrase_words, vocab_dict)


def filter_matching_phrases(
    phrases: List[str], vocab_dictionary: Dict[str, List[str]]
) -> List[str]:
    """
    Filter a list of phrases to only include those that match the vocabulary.

    Args:
        phrases: List of English phrases to check
        vocab_dictionary: Dictionary with 'verbs' and 'vocab' lists

    Returns:
        List[str]: Filtered list of phrases that only use words from the vocabulary
    """
    nlp = load_spacy_model()

    # Convert vocabulary lists to lowercase for matching
    vocab_dict = {
        "verbs": [v.lower() for v in vocab_dictionary.get("verbs", [])],
        "vocab": [v.lower() for v in vocab_dictionary.get("vocab", [])],
    }

    matching_phrases = []
    for phrase in phrases:
        phrase_words = extract_content_words(phrase, nlp)
        if check_vocab_match(phrase_words, vocab_dict):
            matching_phrases.append(phrase)

    return matching_phrases


def prepare_phrase_dataframe(phrases: List[str]) -> pd.DataFrame:
    """Create DataFrame with parsed phrases and extract content words.

    Args:
        phrases: List of phrases to analyze

    Returns:
        DataFrame with columns:
            - phrase: Original phrase
            - doc: Spacy Doc object
            - content_words: Set of lemmatized content words
    """
    nlp = spacy.load("en_core_web_sm")

    # Create base dataframe
    df = pd.DataFrame({"phrase": phrases})

    # Parse phrases
    df["doc"] = [
        nlp(phrase.lower()) for phrase in tqdm(phrases, desc="Parsing phrases")
    ]

    # Extract content words
    df["content_words"] = df["doc"].apply(
        lambda doc: {
            token.lemma_ for token in doc if not token.is_stop and token.pos_ != "PUNCT"
        }
    )

    # Calculate total words per phrase
    df["total_words"] = df["content_words"].apply(len)

    return df


def calculate_new_words(row: pd.Series, known_vocab: Set[str]) -> Dict:
    """Calculate new vocabulary metrics for a row."""
    new_words = row["content_words"] - known_vocab
    return {
        "new_words": len(new_words),
        "new_vocab": new_words,
        "new_ratio": (
            len(new_words) / row["total_words"] if row["total_words"] > 0 else 0
        ),
    }


def optimize_sequence(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Optimize the sequence of phrases for steady vocabulary acquisition.

    Args:
        df: DataFrame with parsed phrases (from prepare_phrase_dataframe)
        window_size: Size of rolling window for local optimization

    Returns:
        DataFrame with optimized sequence and metrics
    """
    # Calculate ideal rate
    total_vocab = set().union(*df["content_words"].values)
    ideal_rate = len(total_vocab) / len(df)

    # Initialize result storage
    result_indices = []
    known_vocab = set()
    available_indices = set(df.index)

    # Choose initial phrase closest to ideal rate
    initial_metrics = df.apply(
        lambda row: calculate_new_words(row, known_vocab), axis=1
    ).apply(pd.Series)

    best_start_idx = (initial_metrics["new_ratio"] - ideal_rate).abs().idxmin()
    result_indices.append(best_start_idx)
    available_indices.remove(best_start_idx)
    known_vocab.update(df.loc[best_start_idx, "content_words"])

    # Build rest of sequence
    with tqdm(total=len(df) - 1, desc="Optimizing sequence") as pbar:
        while available_indices:
            # Calculate rolling mean of recent phrases
            if len(result_indices) >= window_size:
                recent_indices = result_indices[-window_size:]
                recent_new_words = np.mean(
                    [
                        len(df.loc[idx, "content_words"] - known_vocab)
                        for idx in recent_indices
                    ]
                )
            else:
                recent_new_words = ideal_rate

            # Score remaining phrases
            candidates = []
            for idx in available_indices:
                metrics = calculate_new_words(df.loc[idx], known_vocab)
                score = abs(metrics["new_words"] - ideal_rate)
                # Add penalty for big deviations from recent average
                local_penalty = abs(metrics["new_words"] - recent_new_words)
                score += local_penalty * 0.5  # Weight local smoothness
                candidates.append((idx, metrics, score))

            # Select best candidate
            best_idx = min(candidates, key=lambda x: x[2])[0]
            result_indices.append(best_idx)
            available_indices.remove(best_idx)
            known_vocab.update(df.loc[best_idx, "content_words"])
            pbar.update(1)

    # Create result DataFrame
    result_df = df.loc[result_indices].copy()
    result_df["sequence_position"] = range(len(result_df))

    # Calculate final metrics
    metrics = []
    known = set()
    for _, row in result_df.iterrows():
        m = calculate_new_words(row, known)
        metrics.append(m)
        known.update(row["content_words"])

    metrics_df = pd.DataFrame(metrics)
    result_df = pd.concat([result_df, metrics_df], axis=1)

    return result_df


def analyze_sequence(df: pd.DataFrame) -> Dict:
    """Analyze and print statistics about the optimized sequence."""
    stats = {
        "avg_new_words": df["new_words"].mean(),
        "std_new_words": df["new_words"].std(),
        "min_new_words": df["new_words"].min(),
        "max_new_words": df["new_words"].max(),
        "total_phrases": len(df),
        "cumulative_vocab": len(set().union(*df["content_words"].values)),
    }

    print("\nSequence Analysis:")
    print(f"Total phrases: {stats['total_phrases']}")
    print(f"Total vocabulary: {stats['cumulative_vocab']}")
    print(f"Average new words per phrase: {stats['avg_new_words']:.2f}")
    print(f"Standard deviation: {stats['std_new_words']:.2f}")
    print(f"Min new words: {stats['min_new_words']}")
    print(f"Max new words: {stats['max_new_words']}")

    return stats


def optimize_vocab_sequence(
    phrases: List[str], window_size: int = 5
) -> Tuple[pd.DataFrame, Dict]:
    """Main function to optimize phrase sequence for vocabulary acquisition.

    Args:
        phrases: List of phrases to optimize
        window_size: Size of rolling window for local optimization

    Returns:
        Tuple of (optimized DataFrame, statistics dictionary)
    """
    # Prepare data
    df = prepare_phrase_dataframe(phrases)

    # Optimize sequence
    optimized_df = optimize_sequence(df, window_size)

    # Analyze results
    stats = analyze_sequence(optimized_df)

    return optimized_df, stats
