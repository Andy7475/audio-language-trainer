from src.nlp import (
    extract_substring_matches,
    find_best_card,
    find_candidate_cards,
    get_matching_flashcards_indexed,
    get_vocab_dictionary_from_phrases,
    remove_matching_words,
)
import pytest


@pytest.mark.parametrize(
    "phrases, original_set, expected",
    [
        (
            ["falling"],  # remove one matching item
            {"falling (over)", "running (fast)", "jumping"},
            {"running (fast)", "jumping"},
        ),
        (
            ["falling", "jumping"],
            {"falling (over)", "running (fast)", "jumping"},
            {"running (fast)"},
        ),
        (
            ["walking"],  # Word not in original set
            {"falling (over)", "running (fast)", "jumping"},
            {"falling (over)", "running (fast)", "jumping"},
        ),
        (
            [],  # Empty phrases list
            {"falling (over)", "running (fast)", "jumping"},
            {"falling (over)", "running (fast)", "jumping"},
        ),
        (
            ["falling", "running", "jumping"],  # Remove all
            {"falling (over)", "running (fast)", "jumping"},
            set(),
        ),
        (
            ["falling"],  # it will remove both
            {"falling (over)", "falling (over2)"},
            set(),
        ),
        (
            ["falling"],  # it will remove both no matter where () are
            {"(over) falling", "falling (over)"},
            set(),
        ),
        (
            ["two words"],  # it will remove both no matter where () are
            {
                "two words",
                "(a) two words",
                "two words (b)",
                "two (c) words",
                "two (d e) words",
                "two",
                "words",
            },
            {"two", "words"},
        ),
        (
            ["what's the time?"],  # it will remove both no matter where () are
            {"what's the time?", "(asking) what's the time?"},
            set(),
        ),
    ],
)
def test_remove_matching_words(phrases, original_set, expected):

    result = remove_matching_words(phrases, original_set)
    assert result == expected


@pytest.mark.parametrize(
    "new_phrases, target_phrases, expected",
    [
        # Basic question mark handling
        (
            ["What is your name?", "How are you?"],
            {"What is your name?", "your name", "how"},
            {"what is your name?", "your name", "how"},
        ),
        # Mixed punctuation
        (
            ["What's your name? How are you doing?"],
            {"what's your", "you doing?"},
            {"what's your", "you doing?"},
        ),
        # Question words without question marks shouldn't match those with
        (
            ["What is the time? Do you know?"],
            {"what?", "what", "time?"},
            {"what", "time?"},
        ),
        # Question mark in middle of target (although unusual)
        (
            ["Is this really? true"],
            {"really?", "really? true"},
            {"really?", "really? true"},
        ),
        # Empty and non-matching cases
        (["Hello world"], {"hello?", "world?"}, set()),
        # Multiple punctuation in target
        (
            ["What's your name? Tell me!"],
            {"what's your name?", "tell"},
            {"what's your name?", "tell"},
        ),
    ],
)
def test_extract_substring_matches_question_marks(
    new_phrases, target_phrases, expected
):
    result = extract_substring_matches(new_phrases, target_phrases)
    assert result == expected


import pytest
from typing import Dict, List


@pytest.mark.parametrize(
    "phrases, expected_dict",
    [
        # Simple case: single verb and noun
        (["I run home"], {"verbs": ["run"], "vocab": ["i", "home"]}),
        # Multiple phrases with various parts of speech
        (
            ["The black cat sleeps quietly", "She drinks coffee quickly"],
            {
                "verbs": ["sleep", "drink"],
                "vocab": ["the", "black", "cat", "quietly", "she", "coffee", "quickly"],
            },
        ),
        # Phrases with auxiliary verbs and adjectives
        (
            ["I am trying to learn", "The beautiful flowers are blooming"],
            {
                "verbs": ["be", "try", "learn", "bloom"],
                "vocab": ["i", "to", "the", "beautiful", "flower"],
            },
        ),
        # Edge case: no verbs
        (
            ["Hello there!", "Good morning"],
            {"verbs": [], "vocab": ["hello", "there", "good", "morning"]},
        ),
        # Case with repetitions (should only appear once in output)
        (
            ["I run fast", "She runs quickly", "They are running"],
            {"verbs": ["run", "be"], "vocab": ["i", "fast", "she", "quickly", "they"]},
        ),
    ],
)
def test_get_vocab_dictionary_from_phrases(
    phrases: List[str], expected_dict: Dict[str, List[str]]
):
    result = get_vocab_dictionary_from_phrases(phrases)

    # Check that we have the expected keys
    assert sorted(result.keys()) == sorted(expected_dict.keys())

    # Check that each list contains exactly the expected items (order doesn't matter)
    assert sorted(result["verbs"]) == sorted(expected_dict["verbs"])
    assert sorted(result["vocab"]) == sorted(expected_dict["vocab"])


import pytest
from src.nlp import create_flashcard_index, process_phrase_vocabulary


@pytest.mark.parametrize(
    "phrase,expected_verb_count,expected_vocab_count",
    [
        ("The cat runs", 1, 2),  # Simple present
        ("I am running", 2, 1),  # Present continuous with auxiliary
        ("The big cat", 0, 3),  # No verbs
        ("They have been sleeping", 3, 1),  # Perfect continuous
    ],
)
def test_process_phrase_vocabulary(phrase, expected_verb_count, expected_vocab_count):
    """Test vocabulary extraction from individual phrases"""
    vocab_used, verb_matches, vocab_matches = process_phrase_vocabulary(phrase)

    assert len(verb_matches) == expected_verb_count
    assert len(vocab_matches) == expected_vocab_count
    assert len(vocab_used) == expected_verb_count + expected_vocab_count


@pytest.mark.parametrize(
    "phrases,search_word,expected_indices",
    [
        # Test verb 'run' appearing in different positions
        (["The cat runs", "Dogs sleep", "Birds fly"], "run", [0]),
        (["The cat walks", "The dog runs", "Birds fly"], "run", [1]),
        (["The cat walks", "Dogs sleep", "He runs fast"], "run", [2]),
        # Test noun 'cat' appearing in multiple phrases
        (["The cat runs", "Another cat walks", "A third cat sleeps"], "cat", [0, 1, 2]),
        # Test word not present
        (["The cat runs", "Dogs sleep", "Birds fly"], "jump", []),
    ],
)
def test_flashcard_index(phrases, search_word, expected_indices):
    """Test the full indexing functionality with real spaCy processing"""
    result = create_flashcard_index(phrases)

    # The word could be in either verb_index or vocab_index
    found_indices = result["verb_index"].get(search_word, []) + result[
        "vocab_index"
    ].get(search_word, [])

    assert sorted(found_indices) == sorted(expected_indices)
    assert len(result["word_counts"]) == len(phrases)
    assert result["phrases"] == phrases


def test_complex_example():
    """Test a more complex example with multiple types of matches"""
    phrases = [
        "The big cat runs quickly",
        "Two cats are sleeping peacefully",
        "A small dog and a big cat play together",
        "Birds fly in the blue sky",
    ]

    result = create_flashcard_index(phrases)

    # Test various aspects of the indexing
    assert len(result["word_counts"]) == 4
    assert "cat" in result["vocab_index"]
    assert set(result["vocab_index"]["cat"]) == {
        0,
        1,
        2,
    }  # 'cat' appears in first, second, and third phrases
    assert "run" in result["verb_index"]
    assert set(result["verb_index"]["run"]) == {0}  # 'run' appears in first phrase

    # Check word counts for first phrase
    assert result["word_counts"][0]["verb_count"] == 1  # 'runs'
    assert result["word_counts"][0]["vocab_count"] >= 3  # 'big', 'cat', 'quickly'


import pytest


def test_find_candidate_cards():
    # Create simple test index
    flashcard_index = {
        "verb_index": {"run": [0, 2], "cry": [1]},
        "vocab_index": {"mum": [0, 1], "fast": [2]},
    }

    remaining_verbs = {"run", "cry"}
    remaining_vocab = {"mum"}

    candidates = find_candidate_cards(remaining_verbs, remaining_vocab, flashcard_index)
    assert candidates == {0, 1, 2}  # Should find all cards containing run, cry or mum


def test_find_best_card():
    flashcard_index = {
        "word_counts": [
            {"words": [("run", "VERB"), ("mum", "NOUN")]},  # card 0
            {"words": [("cry", "VERB"), ("mum", "NOUN")]},  # card 1
            {"words": [("run", "VERB"), ("fast", "ADJ")]},  # card 2
        ]
    }

    remaining_verbs = {"run", "cry"}
    remaining_vocab = {"mum", "fast"}
    candidates = {0, 1, 2}

    best_idx, best_matches = find_best_card(
        candidates, remaining_verbs, remaining_vocab, flashcard_index
    )

    assert best_idx in (0, 1)  # Should pick card with run+mum or cry+mum (2 matches)
    assert len(best_matches["verbs"]) + len(best_matches["vocab"]) == 2


@pytest.mark.parametrize(
    "vocab_dict,expected_matches",
    [
        # Simple match
        ({"verbs": ["run"], "vocab": ["fast"]}, {"verbs": {"run"}, "vocab": {"fast"}}),
        # Multiple words match single card
        (
            {"verbs": ["run", "cry"], "vocab": ["mum"]},
            {"verbs": {"run", "cry"}, "vocab": {"mum"}},
        ),
        # No matches
        ({"verbs": ["jump"], "vocab": ["slowly"]}, {"verbs": set(), "vocab": set()}),
    ],
)
def test_get_matching_flashcards(vocab_dict, expected_matches):
    # Create test index with sample phrases
    flashcard_index = {
        "phrases": ["He ran crying to his mum", "The boy cried to mum", "Run fast"],
        "verb_index": {"run": [0, 2], "cry": [0, 1]},
        "vocab_index": {"mum": [0, 1], "fast": [2]},
        "word_counts": [
            {"words": [("run", "VERB"), ("cry", "VERB"), ("mum", "NOUN")]},
            {"words": [("cry", "VERB"), ("mum", "NOUN")]},
            {"words": [("run", "VERB"), ("fast", "ADJ")]},
        ],
    }

    result = get_matching_flashcards_indexed(vocab_dict, flashcard_index)

    # Check matches were found or not
    all_verb_matches = {
        match for card in result["selected_cards"] for match in card["verb_matches"]
    }
    all_vocab_matches = {
        match for card in result["selected_cards"] for match in card["vocab_matches"]
    }

    assert all_verb_matches == expected_matches["verbs"]
    assert all_vocab_matches == expected_matches["vocab"]

    # Check remaining words are those that couldn't be matched
    assert (
        result["remaining_vocab"]["verbs"]
        == set(vocab_dict["verbs"]) - all_verb_matches
    )
    assert (
        result["remaining_vocab"]["vocab"]
        == set(vocab_dict["vocab"]) - all_vocab_matches
    )
