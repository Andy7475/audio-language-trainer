from src.nlp import (
    extract_substring_matches,
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
