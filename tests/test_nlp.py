from src.nlp import extract_substring_matches, remove_matching_words
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
            {"what?", "time?"},
        ),
        # Multiple question marks
        (
            ["Really?? Are you sure???"],
            {"really??", "sure???"},
            {"really??", "sure???"},
        ),
        # Question mark in middle of target (although unusual)
        (
            ["Is this really? true"],
            {"really?", "really? true"},
            {"really?", "really? true"},
        ),
        # Empty and non-matching cases
        (["Hello world"], {"hello?", "world?"}, {"hello?", "world?"}),
        # Multiple punctuation in target
        (
            ["What's your name? Tell me!"],
            {"what's your name?", "tell"},
            {"what's you name?", "tell"},
        ),
    ],
)
def test_extract_substring_matches_question_marks(
    new_phrases, target_phrases, expected
):
    result = extract_substring_matches(new_phrases, target_phrases)
    assert result == expected
