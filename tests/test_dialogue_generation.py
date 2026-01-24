from typing import Dict

import pytest

from src.nlp import get_vocab_dict_from_dialogue


@pytest.fixture
def sample_story_dict() -> Dict:
    return {
        "introduction": {
            "dialogue": [
                {"speaker": "Alice", "text": "I am reading a book."},
                {"speaker": "Bob", "text": "That's great! I love reading too."},
                {
                    "speaker": "Alice",
                    "text": "It's about space exploration. Very interesting!",
                },
                {"speaker": "Bob", "text": "Wow! I read a book about Mars last month."},
            ]
        }
    }


@pytest.mark.parametrize(
    "story_dict, expected_output",
    [
        # Empty dialogue
        ({"part1": {"dialogue": []}}, {"verbs": [], "vocab": []}),
        # Simple dialogue with punctuation
        (
            {
                "part1": {
                    "dialogue": [
                        {"speaker": "Alice", "text": "Hello, world! How are you?"}
                    ]
                }
            },
            {"verbs": ["be"], "vocab": ["hello", "world", "how", "you"]},
        ),
        # Dialogue with repeated words
        (
            {
                "part1": {
                    "dialogue": [
                        {
                            "speaker": "Alice",
                            "text": "The cat and the dog. The cat is sleeping.",
                        }
                    ]
                }
            },
            {"verbs": ["sleep", "be"], "vocab": ["the", "cat", "and", "dog"]},
        ),
        # Dialogue with names to exclude
        (
            {
                "part1": {
                    "dialogue": [
                        {"speaker": "Sam", "text": "Hello Alex, I live in York."},
                        {"speaker": "Alex", "text": "Nice to meet you, Sam!"},
                    ]
                }
            },
            {
                "verbs": ["live", "meet"],
                "vocab": ["hello", "i", "in", "york", "nice", "to", "you"],
            },
        ),
    ],
)
def test_get_vocab_dict_from_dialogue(story_dict, expected_output):
    result = get_vocab_dict_from_dialogue(story_dict)
    assert set(result["verbs"]) == set(expected_output["verbs"])
    assert set(result["vocab"]) == set(expected_output["vocab"])


def test_get_vocab_dict_with_limit_parts():
    story_dict = {
        "part1": {"dialogue": [{"speaker": "Alice", "text": "I run fast."}]},
        "part2": {"dialogue": [{"speaker": "Bob", "text": "I swim well."}]},
    }

    result = get_vocab_dict_from_dialogue(story_dict, limit_story_parts=["part1"])
    assert set(result["verbs"]) == {"run"}
    assert set(result["vocab"]) == {"i", "fast"}


def test_missing_story_part():
    story_dict = {"part1": {"dialogue": []}}

    with pytest.raises(KeyError):
        get_vocab_dict_from_dialogue(story_dict, limit_story_parts=["missing_part"])
