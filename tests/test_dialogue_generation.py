import json
import unittest
from typing import Dict, List, Set, Tuple
from unittest.mock import mock_open, patch, ANY

import pytest

from src.dialogue_generation import (
    GRAMMAR_USAGE_PATH,
    generate_dialogue_prompt,
    get_vocab_from_dialogue,
    select_grammar_concepts,
    update_grammar_concept_usage,
    update_vocab_usage,
)


@pytest.fixture
def mock_grammar_concepts():
    return {
        "verb_tenses": {
            "Simple Present": {"use": True, "times_seen": 0},
            "Present Continuous": {"use": True, "times_seen": 1},
            "Simple Past": {"use": False, "times_seen": 2},
        },
        "noun_forms": {
            "Singular": {"use": True, "times_seen": 0},
            "Plural": {"use": True, "times_seen": 3},
        },
    }


def test_select_grammar_concepts(mock_grammar_concepts):
    selected = select_grammar_concepts(mock_grammar_concepts, 2)
    assert len(selected) == 2
    assert "verb_tenses - Simple Present" in selected
    assert "noun_forms - Singular" in selected


def test_select_grammar_concepts_more_than_available(mock_grammar_concepts):
    selected = select_grammar_concepts(mock_grammar_concepts, 10)
    assert len(selected) == 4
    assert set(selected) == {
        "verb_tenses - Simple Present",
        "verb_tenses - Present Continuous",
        "noun_forms - Singular",
        "noun_forms - Plural",
    }


def test_update_grammar_concept_usage(mock_grammar_concepts):
    used_concepts = ["verb_tenses - Simple Present", "noun_forms - Plural"]

    with patch("builtins.open", mock_open()) as mock_file, patch(
        "json.dump"
    ) as mock_json_dump:
        update_grammar_concept_usage(mock_grammar_concepts, used_concepts)

    assert mock_grammar_concepts["verb_tenses"]["Simple Present"]["times_seen"] == 1
    assert mock_grammar_concepts["noun_forms"]["Plural"]["times_seen"] == 4
    assert mock_grammar_concepts["verb_tenses"]["Present Continuous"]["times_seen"] == 1

    mock_file.assert_called_once_with(GRAMMAR_USAGE_PATH, "w")
    mock_json_dump.assert_called_once_with(mock_grammar_concepts, ANY, indent=2)


@patch("src.dialogue_generation.load_json")
@patch("src.dialogue_generation.get_least_used_words")
@patch("src.dialogue_generation.select_grammar_concepts")
@patch("src.dialogue_generation.update_grammar_concept_usage")
def test_generate_dialogue_prompt(
    mock_update_usage, mock_select_concepts, mock_get_words, mock_load_json
):
    mock_load_json.return_value = {"mock": "grammar_concepts"}
    mock_get_words.side_effect = [
        ["run", "jump"],  # verbs
        ["apple", "banana"],  # vocab
    ]
    mock_select_concepts.return_value = [
        "verb_tenses - Simple Present",
        "noun_forms - Plural",
    ]

    prompt = generate_dialogue_prompt(
        "Exposition",
        "Introduce the characters",
        "Last time, Alex and Sam met at the library.",
        verb_count=2,
        verb_use_count=1,
        vocab_count=2,
        vocab_use_count=1,
        grammar_concept_count=2,
        grammar_use_count=2,
    )

    assert "run, jump" in prompt
    assert "apple, banana" in prompt
    assert "verb_tenses - Simple Present, noun_forms - Plural" in prompt
    assert "Exposition: Introduce the characters" in prompt
    assert "Last time, Alex and Sam met at the library." in prompt

    mock_load_json.assert_called_once_with(GRAMMAR_USAGE_PATH)
    mock_get_words.assert_has_calls(
        [unittest.mock.call("verbs", 2), unittest.mock.call("vocab", 2)]
    )
    mock_select_concepts.assert_called_once_with({"mock": "grammar_concepts"}, 2)
    mock_update_usage.assert_called_once_with(
        {"mock": "grammar_concepts"},
        ["verb_tenses - Simple Present", "noun_forms - Plural"],
    )


# Mock the Google Cloud clients
@pytest.fixture(autouse=True)
def mock_google_cloud():
    with patch("google.cloud.texttospeech.TextToSpeechClient"), patch(
        "google.cloud.translate_v2.Client"
    ):
        yield


@pytest.fixture
def sample_dialogue() -> List[Dict[str, str]]:
    return [
        {"speaker": "Alice", "text": "I am reading a book."},
        {"speaker": "Bob", "text": "That's great! I love reading too."},
        {"speaker": "Alice", "text": "It's about space exploration. Very interesting!"},
        {"speaker": "Bob", "text": "Wow! I read a book about Mars last month."},
    ]


@pytest.fixture
def expected_vocab() -> Set[Tuple[str, str]]:
    return {
        ("i", "PRON"),
        ("be", "AUX"),
        ("read", "VERB"),
        ("a", "DET"),
        ("book", "NOUN"),
        ("that", "PRON"),
        ("great", "ADJ"),
        ("love", "VERB"),
        ("too", "ADV"),
        ("it", "PRON"),
        ("about", "ADP"),
        ("space", "NOUN"),
        ("exploration", "NOUN"),
        ("very", "ADV"),
        ("interesting", "ADJ"),
        ("wow", "INTJ"),
        ("mars", "PROPN"),
        ("last", "ADJ"),
        ("month", "NOUN"),
    }


def test_get_vocab_from_dialogue(
    sample_dialogue: List[Dict[str, str]], expected_vocab: Set[Tuple[str, str]]
):
    result = get_vocab_from_dialogue(sample_dialogue)
    assert result == expected_vocab, f"Expected {expected_vocab}, but got {result}"


# Additional test cases
def test_empty_dialogue():
    assert (
        get_vocab_from_dialogue([]) == set()
    ), "Empty dialogue should return an empty set"


def test_dialogue_with_punctuation():
    dialogue = [{"speaker": "Alice", "text": "Hello, world! How are you?"}]
    expected = {
        ("hello", "INTJ"),
        ("world", "NOUN"),
        ("how", "SCONJ"),
        ("be", "AUX"),
        ("you", "PRON"),
    }
    assert (
        get_vocab_from_dialogue(dialogue) == expected
    ), "Punctuation should be ignored"


def test_dialogue_with_repeated_words():
    dialogue = [
        {"speaker": "Alice", "text": "The cat and the dog. The cat is sleeping."}
    ]
    expected = {
        ("the", "DET"),
        ("cat", "NOUN"),
        ("and", "CCONJ"),
        ("dog", "NOUN"),
        ("be", "AUX"),
        ("sleep", "VERB"),
    }
    assert (
        get_vocab_from_dialogue(dialogue) == expected
    ), "Repeated words should only appear once in the output"


def test_update_vocab_usage_existing_words():
    mock_vocab_usage = {"verbs": {"run": 1}, "vocab": {"fast": 2}}
    used_words = {("run", "VERB"), ("fast", "ADJ"), ("quick", "ADJ")}

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        with patch("json.dump") as mock_dump:
            update_vocab_usage(used_words)

    expected_vocab_usage = {"verbs": {"run": 2}, "vocab": {"fast": 3, "quick": 1}}
    mock_dump.assert_called_once_with(expected_vocab_usage, unittest.mock.ANY, indent=2)


def test_update_vocab_usage_new_words():
    mock_vocab_usage = {"verbs": {}, "vocab": {}}
    used_words = {("jump", "VERB"), ("high", "ADJ")}

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        with patch("json.dump") as mock_dump:
            update_vocab_usage(used_words)

    expected_vocab_usage = {"verbs": {"jump": 1}, "vocab": {"high": 1}}
    mock_dump.assert_called_once_with(expected_vocab_usage, unittest.mock.ANY, indent=2)


def test_update_vocab_usage_empty_set():
    mock_vocab_usage = {"verbs": {"exist": 1}, "vocab": {"test": 1}}
    used_words = set()

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        with patch("json.dump") as mock_dump:
            update_vocab_usage(used_words)

    expected_vocab_usage = {"verbs": {"exist": 1}, "vocab": {"test": 1}}
    mock_dump.assert_called_once_with(expected_vocab_usage, unittest.mock.ANY, indent=2)


def test_update_vocab_usage_non_verb_pos():
    mock_vocab_usage = {"verbs": {}, "vocab": {}}
    used_words = {("beautiful", "ADJ"), ("quickly", "ADV"), ("the", "DET")}

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        with patch("json.dump") as mock_dump:
            update_vocab_usage(used_words)

    expected_vocab_usage = {
        "verbs": {},
        "vocab": {"beautiful": 1, "quickly": 1, "the": 1},
    }
    mock_dump.assert_called_once_with(expected_vocab_usage, unittest.mock.ANY, indent=2)


def test_update_vocab_usage_file_not_found():
    used_words = {("test", "NOUN")}

    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            update_vocab_usage(used_words)


def test_update_vocab_usage_permission_error():
    used_words = {("test", "NOUN")}

    with patch("builtins.open", side_effect=PermissionError):
        with pytest.raises(PermissionError):
            update_vocab_usage(used_words)
