import json
import unittest
from typing import Dict, List, Set, Tuple
from unittest.mock import mock_open, patch, ANY
from src.dialogue_generation import add_usage_to_words, VOCAB_USAGE_PATH
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
@patch("src.dialogue_generation.select_grammar_concepts")
@patch("src.dialogue_generation.update_grammar_concept_usage")
def test_generate_dialogue_prompt(
    mock_update_usage, mock_select_concepts, mock_load_json
):
    mock_load_json.return_value = {"mock": "grammar_concepts"}
    mock_select_concepts.return_value = [
        "verb_tenses - Simple Present",
        "noun_forms - Plural",
    ]

    prompt = generate_dialogue_prompt(
        "Exposition",
        "Introduce the characters",
        "Last time, Alex and Sam met at the library.",
        verb_usage_str="""{'can' : 2, 'run' : 0, 'jump' : 0}""",
        verb_use_count=1,
        vocab_use_count=2,
        vocab_usage_str="""{'hello' : 1}""",
        grammar_concept_count=2,
        grammar_use_count=2,
    )

    assert "jump" in prompt
    assert "hello" in prompt
    assert "verb_tenses - Simple Present, noun_forms - Plural" in prompt
    assert "Exposition: Introduce the characters" in prompt
    assert "Last time, Alex and Sam met at the library." in prompt

    mock_load_json.assert_called_once_with(GRAMMAR_USAGE_PATH)
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
    used_words = {("run", "VERB"), ("fast", "ADJ"), ("quick", "ADJ"), ("can", "AUX")}

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        with patch("json.dump") as mock_dump:
            update_vocab_usage(used_words)

    expected_vocab_usage = {
        "verbs": {"run": 2, "can": 1},
        "vocab": {"fast": 3, "quick": 1},
    }
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


## add_usage_to_words test


@pytest.fixture
def mock_vocab_usage():
    return {
        "verbs": {"run": 2, "jump": 1, "swim": 3},
        "vocab": {"apple": 1, "banana": 2, "cherry": 0},
    }


def test_add_usage_to_words_verbs(mock_vocab_usage):
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        result = add_usage_to_words(["run", "jump", "swim"], "verbs")
    assert result == "{'jump': 1, 'run': 2, 'swim': 3}"


def test_add_usage_to_words_vocab(mock_vocab_usage):
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        result = add_usage_to_words(["apple", "banana", "cherry"], "vocab")
    assert result == "{'cherry': 0, 'apple': 1, 'banana': 2}"


def test_add_usage_to_words_missing_word(mock_vocab_usage):
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        result = add_usage_to_words(["run", "jump", "fly"], "verbs")
    assert result == "{'fly': 0, 'jump': 1, 'run': 2}"


def test_add_usage_to_words_invalid_category(mock_vocab_usage):
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        with pytest.raises(ValueError) as excinfo:
            add_usage_to_words(["word"], "adjectives")
    assert "Category 'adjectives' not found in vocabulary usage data" in str(
        excinfo.value
    )


def test_add_usage_to_words_empty_list(mock_vocab_usage):
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        result = add_usage_to_words([], "verbs")
    assert result == "{}"


import pytest
from unittest.mock import patch, mock_open, call
import json
from src.dialogue_generation import get_least_used_words, VOCAB_USAGE_PATH


@pytest.mark.parametrize(
    "category, count, expected_pos",
    [("verbs", 2, "VERB"), ("vocab", 3, "vocab")],
)
def test_get_least_used_words(category, count, expected_pos):
    mock_vocab_usage = {
        "verbs": {"run": 2, "jump": 0, "swim": 1, "can": 0},
        "vocab": {"apple": 1, "banana": 0, "cherry": 3, "date": 2},
    }

    if category == "verbs":
        mock_selected_words = ["jump", "swim"]
    else:
        mock_selected_words = ["banana", "apple", "date"]

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        with patch("random.choices", return_value=mock_selected_words):
            with patch(
                "src.dialogue_generation.update_vocab_usage"
            ) as mock_update_vocab_usage:
                result = get_least_used_words(category, count)

                # Check if the correct words were returned
                assert result == mock_selected_words

                expected_words_with_pos = [
                    (word, expected_pos) for word in mock_selected_words
                ]

                mock_update_vocab_usage.assert_called_once_with(expected_words_with_pos)

                # Check if the correct number of words were selected
                assert len(result) == count

                # Check if the selected words are in the original vocab_usage
                assert all(word in mock_vocab_usage[category] for word in result)

    # Additional checks for weight calculation
    words = list(mock_vocab_usage[category].keys())
    usages = list(mock_vocab_usage[category].values())
    weights = [1 / (usage + 1) for usage in usages]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Check if the words with lower usage have higher normalized weights
    sorted_word_weights = sorted(
        zip(words, normalized_weights), key=lambda x: x[1], reverse=True
    )
    assert sorted_word_weights[0][0] in [
        "jump",
        "can",
        "banana",
    ]  # The least used word should have the highest weight


def test_update_vocab_usage_with_aux_and_verb():
    # Initial vocab usage
    initial_vocab_usage = {
        "verbs": {"run": 1, "jump": 0},
        "vocab": {"cat": 1, "dog": 2},
    }

    # Simulated response from get_vocab_from_dialogue
    words_with_pos = [
        ("cat", "NOUN"),
        ("is", "AUX"),
        ("running", "VERB"),
        ("and", "CCONJ"),
        ("can", "AUX"),
        ("jump", "VERB"),
    ]

    # Expected updated vocab usage
    expected_vocab_usage = {
        "verbs": {"run": 1, "jump": 1, "is": 1, "can": 1, "running": 1},
        "vocab": {"cat": 2, "dog": 2, "and": 1},
    }

    # Mock file operations
    mock_file_content = json.dumps(initial_vocab_usage)
    with patch("builtins.open", mock_open(read_data=mock_file_content)) as mock_file:
        with patch("json.dump") as mock_json_dump:
            # Call the function we're testing
            update_vocab_usage(words_with_pos)

            # Check if the file was opened for writing
            mock_file.assert_called_with(VOCAB_USAGE_PATH, "w")

            # Check if json.dump was called with the correct updated vocab_usage
            mock_json_dump.assert_called_once()
            actual_updated_vocab_usage = mock_json_dump.call_args[0][0]

            # Check if the vocab usage was updated correctly
            assert actual_updated_vocab_usage == expected_vocab_usage

            # Specific checks for AUX and VERB updates
            assert "is" in actual_updated_vocab_usage["verbs"]
            assert "can" in actual_updated_vocab_usage["verbs"]
            assert "running" in actual_updated_vocab_usage["verbs"]
            assert "jump" in actual_updated_vocab_usage["verbs"]

            # Check that non-verb words are in the vocab section
            assert "and" in actual_updated_vocab_usage["vocab"]
            assert "cat" in actual_updated_vocab_usage["vocab"]


def test_get_least_used_words():
    # Predefined vocab usage
    mock_vocab_usage = {
        "verbs": {"run": 2, "jump": 0, "swim": 1},
        "vocab": {"apple": 1, "banana": 0, "cherry": 3},
    }

    # Mock file content
    mock_file_content = json.dumps(mock_vocab_usage)

    # Predefined selected words
    mock_selected_words = ["jump", "swim"]

    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        with patch("random.choices", return_value=mock_selected_words):
            with patch(
                "src.dialogue_generation.update_vocab_usage"
            ) as mock_update_vocab_usage:
                # Call the function we're testing
                result = get_least_used_words("verbs", 2)

                # Check if the correct words were returned
                assert result == mock_selected_words

                # Check if update_vocab_usage was called with the correct arguments
                mock_update_vocab_usage.assert_called_once_with(
                    [("jump", "VERB"), ("swim", "VERB")]
                )

    # Test for vocab category
    mock_selected_words = ["banana", "apple"]

    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        with patch("random.choices", return_value=mock_selected_words):
            with patch(
                "src.dialogue_generation.update_vocab_usage"
            ) as mock_update_vocab_usage:
                # Call the function we're testing
                result = get_least_used_words("vocab", 2)

                # Check if the correct words were returned
                assert result == mock_selected_words

                # Check if update_vocab_usage was called with the correct arguments
                mock_update_vocab_usage.assert_called_once_with(
                    [("banana", "vocab"), ("apple", "vocab")]
                )


def test_get_vocab_from_dialogue():
    dialogue = [
        {"speaker": "Sam", "text": "Hello, I'm Sam. I live in York."},
        {"speaker": "Alex", "text": "Nice to meet you, Sam! I'm Alex from London."},
        {"speaker": "Sam", "text": "Do you know John? He's also from London."},
    ]

    vocab = get_vocab_from_dialogue(dialogue)

    # Check that 'Sam', 'Alex', and 'John' are not in the vocabulary
    assert not any(
        word for word, pos in vocab if word.lower() in ["sam", "alex", "john"]
    )

    # Check that other words are included
    assert ("hello", "INTJ") in vocab
    assert ("live", "VERB") in vocab
    assert ("london", "PROPN") in vocab

    # Check that punctuation is excluded
    assert not any(pos for word, pos in vocab if pos == "PUNCT")

    print("Extracted vocabulary:", vocab)
