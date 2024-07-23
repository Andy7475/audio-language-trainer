import unittest
from typing import Dict, List, Set, Tuple
from unittest.mock import mock_open, patch
import json
import pytest

from src.dialogue_generation import (get_current_story_part, get_last_recap,
                                     get_story_recap, get_vocab_from_dialogue,
                                     load_recaps, load_story_plan, update_vocab_usage)

# Mock the Google Cloud clients
@pytest.fixture(autouse=True)
def mock_google_cloud():
    with patch('google.cloud.texttospeech.TextToSpeechClient'), \
         patch('google.cloud.translate_v2.Client'):
        yield

@pytest.fixture
def sample_dialogue() -> List[Dict[str, str]]:
    return [
        {'speaker': 'Alice', 'text': "I am reading a book."},
        {'speaker': 'Bob', 'text': "That's great! I love reading too."},
        {'speaker': 'Alice', 'text': "It's about space exploration. Very interesting!"},
        {'speaker': 'Bob', 'text': "Wow! I read a book about Mars last month."}
    ]

@pytest.fixture
def expected_vocab() -> Set[Tuple[str, str]]:
    return {
        ('i', 'PRON'),
        ('be', 'AUX'),
        ('read', 'VERB'),
        ('a', 'DET'),
        ('book', 'NOUN'),
        ('that', 'PRON'),
        ('great', 'ADJ'),
        ('love', 'VERB'),
        ('too', 'ADV'),
        ('it', 'PRON'),
        ('about', 'ADP'),
        ('space', 'NOUN'),
        ('exploration', 'NOUN'),
        ('very', 'ADV'),
        ('interesting', 'ADJ'),
        ('wow', 'INTJ'),
        ('mars', 'PROPN'),
        ('last', 'ADJ'),
        ('month', 'NOUN')
    }

def test_get_vocab_from_dialogue(sample_dialogue: List[Dict[str, str]], expected_vocab: Set[Tuple[str, str]]):
    result = get_vocab_from_dialogue(sample_dialogue)
    assert result == expected_vocab, f"Expected {expected_vocab}, but got {result}"

# Additional test cases
def test_empty_dialogue():
    assert get_vocab_from_dialogue([]) == set(), "Empty dialogue should return an empty set"

def test_dialogue_with_punctuation():
    dialogue = [{'speaker': 'Alice', 'text': "Hello, world! How are you?"}]
    expected = {
        ('hello', 'INTJ'),
        ('world', 'NOUN'),
        ('how', 'SCONJ'),
        ('be', 'AUX'),
        ('you', 'PRON')
    }
    assert get_vocab_from_dialogue(dialogue) == expected, "Punctuation should be ignored"

def test_dialogue_with_repeated_words():
    dialogue = [{'speaker': 'Alice', 'text': "The cat and the dog. The cat is sleeping."}]
    expected = {
        ('the', 'DET'),
        ('cat', 'NOUN'),
        ('and', 'CCONJ'),
        ('dog', 'NOUN'),
        ('be', 'AUX'),
        ('sleep', 'VERB')
    }
    assert get_vocab_from_dialogue(dialogue) == expected, "Repeated words should only appear once in the output"


def test_update_vocab_usage_existing_words():
    mock_vocab_usage = {
        "verbs": {"run": 1},
        "vocab": {"fast": 2}
    }
    used_words = {("run", "VERB"), ("fast", "ADJ"), ("quick", "ADJ")}
    
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        with patch("json.dump") as mock_dump:
            update_vocab_usage(used_words)
            
    expected_vocab_usage = {
        "verbs": {"run": 2},
        "vocab": {"fast": 3, "quick": 1}
    }
    mock_dump.assert_called_once_with(expected_vocab_usage, unittest.mock.ANY, indent=2)

def test_update_vocab_usage_new_words():
    mock_vocab_usage = {
        "verbs": {},
        "vocab": {}
    }
    used_words = {("jump", "VERB"), ("high", "ADJ")}
    
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        with patch("json.dump") as mock_dump:
            update_vocab_usage(used_words)
            
    expected_vocab_usage = {
        "verbs": {"jump": 1},
        "vocab": {"high": 1}
    }
    mock_dump.assert_called_once_with(expected_vocab_usage, unittest.mock.ANY, indent=2)

def test_update_vocab_usage_empty_set():
    mock_vocab_usage = {
        "verbs": {"exist": 1},
        "vocab": {"test": 1}
    }
    used_words = set()
    
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        with patch("json.dump") as mock_dump:
            update_vocab_usage(used_words)
            
    expected_vocab_usage = {
        "verbs": {"exist": 1},
        "vocab": {"test": 1}
    }
    mock_dump.assert_called_once_with(expected_vocab_usage, unittest.mock.ANY, indent=2)

def test_update_vocab_usage_non_verb_pos():
    mock_vocab_usage = {
        "verbs": {},
        "vocab": {}
    }
    used_words = {("beautiful", "ADJ"), ("quickly", "ADV"), ("the", "DET")}
    
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_vocab_usage))):
        with patch("json.dump") as mock_dump:
            update_vocab_usage(used_words)
            
    expected_vocab_usage = {
        "verbs": {},
        "vocab": {"beautiful": 1, "quickly": 1, "the": 1}
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

class TestStoryProgressionFunctions(unittest.TestCase):
    def setUp(self):
        self.story_plan = {
            "exposition": "Start of the story",
            "rising_action": "Middle part",
            "climax": "Exciting part",
            "falling_action": "Wrapping up",
            "resolution": "The end",
        }

    def test_get_current_story_part_beginning(self):
        recaps = []
        result = get_current_story_part(recaps, self.story_plan)
        self.assertEqual(result, "exposition")

    def test_get_current_story_part_middle(self):
        recaps = ["Recap 1", "Recap 2", "Recap 3"]
        result = get_current_story_part(recaps, self.story_plan)
        self.assertEqual(result, "falling_action")

    def test_get_current_story_part_end(self):
        recaps = ["Recap 1", "Recap 2", "Recap 3", "Recap 4"]
        result = get_current_story_part(recaps, self.story_plan)
        self.assertEqual(result, "resolution")

    def test_get_last_recap_empty(self):
        recaps = []
        result = get_last_recap(recaps)
        self.assertEqual(result, "This is the beginning of the story.")

    def test_get_last_recap_with_recaps(self):
        recaps = ["Recap 1", "Recap 2", "Last recap"]
        result = get_last_recap(recaps)
        self.assertEqual(result, "Last recap")

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"exposition": "Start", "resolution": "End"}',
    )
    def test_load_story_plan(self, mock_file):
        result = load_story_plan()
        self.assertEqual(result, {"exposition": "Start", "resolution": "End"})

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"recaps": ["Recap 1", "Recap 2"]}',
    )
    def test_load_recaps(self, mock_file):
        result = load_recaps()
        self.assertEqual(result, ["Recap 1", "Recap 2"])

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_story_plan_file_not_found(self, mock_file):
        result = load_story_plan()
        self.assertEqual(result, {})

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_recaps_file_not_found(self, mock_file):
        result = load_recaps()
        self.assertEqual(result, [])

    @patch("os.path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"recaps": ["Recap 1", "Recap 2", "Recap 3"]}',
    )
    def test_get_story_recap_all(self, mock_file, mock_exists):
        mock_exists.return_value = True
        result = get_story_recap()
        self.assertEqual(result, ["Recap 1", "Recap 2", "Recap 3"])

    @patch("os.path.exists")
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"recaps": ["Recap 1", "Recap 2", "Recap 3"]}',
    )
    def test_get_story_recap_limited(self, mock_file, mock_exists):
        mock_exists.return_value = True
        result = get_story_recap(2)
        self.assertEqual(result, ["Recap 2", "Recap 3"])

    @patch("os.path.exists")
    def test_get_story_recap_no_file(self, mock_exists):
        mock_exists.return_value = False
        result = get_story_recap()
        self.assertEqual(result, [])
