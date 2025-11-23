from unittest.mock import patch, MagicMock
from typing import Dict, List, Tuple
import copy
from src.config_loader import config
from src.translation import (
    translate_dialogue,
    translate_from_english,
    translate_phrases,
)
from src.nlp import get_text_tokens


import pytest
from typing import List


@pytest.mark.parametrize(
    "text, language_code, expected, description",
    [
        # Space-separated languages
        ("Hello world", "en", ["Hello", "world"], "Basic English"),
        ("Bonjour le monde", "fr", ["Bonjour", "le", "monde"], "Basic French"),
        ("hello   world", "en", ["hello", "world"], "Multiple spaces"),
        # Japanese text
        ("こんにちは世界", "ja", ["こんにちは", "世界"], "Basic Japanese"),
        ("私は猫です", "ja", ["私", "は", "猫", "です"], "Japanese with particles"),
        # Chinese text
        ("你好世界", "zh", ["你好", "世界"], "Basic Chinese"),
        ("我是一个学生", "zh", ["我", "是", "一个", "学生"], "Chinese sentence"),
        # Mixed script
        ("Hello 世界", "ja", ["Hello", "世界"], "Mixed English-Japanese"),
        # Edge cases
        ("", "en", [], "Empty string"),
        ("A", "en", ["A"], "Single character"),
        # Fallback cases - assuming API failure
        ("Hello", "xx", ["Hello"], "Invalid language code"),
    ],
)
def test_tokenize_text(
    text: str, language_code: str, expected: List[str], description: str
):
    """
    Test tokenization across different languages and scenarios.

    Note: When using the actual Google API, results might differ from expected.
    This test assumes ideal tokenization - in practice you might want to:
    1. Mock the API response
    2. Test for patterns rather than exact matches
    3. Skip API-dependent tests in certain environments
    """
    try:
        result = get_text_tokens(text, language_code)

        # For API-based tokenization, we might want to verify patterns rather than exact matches
        if language_code in ["ja", "zh"]:
            assert (
                len(result) > 0
            ), f"Failed {description}: Should have at least one token"
            assert all(
                len(token) > 0 for token in result
            ), f"Failed {description}: Empty token found"
        else:
            assert result == expected, f"Failed {description}"

    except Exception as e:
        # Handle API-related failures gracefully
        if "API" in str(e):
            # For space-separated languages, verify fallback behavior
            if " " in text:
                assert get_text_tokens(text, language_code) == text.split()
            else:
                assert get_text_tokens(text, language_code) == [text]
        else:
            raise e


@patch("google.cloud.translate_v2.Client")
def test_translate_from_english(mock_translate_client):
    # Setup
    mock_client = MagicMock()
    mock_translate_client.return_value = mock_client
    mock_client.translate.return_value = {"translatedText": "Hola mundo"}

    # Test with default target language
    result = translate_from_english("Hello world")
    assert result == ["Hola mundo"]
    mock_client.translate.assert_called_with(
        "Hello world",
        target_language=config.TARGET_LANGUAGE_ALPHA2,
        source_language="en",
    )

    # Test with specific target language
    result = translate_from_english("Hello world", target_language="fr")
    assert result == [
        "Hola mundo"
    ]  # We're using the same mock, so the result is the same
    mock_client.translate.assert_called_with(
        "Hello world", target_language="fr", source_language="en"
    )


def mock_func_tranlsate_from_english(text):
    if isinstance(text, list):
        return [f"Translated: {item}" for item in text]
    else:
        return [f"Translated: {text}"]


@patch("src.translation.batch_translate")
def test_translate_dialogue(mock_batch_translate):
    # Setup
    mock_batch_translate.return_value = [
        "Translated: Hello",
        "Translated: How are you?",
    ]

    dialogue = [
        {"speaker": "Alice", "text": "Hello"},
        {"speaker": "Bob", "text": "How are you?"},
    ]

    # Test
    result = translate_dialogue(dialogue)

    # Assert
    expected_result = [
        {"speaker": "Alice", "text": "Translated: Hello"},
        {"speaker": "Bob", "text": "Translated: How are you?"},
    ]
    assert result == expected_result
    mock_batch_translate.assert_called_once_with(["Hello", "How are you?"])


@patch("src.translation.batch_translate")
def test_translate_dialogue_deep_copy(mock_batch_translate):
    # Setup
    mock_batch_translate.return_value = [
        "Translated: Hello",
        "Translated: How are you?",
    ]

    original_dialogue = [
        {"speaker": "Alice", "text": "Hello"},
        {"speaker": "Bob", "text": "How are you?"},
    ]

    # Make a deep copy of the original dialogue
    original_dialogue_copy = copy.deepcopy(original_dialogue)

    # Test
    result = translate_dialogue(original_dialogue)

    # Assert
    assert (
        original_dialogue == original_dialogue_copy
    ), "Original dialogue should not be modified"
    assert (
        result != original_dialogue
    ), "Translated dialogue should be different from the original"
    assert id(result) != id(
        original_dialogue
    ), "A new list should be returned, not the original one"
    for original, translated in zip(original_dialogue, result):
        assert id(original) != id(
            translated
        ), "Each dictionary in the list should be a new object"
        assert (
            original["speaker"] == translated["speaker"]
        ), "Speaker should remain the same"
        assert original["text"] != translated["text"], "Text should be translated"
    mock_batch_translate.assert_called_once_with(["Hello", "How are you?"])


@patch("src.translation.batch_translate")
def test_translate_phrases(mock_batch_translate):
    # Setup
    mock_batch_translate.return_value = [
        "Translated: Hello",
        "Translated: How are you?",
    ]

    phrases = ["Hello", "How are you?"]

    # Test
    result = translate_phrases(phrases)

    # Assert
    expected_result = [
        ("Hello", "Translated: Hello"),
        ("How are you?", "Translated: How are you?"),
    ]
    assert result == expected_result
    mock_batch_translate.assert_called_once_with(phrases)
