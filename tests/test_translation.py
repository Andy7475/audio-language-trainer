import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Tuple
import copy
from src.config_loader import config
from src.translation import (
    translate_dialogue,
    translate_from_english,
    translate_phrases,
)


@patch("google.cloud.translate_v2.Client")
def test_translate_from_english(mock_translate_client):
    # Setup
    mock_client = MagicMock()
    mock_translate_client.return_value = mock_client
    mock_client.translate.return_value = {"translatedText": "Hola mundo"}

    # Test with default target language
    result = translate_from_english("Hello world")
    assert result == "Hola mundo"
    mock_client.translate.assert_called_with(
        "Hello world", target_language=config.TARGET_LANGUAGE, source_language="en"
    )

    # Test with specific target language
    result = translate_from_english("Hello world", target_language="fr")
    assert (
        result == "Hola mundo"
    )  # We're using the same mock, so the result is the same
    mock_client.translate.assert_called_with(
        "Hello world", target_language="fr", source_language="en"
    )


@patch("src.translation.translate_from_english")
def test_translate_dialogue(mock_translate_from_english):
    # Setup
    mock_translate_from_english.side_effect = lambda text: f"Translated: {text}"

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
    assert mock_translate_from_english.call_count == 2


@patch("src.translation.translate_from_english")
def test_translate_dialogue_deep_copy(mock_translate_from_english):
    # Setup
    mock_translate_from_english.side_effect = lambda text: f"Translated: {text}"

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


@patch("src.translation.translate_from_english")
def test_translate_phrases(mock_translate_from_english):
    # Setup
    mock_translate_from_english.side_effect = lambda text: f"Translated: {text}"

    phrases = ["Hello", "How are you?"]

    # Test
    result = translate_phrases(phrases)

    # Assert
    expected_result = [
        ("Hello", "Translated: Hello"),
        ("How are you?", "Translated: How are you?"),
    ]
    assert result == expected_result
    assert mock_translate_from_english.call_count == 2
