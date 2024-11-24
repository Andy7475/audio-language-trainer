import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Tuple
import copy
from src.config_loader import config
from src.translation import (
    translate_dialogue,
    translate_from_english,
    translate_phrases,
    tokenize_text,
)


def test_space_separated():
    """Test normal space-separated text"""
    assert tokenize_text("Hello world", "en") == ["Hello", "world"]
    assert tokenize_text("Bonjour le monde", "fr") == ["Bonjour", "le", "monde"]


def test_japanese():
    """Test Japanese - should keep meaningful chunks"""
    # Basic phrase
    tokens = tokenize_text("こんにちは世界", "ja")
    print(f"Japanese tokens: {tokens}")
    assert len(tokens) >= 2  # Should have reasonable chunks, not single characters


def test_chinese():
    """Test Chinese - should merge very short tokens"""
    # "Hello world"
    tokens = tokenize_text("你好世界", "zh")
    print(f"Chinese tokens: {tokens}")
    # Should merge single characters but keep meaningful units
    assert len(tokens) >= 1 and len(tokens) <= 3


def test_mixed_script():
    """Test mixed scripts - common in real usage"""
    tokens = tokenize_text("Hello 世界", "ja")
    print(f"Mixed tokens: {tokens}")
    assert "Hello" in tokens
    assert any("世界" in t for t in tokens)


def test_fallback_behavior():
    """Test fallback behavior for API failures"""
    # Test with a very long string that might cause API issues
    long_text = "Hello world " * 1000
    tokens = tokenize_text(long_text, "en")
    assert len(tokens) > 0
    assert all(len(t) > 0 for t in tokens)


def test_edge_cases():
    """Test edge cases"""
    # Empty string
    assert tokenize_text("") == []

    # Single character
    assert len(tokenize_text("A", "en")) == 1

    # Multiple spaces
    assert tokenize_text("hello   world", "en") == ["hello", "world"]


def test_practical_tts_chunks():
    """Test that chunks are practical for TTS breaks"""
    # Japanese example with particles
    tokens = tokenize_text("私は猫です", "ja")
    print(f"Japanese sentence tokens: {tokens}")
    # Shouldn't have too many breaks
    assert len(tokens) <= 4

    # Chinese example
    tokens = tokenize_text("我是一个学生", "zh")
    print(f"Chinese sentence tokens: {tokens}")
    # Shouldn't break every character
    assert len(tokens) <= 4


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
