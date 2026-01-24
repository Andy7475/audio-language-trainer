"""Tests for the Anki tools module.

These tests demonstrate how to use the modular anki_tools functions to create
flashcard decks from Phrase objects with various language combinations.
"""

import os
import tempfile

import pytest
import genanki

from src.anki_tools import (
    get_anki_model,
    create_anki_note_from_phrase,
    create_anki_deck,
    save_anki_deck,
    create_and_save_anki_deck,
    _get_sort_field,
    _string_to_large_int,
)
from src.models import BCP47Language
from src.phrases.phrase_model import Phrase, Translation, PhraseAudio
from src.audio.voices import VoiceInfo
from pydub import AudioSegment
from PIL import Image


@pytest.fixture
def mock_phrase():
    """Create a mock phrase with English and French translations."""
    phrase = Phrase(
        phrase_hash="hello_world_abc123",
        english="Hello, world!",
        english_lower="hello, world!",
        tokens=["Hello", "world"],
        verbs=[],
        vocab=["hello", "world"],
    )

    # Add English translation
    en_translation = Translation(
        phrase_hash="hello_world_abc123",
        language=BCP47Language.get("en-GB"),
        text="Hello, world!",
        text_lower="hello, world!",
        tokens=["Hello", "world"],
        image_file_path="phrases/en-GB/images/hello_world_abc123.png",
    )

    # Create mock image
    en_translation.image = Image.new("RGB", (100, 100), color="red")

    # Add French translation with audio
    fr_translation = Translation(
        phrase_hash="hello_world_abc123",
        language=BCP47Language.get("fr-FR"),
        text="Bonjour le monde!",
        text_lower="bonjour le monde!",
        tokens=["Bonjour", "le", "monde"],
        image_file_path="phrases/en-GB/images/hello_world_abc123.png",  # Shared image
    )

    # Create mock audio
    mock_audio_normal = PhraseAudio(
        phrase_hash="hello_world_abc123",
        text="Bonjour le monde!",
        file_path="phrases/fr-FR/audio/flashcard/normal/hello_world_abc123.mp3",
        language=BCP47Language.get("fr-FR"),
        context="flashcard",
        speed="normal",
        voice_info=VoiceInfo(provider="google", voice_id="fr-FR-Standard-A"),
    )
    # Create a simple silent audio segment (100ms)
    mock_audio_normal.audio_segment = AudioSegment.silent(duration=100)

    mock_audio_slow = PhraseAudio(
        phrase_hash="hello_world_abc123",
        text="Bonjour le monde!",
        file_path="phrases/fr-FR/audio/flashcard/slow/hello_world_abc123.mp3",
        language=BCP47Language.get("fr-FR"),
        context="flashcard",
        speed="slow",
        voice_info=VoiceInfo(provider="google", voice_id="fr-FR-Standard-A"),
    )
    mock_audio_slow.audio_segment = AudioSegment.silent(duration=200)

    # Add audio to translation
    fr_translation.audio = {
        "flashcard": {
            "normal": mock_audio_normal,
            "slow": mock_audio_slow,
        }
    }

    # Use same image as English
    fr_translation.image = en_translation.image

    # Add translations to phrase
    phrase.translations = {
        "en-GB": en_translation,
        "fr-FR": fr_translation,
    }

    return phrase


def test_get_anki_model():
    """Test that get_anki_model returns a valid genanki.Model."""
    model = get_anki_model()

    assert isinstance(model, genanki.Model)
    assert model.model_id == 1607392313
    assert model.name == "FirePhrase"

    # Check fields
    field_names = [f["name"] for f in model.fields]
    assert "SourceText" in field_names
    assert "TargetText" in field_names
    assert "TargetAudio" in field_names

    # Check templates
    template_names = [t["name"] for t in model.templates]
    assert "Listening Card" in template_names
    assert "Reading Card" in template_names
    assert "Speaking Card" in template_names


def test_get_sort_field():
    """Test sort field generation."""
    sort1 = _get_sort_field(0, "hello")
    sort2 = _get_sort_field(1, "world")
    sort3 = _get_sort_field(0, "hello")  # Same as sort1

    # Should start with zero-padded index
    assert sort1.startswith("0000-")
    assert sort2.startswith("0001-")

    # Same input should produce same output
    assert sort1 == sort3

    # Different inputs should produce different outputs
    assert sort1 != sort2


def test_string_to_large_int():
    """Test string to integer conversion."""
    id1 = _string_to_large_int("test_deck")
    id2 = _string_to_large_int("test_deck")
    id3 = _string_to_large_int("other_deck")

    # Should be consistent
    assert id1 == id2

    # Should be different for different strings
    assert id1 != id3

    # Should be a large positive integer
    assert id1 > 0
    assert id1 < 10**10


def test_create_anki_note_from_phrase(mock_phrase):
    """Test creating an Anki note from a Phrase object."""
    model = get_anki_model()

    with tempfile.TemporaryDirectory() as temp_dir:
        note, media_files = create_anki_note_from_phrase(
            phrase=mock_phrase,
            source_language="en-GB",
            target_language="fr-FR",
            index=0,
            model=model,
            temp_dir=temp_dir,
        )

        # Check note
        assert isinstance(note, genanki.Note)
        assert note.model == model

        # Check fields
        fields = note.fields
        assert "Hello, world!" in fields[1]  # SourceText
        assert "Bonjour le monde!" in fields[2]  # TargetText
        assert "[sound:" in fields[3]  # TargetAudio
        assert "[sound:" in fields[4]  # TargetAudioSlow
        assert "<img src=" in fields[6]  # Picture

        # Check media files were created
        assert len(media_files) == 3  # 2 audio + 1 image
        for media_file in media_files:
            assert os.path.exists(media_file)


def test_create_anki_note_missing_translation(mock_phrase):
    """Test error handling when translation is missing."""
    model = get_anki_model()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Try to create note with missing Spanish translation
        with pytest.raises(ValueError, match="missing target translation"):
            create_anki_note_from_phrase(
                phrase=mock_phrase,
                source_language="en-GB",
                target_language="es-ES",
                index=0,
                model=model,
                temp_dir=temp_dir,
            )


def test_create_anki_deck(mock_phrase):
    """Test creating an Anki deck from phrases."""
    phrases = [mock_phrase]

    package = create_anki_deck(
        phrases=phrases,
        source_language="en-GB",
        target_language="fr-FR",
        deck_name="Test Deck::French",
    )

    assert isinstance(package, genanki.Package)
    assert len(package.decks) == 1
    assert package.decks[0].name == "Test Deck::French"

    # Check that deck has notes
    deck = package.decks[0]
    assert len(deck.notes) == 1


def test_create_anki_deck_auto_name(mock_phrase):
    """Test automatic deck name generation."""
    phrases = [mock_phrase]

    package = create_anki_deck(
        phrases=phrases,
        source_language="en-GB",
        target_language="fr-FR",
    )

    # Should auto-generate name
    assert package.decks[0].name
    assert "French" in package.decks[0].name


def test_create_anki_deck_empty_list():
    """Test error handling for empty phrase list."""
    with pytest.raises(ValueError, match="cannot be empty"):
        create_anki_deck(
            phrases=[],
            source_language="en-GB",
            target_language="fr-FR",
        )


def test_save_anki_deck(mock_phrase):
    """Test saving an Anki deck to file."""
    phrases = [mock_phrase]
    package = create_anki_deck(
        phrases=phrases,
        source_language="en-GB",
        target_language="fr-FR",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_deck.apkg")
        saved_path = save_anki_deck(package, output_path)

        # Check file was created
        assert os.path.exists(saved_path)
        assert saved_path.endswith(".apkg")

        # Check file has content
        assert os.path.getsize(saved_path) > 0


def test_save_anki_deck_auto_extension(mock_phrase):
    """Test that .apkg extension is added automatically."""
    phrases = [mock_phrase]
    package = create_anki_deck(
        phrases=phrases,
        source_language="en-GB",
        target_language="fr-FR",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_deck")  # No extension
        saved_path = save_anki_deck(package, output_path)

        assert saved_path.endswith(".apkg")
        assert os.path.exists(saved_path)


def test_create_and_save_anki_deck(mock_phrase):
    """Test the convenience function for creating and saving in one step."""
    phrases = [mock_phrase]

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_deck.apkg")

        saved_path = create_and_save_anki_deck(
            phrases=phrases,
            source_language="en-GB",
            target_language="fr-FR",
            output_path=output_path,
            deck_name="Test::Deck",
        )

        # Check file was created
        assert os.path.exists(saved_path)
        assert saved_path.endswith(".apkg")


def test_multiple_language_combinations(mock_phrase):
    """Test that different language combinations work."""
    # Add a Japanese translation to the mock phrase
    ja_translation = Translation(
        phrase_hash="hello_world_abc123",
        language=BCP47Language.get("ja-JP"),
        text="こんにちは世界",
        text_lower="こんにちは世界",
        tokens=["こんにちは", "世界"],
        image_file_path="phrases/en-GB/images/hello_world_abc123.png",
    )

    # Add mock audio
    mock_audio = PhraseAudio(
        phrase_hash="hello_world_abc123",
        text="こんにちは世界",
        file_path="phrases/ja-JP/audio/flashcard/normal/hello_world_abc123.mp3",
        language=BCP47Language.get("ja-JP"),
        context="flashcard",
        speed="normal",
        voice_info=VoiceInfo(provider="google", voice_id="ja-JP-Standard-A"),
    )
    mock_audio.audio_segment = AudioSegment.silent(duration=100)

    ja_translation.audio = {"flashcard": {"normal": mock_audio}}
    ja_translation.image = mock_phrase.translations["en-GB"].image

    mock_phrase.translations["ja-JP"] = ja_translation

    # Test English to Japanese
    package = create_anki_deck(
        phrases=[mock_phrase],
        source_language="en-GB",
        target_language="ja-JP",
        deck_name="Japanese::Test",
    )

    assert package.decks[0].name == "Japanese::Test"
    assert len(package.decks[0].notes) == 1

    # Test French to Japanese
    package2 = create_anki_deck(
        phrases=[mock_phrase],
        source_language="fr-FR",
        target_language="ja-JP",
        deck_name="Japanese from French",
    )

    assert len(package2.decks[0].notes) == 1
