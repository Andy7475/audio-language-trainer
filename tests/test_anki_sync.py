"""Tests for the live-collection Anki sync module (anki_sync.py)."""

from types import SimpleNamespace

import pytest
from anki.collection import Collection
from PIL import Image
from pydub import AudioSegment

from anki_sync import (
    build_content_index,
    create_note_in_collection,
    find_existing_note,
    sync_note_tags,
    sync_phrases_to_anki,
)
from audio.voices import VoiceInfo
from models import BCP47Language
from phrases.phrase_model import Phrase, PhraseAudio, Translation
from phrases.utils import generate_note_guid

ANKI_NOTE_FIELDS = [
    "SortOrder",
    "SourceText",
    "TargetText",
    "TargetAudio",
    "TargetAudioSlow",
    "WiktionaryLinks",
    "Picture",
    "SourceLanguageName",
    "TargetLanguageName",
]


@pytest.fixture
def anki_collection(tmp_path):
    """A throwaway Anki collection (never the user's real one) with a
    minimal FirePhrase2-shaped notetype - enough to exercise field/tag/guid
    logic without needing real templates/CSS."""
    col = Collection(str(tmp_path / "test.anki2"))

    notetype = col.models.new("FirePhrase2")
    for name in ANKI_NOTE_FIELDS:
        col.models.add_field(notetype, col.models.new_field(name))
    template = col.models.new_template("Card 1")
    template["qfmt"] = "{{SourceText}}"
    template["afmt"] = "{{TargetText}}"
    col.models.add_template(notetype, template)
    col.models.add_dict(notetype)

    yield col
    col.close()


@pytest.fixture
def sample_phrase():
    """A phrase with en-GB/fr-FR translations and Firestore tags, mirroring
    the mock_phrase fixture in test_anki_tools.py."""
    phrase = Phrase(
        phrase_hash="hello_world_abc123",
        english="Hello, world!",
        english_lower="hello, world!",
        tokens=["Hello", "world"],
        verbs=[],
        vocab=["hello", "world"],
    )

    en_translation = Translation(
        phrase_hash="hello_world_abc123",
        language=BCP47Language.get("en-GB"),
        text="Hello, world!",
        text_lower="hello, world!",
        tokens=["Hello", "world"],
        image_file_path="phrases/en-GB/images/hello_world_abc123.png",
    )
    en_translation.image = Image.new("RGB", (100, 100), color="red")

    fr_translation = Translation(
        phrase_hash="hello_world_abc123",
        language=BCP47Language.get("fr-FR"),
        text="Bonjour le monde!",
        text_lower="bonjour le monde!",
        tokens=["Bonjour", "le", "monde"],
        image_file_path="phrases/en-GB/images/hello_world_abc123.png",
        tags=["SURVIVAL", "Pack01"],
    )

    mock_audio_normal = PhraseAudio(
        phrase_hash="hello_world_abc123",
        text="Bonjour le monde!",
        file_path="phrases/fr-FR/audio/flashcard/normal/hello_world_abc123.mp3",
        language=BCP47Language.get("fr-FR"),
        context="flashcard",
        speed="normal",
        voice_info=VoiceInfo(
            provider="google", voice_id="fr-FR-Standard-A", language_code="fr-FR"
        ),
    )
    mock_audio_normal.audio_segment = AudioSegment.silent(duration=100)

    mock_audio_slow = PhraseAudio(
        phrase_hash="hello_world_abc123",
        text="Bonjour le monde!",
        file_path="phrases/fr-FR/audio/flashcard/slow/hello_world_abc123.mp3",
        language=BCP47Language.get("fr-FR"),
        context="flashcard",
        speed="slow",
        voice_info=VoiceInfo(
            provider="google", voice_id="fr-FR-Standard-A", language_code="fr-FR"
        ),
    )
    mock_audio_slow.audio_segment = AudioSegment.silent(duration=200)

    fr_translation.audio = {
        "flashcard": {"normal": mock_audio_normal, "slow": mock_audio_slow}
    }
    fr_translation.image = en_translation.image

    phrase.translations = {"en-GB": en_translation, "fr-FR": fr_translation}
    return phrase


@pytest.fixture
def no_gcs(monkeypatch):
    """Bypass Phrase.download()/get_image() unconditionally hitting GCS -
    sample_phrase already has audio_segment/image populated directly."""
    monkeypatch.setattr(Phrase, "download", lambda self, **kwargs: None)
    monkeypatch.setattr(Phrase, "get_image", lambda self, **kwargs: None)


# ---------------------------------------------------------------------------
# sync_note_tags - pure, unit-testable with a stub object
# ---------------------------------------------------------------------------


def test_sync_note_tags_replaces_managed_tags_only():
    note = SimpleNamespace(tags=["fs::OLD", "marked", "fs::Pack01"])

    changed = sync_note_tags(note, ["NEW", "Pack01"])

    assert changed is True
    assert set(note.tags) == {"marked", "fs::NEW", "fs::Pack01"}


def test_sync_note_tags_no_change_returns_false():
    note = SimpleNamespace(tags=["marked", "fs::SURVIVAL"])

    changed = sync_note_tags(note, ["SURVIVAL"])

    assert changed is False


def test_sync_note_tags_ignores_tag_order():
    """Tags are semantically a set - reordering alone isn't a change."""
    note = SimpleNamespace(tags=["fs::SURVIVAL", "marked", "fs::Pack01"])

    changed = sync_note_tags(note, ["Pack01", "SURVIVAL"])

    assert changed is False


def test_sync_note_tags_preserves_non_managed_tags():
    note = SimpleNamespace(tags=["leech", "marked", "my-custom-tag"])

    sync_note_tags(note, ["SURVIVAL"])

    assert "leech" in note.tags
    assert "marked" in note.tags
    assert "my-custom-tag" in note.tags


def test_sync_note_tags_empty_firestore_tags_removes_all_managed():
    note = SimpleNamespace(tags=["fs::OLD", "marked"])

    changed = sync_note_tags(note, [])

    assert changed is True
    assert note.tags == ["marked"]


# ---------------------------------------------------------------------------
# create_note_in_collection / find_existing_note - throwaway Collection
# ---------------------------------------------------------------------------


def test_create_note_in_collection_sets_fields_guid_and_empty_tags(
    anki_collection, sample_phrase, no_gcs
):
    notetype = anki_collection.models.by_name("FirePhrase2")
    deck_id = anki_collection.decks.id_for_name("Default")

    note_id = create_note_in_collection(
        anki_collection, sample_phrase, "en-GB", "fr-FR", deck_id, notetype
    )

    note = anki_collection.get_note(note_id)
    assert note["SourceText"] == "Hello, world!"
    assert note["TargetText"] == "Bonjour le monde!"
    assert "[sound:" in note["TargetAudio"]
    assert "[sound:" in note["TargetAudioSlow"]
    assert "<img src=" in note["Picture"]
    assert note.guid == str(generate_note_guid("hello_world_abc123", "en-GB", "fr-FR"))
    # tags are deliberately left for sync_note_tags to set, not this function
    assert note.tags == []


def test_find_existing_note_guid_fast_path(anki_collection, sample_phrase, no_gcs):
    notetype = anki_collection.models.by_name("FirePhrase2")
    deck_id = anki_collection.decks.id_for_name("Default")
    note_id = create_note_in_collection(
        anki_collection, sample_phrase, "en-GB", "fr-FR", deck_id, notetype
    )

    found = find_existing_note(
        anki_collection, sample_phrase, "en-GB", "fr-FR", notetype["id"]
    )

    assert found is not None
    assert found.note_id == note_id
    assert found.matched_by == "guid"
    assert found.guid_needs_healing is False


def test_find_existing_note_content_fallback_needs_healing(
    anki_collection, sample_phrase, no_gcs
):
    notetype = anki_collection.models.by_name("FirePhrase2")
    deck_id = anki_collection.decks.id_for_name("Default")
    note_id = create_note_in_collection(
        anki_collection, sample_phrase, "en-GB", "fr-FR", deck_id, notetype
    )

    # Simulate a note that predates generate_note_guid: its stored guid is
    # an old, unrecoverable random value that won't match a fresh guid
    # computation for the same phrase.
    note = anki_collection.get_note(note_id)
    note.guid = "some_old_random_guid_12345"
    anki_collection.update_note(note)

    content_index = build_content_index(anki_collection, notetype["id"])
    found = find_existing_note(
        anki_collection, sample_phrase, "en-GB", "fr-FR", notetype["id"], content_index
    )

    assert found is not None
    assert found.note_id == note_id
    assert found.matched_by == "content"
    assert found.guid_needs_healing is True


def test_find_existing_note_returns_none_when_absent(anki_collection, sample_phrase):
    notetype = anki_collection.models.by_name("FirePhrase2")

    found = find_existing_note(
        anki_collection, sample_phrase, "en-GB", "fr-FR", notetype["id"]
    )

    assert found is None


# ---------------------------------------------------------------------------
# sync_phrases_to_anki - orchestration
# ---------------------------------------------------------------------------


def test_sync_creates_missing_note_and_sets_tags(anki_collection, sample_phrase, no_gcs):
    report = sync_phrases_to_anki(
        anki_collection, [sample_phrase], "en-GB", "fr-FR", dry_run=False
    )

    assert report.results[0].created is True
    assert report.results[0].tags_changed is True
    assert anki_collection.note_count() == 1

    notetype = anki_collection.models.by_name("FirePhrase2")
    note_id = anki_collection.db.scalar(
        "select id from notes where mid = ?", notetype["id"]
    )
    note = anki_collection.get_note(note_id)
    assert "fs::SURVIVAL" in note.tags
    assert "fs::Pack01" in note.tags


def test_sync_is_idempotent_on_resync(anki_collection, sample_phrase, no_gcs):
    sync_phrases_to_anki(
        anki_collection, [sample_phrase], "en-GB", "fr-FR", dry_run=False
    )
    assert anki_collection.note_count() == 1

    report2 = sync_phrases_to_anki(
        anki_collection, [sample_phrase], "en-GB", "fr-FR", dry_run=False
    )

    assert anki_collection.note_count() == 1  # no duplicate note created
    assert report2.results[0].created is False
    assert report2.results[0].tags_changed is False  # already in sync


def test_sync_heals_legacy_guid_without_duplicating(
    anki_collection, sample_phrase, no_gcs
):
    """A note that predates generate_note_guid must be found via content
    match and healed in place - not duplicated as a second note."""
    notetype = anki_collection.models.by_name("FirePhrase2")
    deck_id = anki_collection.decks.id_for_name("Default")
    note_id = create_note_in_collection(
        anki_collection, sample_phrase, "en-GB", "fr-FR", deck_id, notetype
    )
    note = anki_collection.get_note(note_id)
    note.guid = "some_old_random_guid_12345"
    anki_collection.update_note(note)

    report = sync_phrases_to_anki(
        anki_collection, [sample_phrase], "en-GB", "fr-FR", dry_run=False
    )

    assert anki_collection.note_count() == 1  # still just the one note
    assert report.results[0].created is False
    assert report.results[0].guid_healed is True

    healed_note = anki_collection.get_note(note_id)
    assert healed_note.guid == str(
        generate_note_guid("hello_world_abc123", "en-GB", "fr-FR")
    )


def test_sync_dry_run_makes_no_changes(anki_collection, sample_phrase, no_gcs):
    report = sync_phrases_to_anki(
        anki_collection, [sample_phrase], "en-GB", "fr-FR", dry_run=True
    )

    assert report.dry_run is True
    assert report.results[0].created is True
    assert anki_collection.note_count() == 0  # nothing actually written


def test_sync_missing_notetype_raises(anki_collection, sample_phrase):
    with pytest.raises(RuntimeError, match="not found"):
        sync_phrases_to_anki(
            anki_collection,
            [sample_phrase],
            "en-GB",
            "fr-FR",
            notetype_name="DoesNotExist",
            dry_run=True,
        )
