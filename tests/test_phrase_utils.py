"""Tests for phrase/tag/guid utility functions."""

from phrases.utils import ANKI_MANAGED_TAG_PREFIX, generate_note_guid, to_anki_tags


def test_generate_note_guid_is_deterministic():
    """Same input must produce the same guid, including across separate calls."""
    guid1 = generate_note_guid("hello_world_abc123", "en-GB", "fr-FR")
    guid2 = generate_note_guid("hello_world_abc123", "en-GB", "fr-FR")

    assert guid1 == guid2


def test_generate_note_guid_differs_by_input():
    """Different phrase key, source, or target language must change the guid."""
    base = generate_note_guid("hello_world_abc123", "en-GB", "fr-FR")

    assert base != generate_note_guid("goodbye_world_def456", "en-GB", "fr-FR")
    assert base != generate_note_guid("hello_world_abc123", "en-GB", "sv-SE")
    assert base != generate_note_guid("hello_world_abc123", "fr-FR", "fr-FR")


def test_generate_note_guid_range():
    """Guid must be a large positive int, matching the formula it replaces."""
    guid = generate_note_guid("hello_world_abc123", "en-GB", "fr-FR")

    assert isinstance(guid, int)
    assert 0 <= guid < 10**10


def test_to_anki_tags_adds_prefix():
    assert to_anki_tags(["SURVIVAL", "Pack01"]) == ["fs::SURVIVAL", "fs::Pack01"]


def test_to_anki_tags_empty_list():
    assert to_anki_tags([]) == []


def test_to_anki_tags_uses_prefix_constant():
    tags = to_anki_tags(["media::film::bron"])

    assert tags == [f"{ANKI_MANAGED_TAG_PREFIX}media::film::bron"]
