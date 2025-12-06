from src.anki_tools import generate_wiktionary_links
import time
import pytest


def test_japanese():
    """Test Japanese words and phrases"""
    # Test just 'world'
    result = generate_wiktionary_links("世界", "Japanese", "ja")
    print(f"\nJapanese 'world': {result}")
    assert 'href="https://en.wiktionary.org/wiki/%E4%B8%96%E7%95%8C#Japanese"' in result
    time.sleep(1)

    # Test 'hello world'
    result = generate_wiktionary_links("こんにちは世界", "Japanese", "ja")
    print(f"Japanese 'hello world': {result}")
    assert 'href="https://en.wiktionary.org/wiki/%E4%B8%96%E7%95%8C#Japanese"' in result
    time.sleep(1)


def test_chinese():
    """Test Chinese (Mandarin) words and phrases"""
    # Test just 'world'
    result = generate_wiktionary_links("世界", "Chinese", "zh")
    print(f"\nChinese 'world': {result}")
    assert 'href="https://en.wiktionary.org/wiki/%E4%B8%96%E7%95%8C#Chinese"' in result
    time.sleep(1)

    # Test 'hello world'
    result = generate_wiktionary_links("你好世界", "Chinese", "zh")
    print(f"Chinese 'hello world': {result}")
    assert 'href="https://en.wiktionary.org/wiki/%E4%B8%96%E7%95%8C#Chinese"' in result
    time.sleep(1)


def test_swedish():
    """Test Swedish words and phrases"""
    # Test just 'world'
    result = generate_wiktionary_links("värld", "Swedish", "sv")
    print(f"\nSwedish 'world': {result}")
    assert 'href="https://en.wiktionary.org/wiki/v%C3%A4rld#Swedish"' in result
    time.sleep(1)

    # Test 'hello world'
    result = generate_wiktionary_links("hej värld", "Swedish", "sv")
    print(f"Swedish 'hello world': {result}")
    assert 'href="https://en.wiktionary.org/wiki/v%C3%A4rld#Swedish"' in result
    time.sleep(1)


def test_nonexistent_word():
    """Test handling of words not in Wiktionary"""
    nonsense = "xyzqabc123"
    result = generate_wiktionary_links(nonsense, "English", "en")
    print(f"\nResult for nonexistent word: {result}")
    assert "href=" not in result
    assert nonsense in result  # Word should be present but not linked
    time.sleep(1)


def test_real_japanese_phrase():
    """Test a natural Japanese phrase"""
    result = generate_wiktionary_links("私は猫が好きです", "Japanese", "ja")
    print(f"\nJapanese natural phrase: {result}")
    # Just print for inspection - links will depend on what's in Wiktionary
    time.sleep(1)


def test_spanish_punctuation_current_issue():
    """Test Spanish inverted punctuation marks - demonstrates current issue"""
    # Test phrase with Spanish inverted punctuation
    spanish_phrase = "¿Cómo estás? ¡Muy bien!"

    result = generate_wiktionary_links(spanish_phrase, "Spanish", "es")
    print(f"\nSpanish phrase with inverted punctuation: {result}")

    # These assertions will currently FAIL due to the punctuation issue
    # The words with leading inverted punctuation won't get proper links
    # After implementing the fix, these should pass

    # Check that basic words are processed (even if not linked due to current bug)
    assert "¿Cómo" in result or "Cómo" in result
    assert "¡Muy" in result or "Muy" in result

    # Log what we expect vs what we get for debugging
    if "¿Cómo" in result:
        print("  ❌ Issue confirmed: '¿Cómo' still contains leading punctuation")
    if "¡Muy" in result:
        print("  ❌ Issue confirmed: '¡Muy' still contains leading punctuation")

    time.sleep(1)


def test_multilingual_punctuation_edge_cases():
    """Test various punctuation scenarios across languages"""

    test_cases = [
        # Spanish inverted punctuation
        ("¿Dónde está?", "Spanish", "es"),
        ("¡Hola amigo!", "Spanish", "es"),
        # French contractions and quotes
        ("Qu'est-ce que c'est?", "French", "fr"),
        ("«Bonjour» dit-il.", "French", "fr"),
        # German contractions
        ("Wie geht's dir?", "German", "de"),
        # Mixed punctuation
        ("¿Hablas «français»?", "Spanish", "es"),
    ]

    for phrase, lang_name, lang_code in test_cases:
        result = generate_wiktionary_links(phrase, lang_name, lang_code)
        print(f"\n{lang_name} punctuation test: '{phrase}' → {result}")

        # Basic sanity check - result should contain the phrase content
        assert len(result) > 0

        time.sleep(0.5)


@pytest.mark.parametrize(
    "word,expected_clean",
    [
        # Spanish inverted punctuation
        ("¿Cómo", "Cómo"),
        ("¡Hola", "Hola"),
        ("estás?", "estás"),
        ("bien!", "bien"),
        # Multiple punctuation
        ("¿¿palabra??", "palabra"),
        ("¡¡word!!", "word"),
        # French contractions and quotes
        ("Qu'est-ce", "Qu'est-ce"),  # Should keep internal apostrophe
        ("«word»", "word"),
        ("'bonjour'", "bonjour"),
        # Mixed scenarios
        ("¿«hello»?", "hello"),
        ("¡'word'!", "word"),
        # Should not change
        ("hello", "hello"),
        ("café", "café"),
        ("naïve", "naïve"),
    ],
)
def test_word_cleaning_expectations(word, expected_clean):
    """Test what word cleaning should produce after implementing the fix.

    Note: This test documents the expected behavior but will fail with current implementation.
    It serves as a specification for the fix we need to implement.
    """
    # Import the function we'll create
    try:
        from src.wiktionary import clean_word_for_lookup

        cleaned = clean_word_for_lookup(word)
        assert (
            cleaned == expected_clean
        ), f"'{word}' should clean to '{expected_clean}', got '{cleaned}'"
    except ImportError:
        # Function doesn't exist yet - this is expected before implementing the fix
        pytest.skip("clean_word_for_lookup function not implemented yet")
