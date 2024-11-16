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
