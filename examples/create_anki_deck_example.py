"""Example script demonstrating how to create Anki decks using the new anki_tools module.

This script shows various ways to create flashcard decks from Phrase objects
with different source and target language combinations.
"""

from src.phrases.phrase_model import get_phrase_by_english
from src.anki_tools import (
    create_anki_deck,
    save_anki_deck,
    create_and_save_anki_deck,
)
from src.models import BCP47Language


def example_basic_deck():
    """Create a basic French deck from English phrases."""
    print("\n=== Example 1: Basic French Deck ===")

    # Get some phrases
    phrases = [
        get_phrase_by_english("Hello"),
        get_phrase_by_english("Good morning"),
        get_phrase_by_english("Thank you"),
    ]

    # Ensure they have French translations
    for phrase in phrases:
        if "fr-FR" not in phrase.translations:
            print(f"Translating: {phrase.english}")
            phrase.translate("fr-FR")
            phrase.generate_audio("flashcard", "fr-FR")

    # Create the deck
    package = create_anki_deck(
        phrases=phrases,
        source_language="en-GB",
        target_language="fr-FR",
        deck_name="French::Beginner::Greetings",
    )

    # Save it
    output_path = save_anki_deck(package, "outputs/french_greetings.apkg")
    print(f"Saved deck to: {output_path}")


def example_reverse_direction():
    """Create a deck learning English from French."""
    print("\n=== Example 2: English from French ===")

    # Get phrases with both English and French
    phrases = [
        get_phrase_by_english("The cat is sleeping"),
        get_phrase_by_english("The dog is running"),
    ]

    # Ensure French translations exist
    for phrase in phrases:
        if "fr-FR" not in phrase.translations:
            phrase.translate("fr-FR")

    # Create deck with French as source, English as target
    # This creates cards where you see French and learn English
    package = create_anki_deck(
        phrases=phrases,
        source_language="fr-FR",  # What you know
        target_language="en-GB",  # What you're learning
        deck_name="English::From French::Animals",
    )

    save_anki_deck(package, "outputs/english_from_french.apkg")


def example_japanese_deck():
    """Create a Japanese deck from English."""
    print("\n=== Example 3: Japanese Deck ===")

    # Get phrases
    phrases = [
        get_phrase_by_english("Good morning"),
        get_phrase_by_english("Good evening"),
    ]

    # Add Japanese translations
    for phrase in phrases:
        if "ja-JP" not in phrase.translations:
            print(f"Translating to Japanese: {phrase.english}")
            phrase.translate("ja-JP")
            phrase.generate_audio("flashcard", "ja-JP")

    # Create and save in one step
    create_and_save_anki_deck(
        phrases=phrases,
        source_language="en-GB",
        target_language="ja-JP",
        output_path="outputs/japanese_greetings.apkg",
        deck_name="Japanese::Beginner::Greetings",
    )


def example_multiple_decks():
    """Create multiple decks from the same phrases in different languages."""
    print("\n=== Example 4: Multiple Language Decks ===")

    # Get a set of phrases
    phrases = [
        get_phrase_by_english("I like coffee"),
        get_phrase_by_english("I like tea"),
        get_phrase_by_english("I like water"),
    ]

    # Add translations in multiple languages
    target_languages = ["fr-FR", "es-ES", "de-DE"]

    for lang in target_languages:
        for phrase in phrases:
            if lang not in phrase.translations:
                print(f"Translating '{phrase.english}' to {lang}")
                phrase.translate(lang)
                phrase.generate_audio("flashcard", lang)

    # Create separate decks for each language
    for lang in target_languages:
        lang_obj = BCP47Language.get(lang)
        lang_name = lang_obj.display_name("en")

        deck_name = f"{lang_name}::Beginner::Food and Drink"
        output_path = f"outputs/{lang.lower().replace('-', '_')}_food.apkg"

        create_and_save_anki_deck(
            phrases=phrases,
            source_language="en-GB",
            target_language=lang,
            output_path=output_path,
            deck_name=deck_name,
        )
        print(f"Created {lang_name} deck")


def example_custom_deck_hierarchy():
    """Create a deck with custom hierarchical naming."""
    print("\n=== Example 5: Custom Deck Hierarchy ===")

    # Get phrases for a specific topic
    phrases = [
        get_phrase_by_english("One"),
        get_phrase_by_english("Two"),
        get_phrase_by_english("Three"),
    ]

    # Add Spanish translations
    for phrase in phrases:
        if "es-ES" not in phrase.translations:
            phrase.translate("es-ES")
            phrase.generate_audio("flashcard", "es-ES")

    # Use Anki's deck hierarchy with ::
    # This creates: Spanish > A1 Level > Numbers > Basics
    create_and_save_anki_deck(
        phrases=phrases,
        source_language="en-GB",
        target_language="es-ES",
        output_path="outputs/spanish_numbers.apkg",
        deck_name="Spanish::A1 Level::Numbers::Basics",
    )


def example_with_phrase_search():
    """Create a deck using phrase search functionality."""
    print("\n=== Example 6: Using Phrase Search ===")

    # This example assumes you have the phrase search module
    # from src.phrases.search import search_phrases

    # Search for phrases containing specific words
    # phrases = search_phrases(
    #     search_text="weather",
    #     max_results=10
    # )

    # For now, use direct phrase retrieval
    phrases = [
        get_phrase_by_english("The weather is nice today"),
        get_phrase_by_english("It's raining"),
    ]

    # Create German deck
    for phrase in phrases:
        if "de-DE" not in phrase.translations:
            phrase.translate("de-DE")
            phrase.generate_audio("flashcard", "de-DE")

    create_and_save_anki_deck(
        phrases=phrases,
        source_language="en-GB",
        target_language="de-DE",
        output_path="outputs/german_weather.apkg",
        deck_name="German::Topics::Weather",
    )


def main():
    """Run all examples."""
    print("=" * 60)
    print("Anki Deck Creation Examples")
    print("=" * 60)

    # Uncomment the examples you want to run
    # example_basic_deck()
    # example_reverse_direction()
    # example_japanese_deck()
    # example_multiple_decks()
    # example_custom_deck_hierarchy()
    # example_with_phrase_search()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
