"""Modular Anki flashcard deck generation from Phrase models.

This module provides functions to create Anki decks from Phrase objects with support
for multiple source and target languages using BCP-47 language tags.

Key features:
- Create Anki notes from individual Phrase objects
- Build decks from lists of phrases
- Support any source/target language combination
- Download only target language audio and images
- Separate deck generation from file saving for flexibility

Example usage:
    >>> from src.phrases.phrase_model import get_phrase_by_english
    >>> from src.models import BCP47Language
    >>>
    >>> # Get some phrases
    >>> phrases = [get_phrase_by_english("Hello"), get_phrase_by_english("Goodbye")]
    >>>
    >>> # Create a deck
    >>> deck = create_anki_deck(
    ...     phrases=phrases,
    ...     source_language="en-GB",
    ...     target_language="fr-FR",
    ...     deck_name="French::Beginner::Greetings"
    ... )
    >>>
    >>> # Save the deck
    >>> save_anki_deck(deck, output_path="outputs/french_greetings.apkg")
"""

import os
import uuid
from pathlib import Path
from typing import List, Union, Optional
from tempfile import TemporaryDirectory

import genanki
from tqdm import tqdm

from src.models import BCP47Language, get_language
from src.phrases.phrase_model import Phrase
from src.utils import load_template


# ============================================================================
# ANKI MODEL DEFINITION
# ============================================================================

def get_anki_model(model_id: int = 1607392313, model_name: str = "FirePhrase") -> genanki.Model:
    """Get the Anki model (note type) for language learning flashcards.

    This model supports three card types:
    1. Listening Card: Hear audio, guess the meaning
    2. Reading Card: Read text, understand meaning
    3. Speaking Card: See English, speak target language

    Args:
        model_id: Unique model ID (default: 1607392313)
        model_name: Name of the model (default: "FirePhrase")

    Returns:
        genanki.Model: The Anki model with templates and styling
    """
    return genanki.Model(
        model_id,
        model_name,
        fields=[
            {"name": "SortOrder"},
            {"name": "SourceText"},
            {"name": "TargetText"},
            {"name": "TargetAudio"},
            {"name": "TargetAudioSlow"},
            {"name": "WiktionaryLinks"},
            {"name": "Picture"},
            {"name": "SourceLanguageName"},
            {"name": "TargetLanguageName"},
        ],
        templates=[
            {
                "name": "Listening Card",
                "qfmt": load_template("listening_card_front_template.html"),
                "afmt": load_template("card_back_template.html"),
            },
            {
                "name": "Reading Card",
                "qfmt": load_template("reading_card_front_template.html"),
                "afmt": load_template("card_back_template.html"),
            },
            {
                "name": "Speaking Card",
                "qfmt": load_template("speaking_card_front_template.html"),
                "afmt": load_template("card_back_template.html"),
            },
        ],
        css=load_template("card_styles.css", parent_path="../src/templates/styles"),
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_sort_field(order: int, text: str) -> str:
    """Create a unique sort field using order and text hash.

    Args:
        order: Integer for primary sort order (0-9999)
        text: Text used to generate a unique suffix

    Returns:
        str: Sort field like '0001-3f4a9' that will sort correctly in Anki
    """
    # Create a simple hash from the text
    text_hash = hex(hash(text) & 0xFFFFFFFF)[2:10]
    return f"{order:04d}-{text_hash}"


def _string_to_large_int(text: str) -> int:
    """Convert a string to a large integer for use as deck/note ID.

    Args:
        text: String to convert

    Returns:
        int: Large integer suitable for Anki IDs
    """
    # Use Python's hash function and ensure it's positive and large enough
    return abs(hash(text)) % (10 ** 10)


# ============================================================================
# CORE FUNCTIONS: PHRASE TO ANKI NOTE
# ============================================================================

def create_anki_note_from_phrase(
    phrase: Phrase,
    source_language: Union[str, BCP47Language],
    target_language: Union[str, BCP47Language],
    index: int,
    model: genanki.Model,
    temp_dir: str,
    wiktionary_links: Optional[str] = None,
    auto_generate_wiktionary: bool = True,
) -> tuple[genanki.Note, list[str]]:
    """Create an Anki note from a single Phrase object.

    Downloads target language audio and image (if needed) and creates a note with
    all necessary fields populated. Source and target can be any language combination.

    Args:
        phrase: Phrase object with translations
        source_language: BCP47 language tag for source (what user already knows)
        target_language: BCP47 language tag for target (what user is learning)
        index: Index/position of this phrase in the deck (for sorting)
        model: genanki Model to use for the note
        temp_dir: Temporary directory to store media files
        wiktionary_links: Optional HTML string with wiktionary links (if provided,
                         auto_generate_wiktionary is ignored)
        auto_generate_wiktionary: If True and wiktionary_links not provided,
                                 automatically generate from Translation.get_wiktionary_links()

    Returns:
        tuple: (genanki.Note, list of media file paths)

    Raises:
        ValueError: If source or target translation doesn't exist

    Example:
        >>> phrase = get_phrase_by_english("Hello")
        >>> phrase.translate("fr-FR")
        >>> note, media = create_anki_note_from_phrase(
        ...     phrase, "en-GB", "fr-FR", 0, model, "/tmp"
        ... )
    """
    # Normalize language parameters
    source_lang = get_language(source_language)
    target_lang = get_language(target_language)

    # Get translations
    source_tag = source_lang.to_tag()
    target_tag = target_lang.to_tag()

    if source_tag not in phrase.translations:
        raise ValueError(
            f"Phrase '{phrase.phrase_hash}' missing source translation: {source_tag}. "
            f"Available: {list(phrase.translations.keys())}"
        )

    if target_tag not in phrase.translations:
        raise ValueError(
            f"Phrase '{phrase.phrase_hash}' missing target translation: {target_tag}. "
            f"Available: {list(phrase.translations.keys())}"
        )

    source_translation = phrase.translations[source_tag]
    target_translation = phrase.translations[target_tag]

    # Generate wiktionary links if not provided and auto-generation enabled
    if wiktionary_links is None and auto_generate_wiktionary:
        try:
            wiktionary_links = target_translation.get_wiktionary_links()
        except Exception as e:
            print(f"Warning: Failed to generate wiktionary links for {phrase.phrase_hash}: {e}")
            wiktionary_links = ""

    # Download target language multimedia if not already loaded
    if target_translation.audio.get("flashcard", {}).get("normal") is None:
        phrase.download(language=target_lang, local=True)

    media_files = []

    # Handle image (download if needed)
    image_html = ""
    if target_translation.image is None and target_translation.image_file_path:
        phrase.get_image(language=target_lang, local=True)

    if target_translation.image is not None:
        image_filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join(temp_dir, image_filename)
        # Resize and save
        resized_img = target_translation.image.resize((400, 400))
        resized_img.save(image_path, "PNG", optimize=True)
        media_files.append(image_path)
        image_html = f'<img src="{image_filename}">'

    # Handle audio
    audio_normal_html = ""
    audio_slow_html = ""

    # Normal speed audio
    if "flashcard" in target_translation.audio and "normal" in target_translation.audio["flashcard"]:
        phrase_audio_normal = target_translation.audio["flashcard"]["normal"]
        if phrase_audio_normal.audio_segment is None:
            phrase_audio_normal.download(local=True)

        if phrase_audio_normal.audio_segment is not None:
            audio_normal_filename = f"{uuid.uuid4()}.mp3"
            audio_normal_path = os.path.join(temp_dir, audio_normal_filename)
            phrase_audio_normal.audio_segment.export(audio_normal_path, format="mp3")
            media_files.append(audio_normal_path)
            audio_normal_html = f"[sound:{audio_normal_filename}]"

    # Slow speed audio
    if "flashcard" in target_translation.audio and "slow" in target_translation.audio["flashcard"]:
        phrase_audio_slow = target_translation.audio["flashcard"]["slow"]
        if phrase_audio_slow.audio_segment is None:
            phrase_audio_slow.download(local=True)

        if phrase_audio_slow.audio_segment is not None:
            audio_slow_filename = f"{uuid.uuid4()}.mp3"
            audio_slow_path = os.path.join(temp_dir, audio_slow_filename)
            phrase_audio_slow.audio_segment.export(audio_slow_path, format="mp3")
            media_files.append(audio_slow_path)
            audio_slow_html = f"[sound:{audio_slow_filename}]"

    # Create the note
    note = genanki.Note(
        model=model,
        fields=[
            _get_sort_field(index, target_translation.text),  # SortOrder
            source_translation.text,  # SourceText
            target_translation.text,  # TargetText
            audio_normal_html,  # TargetAudio
            audio_slow_html,  # TargetAudioSlow
            wiktionary_links or "",  # WiktionaryLinks
            image_html,  # Picture
            source_lang.language.upper(),  # SourceLanguageName (e.g., "EN")
            target_lang.language.upper(),  # TargetLanguageName (e.g., "FR")
        ],
        guid=_string_to_large_int(f"{phrase.phrase_hash}_{source_tag}_{target_tag}"),
    )

    return note, media_files


# ============================================================================
# DECK CREATION
# ============================================================================

def create_anki_deck(
    phrases: List[Phrase],
    source_language: Union[str, BCP47Language],
    target_language: Union[str, BCP47Language],
    deck_name: Optional[str] = None,
    model: Optional[genanki.Model] = None,
    wiktionary_links_func: Optional[callable] = None,
) -> genanki.Package:
    """Create an Anki deck package from a list of Phrase objects.

    This is the main function for creating Anki decks. It handles:
    - Downloading multimedia for target language only
    - Creating notes for each phrase
    - Building the deck with proper hierarchy
    - Managing temporary files

    Args:
        phrases: List of Phrase objects to include in the deck
        source_language: BCP47 language tag for source (e.g., "en-GB", "ja-JP")
        target_language: BCP47 language tag for target (e.g., "fr-FR", "es-ES")
        deck_name: Name for the Anki deck (if None, auto-generates from languages)
        model: Custom genanki.Model to use (if None, uses default FirePhrase model)
        wiktionary_links_func: Optional function that takes a Phrase and returns
                               HTML string with wiktionary links

    Returns:
        genanki.Package: Package ready to be saved to .apkg file

    Raises:
        ValueError: If phrases list is empty or translations are missing

    Example:
        >>> phrases = [get_phrase_by_english("Hello"), get_phrase_by_english("Goodbye")]
        >>> for p in phrases:
        ...     p.translate("fr-FR")
        ...     p.generate_audio("flashcard", "fr-FR")
        >>>
        >>> package = create_anki_deck(
        ...     phrases=phrases,
        ...     source_language="en-GB",
        ...     target_language="fr-FR",
        ...     deck_name="French::Beginner::Greetings"
        ... )
        >>> save_anki_deck(package, "greetings.apkg")
    """
    if not phrases:
        raise ValueError("phrases list cannot be empty")

    # Normalize languages
    source_lang = get_language(source_language)
    target_lang = get_language(target_language)

    # Generate deck name if not provided
    if deck_name is None:
        source_name = source_lang.display_name("en")
        target_name = target_lang.display_name("en")
        deck_name = f"{target_name} (from {source_name})"

    # Get or create model
    if model is None:
        model = get_anki_model()

    # Create deck
    deck_id = _string_to_large_int(deck_name + "FirePhrase")
    deck = genanki.Deck(deck_id, deck_name)

    # Use temporary directory for media files
    with TemporaryDirectory() as temp_dir:
        all_media_files = []
        notes = []

        print(f"Creating Anki deck: {deck_name}")
        print(f"Source language: {source_lang.to_tag()}")
        print(f"Target language: {target_lang.to_tag()}")
        print(f"Processing {len(phrases)} phrases...")

        # Create notes from phrases
        for index, phrase in enumerate(tqdm(phrases, desc="Creating notes")):
            try:
                # Get wiktionary links if function provided
                wiktionary_links = None
                if wiktionary_links_func is not None:
                    wiktionary_links = wiktionary_links_func(phrase)

                # Create note
                note, media_files = create_anki_note_from_phrase(
                    phrase=phrase,
                    source_language=source_lang,
                    target_language=target_lang,
                    index=index,
                    model=model,
                    temp_dir=temp_dir,
                    wiktionary_links=wiktionary_links,
                )

                notes.append(note)
                all_media_files.extend(media_files)

            except Exception as e:
                print(f"Error processing phrase {phrase.phrase_hash}: {str(e)}")
                continue

        # Add notes to deck
        for note in tqdm(notes, desc="Adding notes to deck"):
            deck.add_note(note)

        # Create package
        package = genanki.Package(deck)
        package.media_files = all_media_files

        print(f"✅ Created deck with {len(notes)} notes")

        # Note: The package holds references to media files in temp_dir
        # Save must be called before temp_dir is destroyed
        # This is handled by save_anki_deck function
        return package


# ============================================================================
# DECK SAVING
# ============================================================================

def save_anki_deck(
    package: genanki.Package,
    output_path: str,
    create_dirs: bool = True,
) -> str:
    """Save an Anki deck package to a .apkg file.

    Args:
        package: genanki.Package to save
        output_path: Path where .apkg file should be saved
        create_dirs: Whether to create parent directories if they don't exist

    Returns:
        str: Absolute path to the saved .apkg file

    Example:
        >>> package = create_anki_deck(phrases, "en-GB", "fr-FR")
        >>> save_anki_deck(package, "outputs/french_deck.apkg")
        'c:/Users/.../outputs/french_deck.apkg'
    """
    # Ensure .apkg extension
    if not output_path.lower().endswith(".apkg"):
        output_path += ".apkg"

    # Create parent directories if needed
    if create_dirs:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save the package
    package.write_to_file(output_path)

    abs_path = os.path.abspath(output_path)
    print(f"✅ Saved Anki deck to: {abs_path}")

    return abs_path


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_and_save_anki_deck(
    phrases: List[Phrase],
    source_language: Union[str, BCP47Language],
    target_language: Union[str, BCP47Language],
    output_path: str,
    deck_name: Optional[str] = None,
    model: Optional[genanki.Model] = None,
) -> str:
    """Create and save an Anki deck in one step.

    Convenience function that combines create_anki_deck and save_anki_deck.

    Args:
        phrases: List of Phrase objects to include
        source_language: BCP47 language tag for source
        target_language: BCP47 language tag for target
        output_path: Path where .apkg file should be saved
        deck_name: Optional name for the deck
        model: Optional custom model

    Returns:
        str: Absolute path to the saved .apkg file

    Example:
        >>> phrases = [get_phrase_by_english("Hello")]
        >>> create_and_save_anki_deck(
        ...     phrases, "en-GB", "fr-FR", "outputs/french.apkg"
        ... )
    """
    package = create_anki_deck(
        phrases=phrases,
        source_language=source_language,
        target_language=target_language,
        deck_name=deck_name,
        model=model,
    )

    return save_anki_deck(package, output_path)
