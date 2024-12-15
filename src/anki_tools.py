import csv
import io
import json
import os
import re
import shutil
import sqlite3
import tempfile
import urllib.parse
import uuid
import zipfile
from collections import defaultdict
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple

import genanki
import pysnooper
import requests
from anki.collection import Collection
from anki.models import NotetypeDict
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image
from pydub import AudioSegment
from tqdm import tqdm

from src.config_loader import config
from src.dialogue_generation import update_vocab_usage
from src.generate import add_audio, add_translations
from src.images import add_image_paths
from src.translation import tokenize_text, translate_from_english
from src.utils import (
    clean_filename,
    create_test_story_dict,
    string_to_large_int,
)


def convert_anki_to_story_dict(collection_path: str, deck_name: str) -> Dict[str, Dict]:
    """
    Read an Anki deck and convert it to the story_data_dict format used by export_anki_with_images

    Args:
        collection_path: Path to the .anki2 collection file
        deck_name: Name of the deck to convert

    Returns:
        Dictionary in the story_data_dict format with phrases and audio
    """
    story_data_dict = defaultdict(lambda: defaultdict(list))

    # Create a temporary directory for media files
    with tempfile.TemporaryDirectory() as temp_dir:
        with AnkiCollectionReader(collection_path) as reader:
            # Get the media files mapping
            media_dir = reader.get_media_dir()

            # Get notes from the deck
            notes = reader.get_notes_for_deck(deck_name)

            # Process each note
            phrase_list = []
            audio_segments = []

            for note in tqdm(notes):
                # Extract fields from note
                fields = note["fields"]

                # Get target and English text
                target_text = fields.get("TargetText", "").strip()
                english_text = fields.get("EnglishText", "").strip()

                if not target_text or not english_text:
                    continue

                # Extract audio filenames from the audio field
                target_audio = fields.get("TargetAudio", "")
                target_audio_slow = fields.get("TargetAudioSlow", "")

                # Extract sound filenames using regex
                audio_pattern = r"\[sound:(.*?)\]"
                normal_audio_match = re.search(audio_pattern, target_audio)
                slow_audio_match = re.search(audio_pattern, target_audio_slow)

                if normal_audio_match and slow_audio_match:
                    normal_audio_file = normal_audio_match.group(1)
                    slow_audio_file = slow_audio_match.group(1)

                    # Get the full paths for the audio files
                    normal_audio_path = os.path.join(media_dir, normal_audio_file)
                    slow_audio_path = os.path.join(media_dir, slow_audio_file)

                    # Load audio segments if files exist
                    try:
                        normal_segment = AudioSegment.from_file(normal_audio_path)
                        slow_segment = AudioSegment.from_file(slow_audio_path)

                        # Add phrase and audio to lists
                        phrase_list.append((english_text, target_text))
                        audio_segments.append([None, slow_segment, normal_segment])

                    except Exception as e:
                        print(f"Error loading audio for note {note['id']}: {str(e)}")
                        continue

            # Add all phrases and audio to the story dict under a single part
            english_phrases = [english for (english, target) in phrase_list]
            if phrase_list and audio_segments:
                story_data_dict["part_1"]["translated_phrase_list"] = phrase_list
                story_data_dict["part_1"][
                    "translated_phrase_list_audio"
                ] = audio_segments
                story_data_dict["part_1"]["corrected_phrase_list"] = english_phrases

    return dict(story_data_dict)


def extract_audio_filename(field_value: str) -> str:
    """Extract the audio filename from an Anki field value containing [sound:] tags"""
    match = re.search(r"\[sound:(.*?)\]", field_value)
    if match:
        return match.group(1)
    return None


def get_anki_path() -> str:
    """
    Get Anki collection path from environment variable with error checking.

    Returns:
        str: Path to Anki collection

    Raises:
        EnvironmentError: If ANKI_COLLECTION_PATH not set or file doesn't exist
    """
    load_dotenv()
    path = os.getenv("ANKI_COLLECTION_PATH")
    if not path:
        raise EnvironmentError("ANKI_COLLECTION_PATH not set in environment variables")
    if not os.path.exists(path):
        raise EnvironmentError(f"Anki collection not found at: {path}")
    return path


def validate_anki_tag(tag: str) -> bool:
    """
    Validate if a tag is acceptable for Anki.

    Args:
        tag: String to validate as an Anki tag

    Returns:
        bool: True if tag is valid, False otherwise
    """
    if not tag or " " in tag:
        return False
    return True


def append_tag_to_note(existing_tags: str, new_tag: str) -> str:
    """
    Append a new tag to existing space-separated tags if not already present.

    Args:
        existing_tags: Space-separated string of existing tags
        new_tag: Tag to append

    Returns:
        str: Updated space-separated tags string
    """
    tags = set(existing_tags.split()) if existing_tags else set()
    tags.add(new_tag)
    return " ".join(sorted(tags))


def add_tag_to_matching_notes(
    phrases: List[str], tag: str, deck_name: str, collection_path: Optional[str] = None
) -> Tuple[int, List[str]]:
    """
    Add tag to notes where EnglishText matches any of the provided phrases.
    Uses ANKI_COLLECTION_PATH from environment if collection_path not provided.

    Args:
        phrases: List of English phrases to match against
        tag: Tag to add to matching notes
        deck_name: Name of the deck to process
        collection_path: Optional override for collection path

    Returns:
        tuple[int, list[str]]: (Number of notes updated, List of errors if any)
    """
    if collection_path is None:
        collection_path = get_anki_path()

    if not validate_anki_tag(tag):
        raise ValueError("Invalid tag - must not contain spaces")

    errors = []
    updates = 0

    try:
        with AnkiCollectionReader(collection_path) as reader:
            notes = reader.get_notes_for_deck(deck_name)

            for note in notes:
                english_text = note["fields"].get("EnglishText", "").strip()

                if english_text in phrases:
                    try:
                        current_tags = " ".join(note["tags"]) if note["tags"] else ""
                        updated_tags = append_tag_to_note(current_tags, tag)

                        anki_note = reader.col.get_note(note["id"])
                        anki_note.tags = updated_tags.split()
                        # Replace flush() with update_note()
                        reader.col.update_note(anki_note)

                        updates += 1

                    except Exception as e:
                        errors.append(f"Error updating note {note['id']}: {str(e)}")

            reader.col.save()

    except Exception as e:
        errors.append(f"Error accessing collection: {str(e)}")

    return updates, errors


class AnkiCollectionReader:
    def __init__(self, collection_path: str = None):
        self.collection_path = os.getenv("ANKI_COLLECTION_PATH", collection_path)
        if not os.path.exists(self.collection_path):
            raise FileNotFoundError(
                f"Anki collection not found at: {self.collection_path}"
            )
        self.col: Optional[Collection] = None

    def connect(self):
        self.col = Collection(self.collection_path)
        return self

    def close(self):
        if self.col:
            self.col.close()
            self.col = None

    def get_deck_names(self) -> Dict[int, str]:
        """returns a dictionary of deck_id : deck_name"""
        if not self.col:
            raise RuntimeError("Not connected to collection")

        decks = {}
        for deck in self.col.decks.all():
            decks[deck["id"]] = deck["name"]
        return decks

    def get_notes_for_deck(self, deck_name: str) -> List[Dict[str, Any]]:
        if not self.col:
            raise RuntimeError("Not connected to collection")

        # Get deck dictionary
        deck = self.col.decks.by_name(deck_name)
        if not deck:
            raise ValueError(f"Deck '{deck_name}' not found")

        # Get unique note IDs from cards
        note_ids = self.col.find_notes(f"did:{deck['id']}")

        # Get note data
        notes = []
        for note_id in note_ids:
            note = self.col.get_note(note_id)
            notetype = self.col.models.get(note.mid)

            note_dict = {
                "id": note_id,
                "guid": note.guid,
                "model_name": notetype["name"],
                "tags": note.tags,
                "fields": {name: value for name, value in note.items()},
                "flags": note.flags,
            }
            notes.append(note_dict)

        return notes

    def get_media_dir(self) -> str:
        """Get the path to the media directory"""
        if not self.col:
            raise RuntimeError("Not connected to collection")

        return os.path.join(os.path.dirname(self.collection_path), "collection.media")

    def list_media_files(self) -> List[str]:
        """List all files in the media directory"""
        media_dir = self.get_media_dir()
        if os.path.exists(media_dir):
            return [f for f in os.listdir(media_dir) if not f.startswith("_")]
        return []

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def print_deck_info(collection_path: str):
    with AnkiCollectionReader(collection_path) as reader:
        # Print all decks
        print("\nAvailable decks:")
        print("-" * 50)
        for deck_id, deck_name in reader.get_deck_names().items():
            print(f"Deck ID: {deck_id}")
            print(f"Deck Name: {deck_name}")
            print("-" * 50)

        # Print media info
        print("\nMedia directory:", reader.get_media_dir())
        media_files = reader.list_media_files()
        print(f"Number of media files: {len(media_files)}")
        if media_files:
            print("\nFirst few media files:")
            for file in media_files[:5]:
                print(f"- {file}")


import pandas as pd
from typing import Optional, Dict, Any


def get_deck_contents(
    deck_name: str,
    collection_path: Optional[str] = None,
    fields_to_extract: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Get contents of a specific Anki deck as a pandas DataFrame.

    Args:
        deck_name: Name of the deck to analyze
        collection_path: Optional path to collection file. Uses ANKI_COLLECTION_PATH from env if None
        fields_to_extract: Optional list of field names to extract. If None, extracts all fields.

    Returns:
        pd.DataFrame with columns for each note field plus model_name and tags

    Raises:
        EnvironmentError: If collection path not found
        ValueError: If deck not found
    """
    try:
        with AnkiCollectionReader(collection_path) as reader:
            notes = reader.get_notes_for_deck(deck_name)

            if not notes:
                return pd.DataFrame()  # Return empty DataFrame if no notes

            # Initialize list to store note data
            note_data = []

            for note in notes:
                # Start with basic note info
                note_dict = {
                    "note_id": note["id"],
                    "model_name": note["model_name"],
                    "tags": " ".join(note["tags"]),
                }

                # Add fields based on fields_to_extract or all fields if None
                fields = note["fields"]
                if fields_to_extract:
                    for field in fields_to_extract:
                        note_dict[field] = fields.get(field, "")
                else:
                    note_dict.update(fields)

                note_data.append(note_dict)

            # Create DataFrame
            df = pd.DataFrame(note_data)

            # Reorder columns to put note_id, model_name, and tags first
            first_cols = ["note_id", "model_name", "tags"]
            other_cols = [col for col in df.columns if col not in first_cols]
            df = df[first_cols + other_cols]

            return df

    except Exception as e:
        raise Exception(f"Error getting deck contents: {str(e)}")


def export_to_anki_with_images_english(
    english_phrases: List[str],
    output_dir: str,
    image_dir: str,
    audio_dir: str,
    story_name: str,
    deck_name: str = None,
):
    """
    Export English learning flashcards to an Anki deck, with images and audio.
    This function assumes:
    1. Images exist at output_dir/{clean_filename(phrase)}.png
    2. Audio exists at output_dir/{clean_filename(phrase)}.mp3
    3. Slow audio exists at output_dir/{clean_filename(phrase)}_slow.mp3

    Args:
        english_phrases: List of English phrases to create cards for
        output_dir: Directory containing image and audio files
        story_name: Name for the generated Anki deck file
        deck_name: Optional custom deck name
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define card styles and scripts
    card_back = """
        <script>
        function copyText(element) {
            var textToCopy = element.textContent;
            navigator.clipboard.writeText(textToCopy).then(function() {
                element.classList.add('copied');
                setTimeout(function() {
                    element.classList.remove('copied');
                }, 1000);
            }).catch(function(err) {
                console.error('Failed to copy text: ', err);
            });
        }
        </script>
        """

    # Create model for English learning cards
    english_learning_model = genanki.Model(
        1607392314,  # New model ID to distinguish from original
        "English Learning With Images",
        fields=[
            {"name": "EnglishText"},  # Swapped to primary position
            {"name": "EnglishAudio"},  # Was TargetAudio
            {"name": "EnglishAudioSlow"},  # Was TargetAudioSlow
            {"name": "NativeText"},  # Was EnglishText
            {"name": "WiktionaryLinks"},
            {"name": "Picture"},
        ],
        templates=[
            {
                "name": "Listening Card",
                "qfmt": f"""
                <div class="picture-container">{{{{Picture}}}}</div>
                <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
                    <div>
                        Normal speed:
                        <br>
                        {{{{EnglishAudio}}}}
                    </div>
                    <div>
                        Slow speed:
                        <br>
                        {{{{EnglishAudioSlow}}}}
                    </div>
                </div>""",
                "afmt": f"""
                <hr id="answer">
                <div class="picture-container">{{{{Picture}}}}</div>
                <div class="english-text" onclick="copyText(this)">{{{{EnglishText}}}}</div>
                <div class="native-text">{{{{NativeText}}}}</div>
                <div>
                    Normal speed: {{{{EnglishAudio}}}}
                </div>
                <div class="wiktionary-links">
                {{{{WiktionaryLinks}}}}
                </div>
                {card_back}
                """,
            },
            {
                "name": "Reading Card",
                "qfmt": """
                <div class="picture-container">{{Picture}}</div>
                <div class="english-text" onclick="copyText(this)">{{EnglishText}}</div>
                """,
                "afmt": f"""
                {{{{FrontSide}}}}
                <hr id="answer">
                <div class="native-text">{{{{NativeText}}}}</div>
                <div>
                    {{{{EnglishAudio}}}}
                </div>
                <div class="wiktionary-links">
                {{{{WiktionaryLinks}}}}
                </div>
                {card_back}
                """,
            },
            {
                "name": "Speaking Card",
                "qfmt": """
                <div class="picture-container">{{Picture}}</div>
                <div class="native-text">{{NativeText}}</div>
                """,
                "afmt": f"""
                {{{{FrontSide}}}}
                <hr id="answer">
                <div class="english-text" onclick="copyText(this)">{{{{EnglishText}}}}</div>
                <div>
                    {{{{EnglishAudio}}}}
                </div>
                <div class="wiktionary-links">
                {{{{WiktionaryLinks}}}}
                </div>
                {card_back}
                """,
            },
        ],
        css="""
        .card {
            font-family: Arial, sans-serif;
            font-size: 20px;
            text-align: center;
            color: black;
            background-color: white;
        }
        .english-text {
            font-size: 28px;
            margin: 20px 0;
            font-weight: bold;
            cursor: pointer;
            position: relative;
        }
        .native-text {
            font-size: 22px;
            margin: 15px 0;
            font-weight: bold;
        }
        .picture-container {
            margin-bottom: 20px;
            text-align: center;
        }
        .picture-container img {
            max-width: 90%;
            max-height: 300px;
            object-fit: contain;
        }
        .wiktionary-links a {
            display: inline-block;
            margin: 5px;
            padding: 5px 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
            text-decoration: none;
            color: #333;
        }
        """,
    )

    media_files = []
    notes = []

    # Set up deck name and ID
    if deck_name is None:
        deck_id = string_to_large_int("english_learning_" + config.TARGET_LANGUAGE_NAME)
        deck_name = f"English Learning - {config.TARGET_LANGUAGE_NAME} speakers"
    else:
        deck_id = string_to_large_int(deck_name)
    deck = genanki.Deck(deck_id, deck_name)

    # Process each English phrase
    for english_phrase in english_phrases:
        # Generate filenames from separate directories
        base_filename = clean_filename(english_phrase)
        image_path = os.path.join(image_dir, f"{base_filename}.png")
        audio_path = os.path.join(audio_dir, f"{base_filename}.mp3")
        audio_slow_path = os.path.join(audio_dir, f"{base_filename}_slow.mp3")

        # Skip if required files don't exist
        if not all(
            os.path.exists(p) for p in [image_path, audio_path, audio_slow_path]
        ):
            print(f"Skipping {english_phrase} - missing required files")
            continue

        # Generate unique filenames for the Anki package
        image_filename = f"{uuid.uuid4()}.png"
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_slow_filename = f"{uuid.uuid4()}.mp3"

        # Copy files to new locations with unique names
        for src, dest in [
            (image_path, os.path.join(output_dir, image_filename)),
            (audio_path, os.path.join(output_dir, audio_filename)),
            (audio_slow_path, os.path.join(output_dir, audio_slow_filename)),
        ]:
            shutil.copy2(src, dest)
            media_files.append(dest.split(os.path.sep)[-1])

        # Translate English phrase to native language
        native_text = translate_from_english(english_phrase)

        # Generate Wiktionary links for the English phrase
        wiktionary_links = generate_wiktionary_links(english_phrase, "English")

        # Create note
        note = genanki.Note(
            model=english_learning_model,
            fields=[
                english_phrase,  # EnglishText
                f"[sound:{audio_filename}]",  # EnglishAudio
                f"[sound:{audio_slow_filename}]",  # EnglishAudioSlow
                native_text[0],  # NativeText
                wiktionary_links,  # WiktionaryLinks
                f'<img src="{image_filename}">',  # Picture
            ],
            guid=string_to_large_int(english_phrase + "_english_learning"),
        )
        notes.append(note)

    # Sort and shuffle notes
    shuffle(notes)
    notes.sort(
        key=lambda note: len(note.fields[0].split())
    )  # Sort by English text length

    # Add notes to deck
    for note in notes:
        deck.add_note(note)

    # Create and save the package
    package = genanki.Package(deck)
    package.media_files = [os.path.join(output_dir, file) for file in media_files]
    output_filename = os.path.join(output_dir, f"{story_name}_english_anki_deck.apkg")
    package.write_to_file(output_filename)
    print(f"English learning Anki deck exported to {output_filename}")

    # Clean up temporary files
    for media_file in media_files:
        file_path = os.path.join(output_dir, media_file)
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

    print("Cleanup of temporary files completed.")


def create_anki_deck_from_english_phrase_list(
    phrase_list: List[str],
    deck_name: str,
    anki_filename_prefix: str,
    image_dir: str,
    batch_size: int = 50,
    output_dir="../outputs/longman",
):
    """Takes a list of english phrases and does: 1) translation 2) text to speech 3) export to Anki deck.
    To avoid overloading the text-to-speech APIs it will batch up the phrases into smaller decks (*.apkg), but these will all have the same 'deck_id'
    and so when you import them into Anki they will merge into the same deck. The decks will be called {anki_filename_prefix}_{from_index}.apkg
    """

    phrase_dict = {
        "anki": {"corrected_phrase_list": phrase_list}
    }  # this format is because it's the same as our story_dictionary
    translated_phrases_dict = add_translations(
        phrase_dict
    )  # this already batches so can pass entire list

    # we will now create slices of the main phrase dictionary using a 'from_index' and batch_size
    for from_index in range(0, len(phrase_list), batch_size):
        partial_dict = create_test_story_dict(
            translated_phrases_dict,
            story_parts=1,
            phrases=batch_size,
            from_index=from_index,
        )
        translated_phrases_dict_audio = add_audio(partial_dict)

        if image_dir:
            translated_phrases_dict_audio = add_image_paths(
                translated_phrases_dict_audio, image_dir
            )
            export_to_anki_with_images(
                translated_phrases_dict_audio,
                output_dir,
                f"{anki_filename_prefix}_{from_index}",
                deck_name=deck_name,
            )
        else:
            raise ValueError("Missing an image directory (image_dir)")
    return translated_phrases_dict_audio


def generate_wiktionary_links_non_english(
    phrase: str, native_language_code: str = "uk"
) -> str:
    """
    Generate Wiktionary links for native speakers of other languages learning English.
    Similar format to the original generate_wiktionary_links function.

    Args:
        phrase: The English phrase to generate links for
        native_language_code: The two-letter language code (e.g., 'uk' for Ukrainian)

    Returns:
        HTML string with Wiktionary links in the native language
    """
    words = phrase.split()
    links: List[str] = []

    # Get translation of "English" in target language
    try:
        english_in_target = translate_from_english("English", native_language_code)
        if isinstance(english_in_target, list):
            english_in_target = english_in_target[0]
        english_in_target = english_in_target.capitalize()
    except Exception:
        # Fallback to "English" if translation fails
        english_in_target = "English"

    for word in words:
        clean_word = "".join(char for char in word if char.isalnum())
        if clean_word:
            # Lowercase the word for URL and search, but keep original for display
            lowercase_word = clean_word.lower()
            # URL encode the lowercase word to handle non-ASCII characters
            encoded_word = urllib.parse.quote(lowercase_word)
            # First try native language Wiktionary
            native_url = (
                f"https://{native_language_code}.wiktionary.org/wiki/{encoded_word}"
            )
            print(f"native url is: {native_url}")

            try:
                response = requests.get(native_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    # Look for the English section using h2 tag
                    language_section = None
                    for heading_level in range(1, 7):
                        if soup.find(f"h{heading_level}", {"id": english_in_target}):
                            language_section = True
                            break

                    if language_section:
                        # If found, create a link with the anchor to the specific language section
                        link = f'<a href="{native_url}#{english_in_target}">{word}</a>'
                        links.append(link)
                    else:
                        # If not found in native Wiktionary, try English Wiktionary
                        english_url = f"https://en.wiktionary.org/wiki/{encoded_word}"
                        link = f'<a href="{english_url}#English">{word}</a>'
                        links.append(link)
                else:
                    # If native Wiktionary fails, use English Wiktionary
                    english_url = f"https://en.wiktionary.org/wiki/{encoded_word}"
                    link = f'<a href="{english_url}#English">{word}</a>'
                    links.append(link)
            except requests.RequestException:
                # If request fails, add without link
                links.append(word)
        else:
            links.append(word)

    return " ".join(links)


def get_wiktionary_language_name(language_name: str) -> str:
    """Map standard language names to Wiktionary header names"""
    wiktionary_mapping = {
        "Mandarin Chinese": "Chinese",
        "Modern Greek": "Greek",
        "Standard Arabic": "Arabic",
        "Brazilian Portuguese": "Portuguese",
        # Add more mappings as discovered
    }
    return wiktionary_mapping.get(language_name, language_name)


def find_language_section(soup: BeautifulSoup, language_name: str) -> Optional[str]:
    """Try different strategies to find the language section"""
    # Try exact match with mapping
    wiktionary_name = get_wiktionary_language_name(language_name)
    if section := soup.find("h2", {"id": wiktionary_name}):
        return wiktionary_name

    # Try words in reverse order (longest to shortest)
    words = language_name.split()
    for i in range(len(words), 0, -1):
        partial_name = " ".join(words[:i])
        if section := soup.find("h2", {"id": partial_name}):
            return partial_name

    # If still not found, try individual words
    for word in words:
        if section := soup.find("h2", {"id": word}):
            return word

    return None


def generate_wiktionary_links(
    phrase: str,
    language_name: str = None,
    language_code: str = None,
) -> str:

    if language_name is None:
        language_name = config.TARGET_LANGUAGE_NAME
    if language_code is None:
        language_code = config.TARGET_LANGUAGE_CODE
    words = tokenize_text(text=phrase, language_code=language_code)
    links: List[str] = []

    for word in words:
        clean_word = "".join(char for char in word if char.isalnum())
        if not clean_word:
            links.append(word)
            continue

        # Try to create wiktionary link
        try:
            # Lowercase the word for URL and search, but keep original for display
            lowercase_word = clean_word.lower()
            encoded_word = urllib.parse.quote(lowercase_word)
            url = f"https://en.wiktionary.org/wiki/{encoded_word}"

            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                if section_name := find_language_section(soup, language_name):
                    links.append(f'<a href="{url}#{section_name}">{word}</a>')
                    continue

            # If any step fails, fall back to original word
            links.append(word)

        except requests.RequestException:
            links.append(word)

    return " ".join(links)


def load_template(filename):
    # print(os.listdir())
    filename = os.path.join("..", "src", filename)
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def export_to_anki_with_images(
    story_data_dict: Dict[str, Dict],
    output_dir: str,
    story_name: str,
    deck_name: str = None,
):
    """
    Export story data to an Anki deck, including images for each card. Use add_image_paths
    with story_data_dict first to get image data.

    The story_name is used as a prefix for the anki file only
    if you want these merged with other decks, ensure the deck_name matches exactly
    as this is used to generate the deck id.
    """
    os.makedirs(output_dir, exist_ok=True)

    language_practice_model = genanki.Model(
        1607392313 + 121,
        "Language Practice With Images",
        fields=[
            {"name": "TargetText"},
            {"name": "TargetAudio"},
            {"name": "TargetAudioSlow"},
            {"name": "EnglishText"},
            {"name": "WiktionaryLinks"},
            {"name": "Picture"},
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
        css=load_template("card_styles.css"),
    )

    media_files = []
    notes = []

    # Set up deck name and ID
    if deck_name is None:
        deck_id = string_to_large_int(config.TARGET_LANGUAGE_NAME + "image")
        deck_name = f"{config.TARGET_LANGUAGE_NAME} - phrases with images"
    else:
        deck_id = string_to_large_int(deck_name + "image")
    deck = genanki.Deck(deck_id, deck_name)

    for story_part, data in story_data_dict.items():
        # Get all the relevant lists
        phrase_pairs = data["translated_phrase_list"]
        audio_segments = data["translated_phrase_list_audio"]
        image_paths = data.get(
            "image_path", [None] * len(phrase_pairs)
        )  # Default to None if missing

        # Zip all three lists together
        for (english, target), audio_segment, image_path in tqdm(
            zip(phrase_pairs, audio_segments, image_paths),
            desc="generating image and sound files",
        ):
            image_filename = None
            if image_path is not None:
                try:
                    image_filename = f"{uuid.uuid4()}.png"
                    output_path = os.path.join(output_dir, image_filename)
                    with Image.open(image_path) as img:
                        img = img.resize((400, 400))  # Simple and direct!
                        img.save(output_path, "PNG", optimize=True)
                    media_files.append(image_filename)
                except Exception as e:
                    print(f"Error copying image for {english}: {str(e)}")
                    image_filename = None

            # Handle audio
            target_audio_normal = f"{uuid.uuid4()}.mp3"
            target_audio_slow = f"{uuid.uuid4()}.mp3"

            if isinstance(audio_segment, AudioSegment):
                target_normal_audio_segment = audio_segment
                target_slow_audio_segment = audio_segment
            elif isinstance(audio_segment, List) and len(audio_segment) > 2:
                target_normal_audio_segment = audio_segment[2]
                target_slow_audio_segment = audio_segment[1]
            else:
                raise Exception(f"Unexpected audio format: {audio_segment}")

            # Export audio files
            target_normal_audio_segment.export(
                os.path.join(output_dir, target_audio_normal), format="mp3"
            )
            target_slow_audio_segment.export(
                os.path.join(output_dir, target_audio_slow), format="mp3"
            )
            media_files.extend([target_audio_normal, target_audio_slow])

            # Generate Wiktionary links
            wiktionary_links = generate_wiktionary_links(target)

            # Create note
            note = genanki.Note(
                model=language_practice_model,
                fields=[
                    target,
                    f"[sound:{target_audio_normal}]",
                    f"[sound:{target_audio_slow}]",
                    english,
                    wiktionary_links,
                    f'<img src="{image_filename}">' if image_filename else "",
                    config.TARGET_LANGUAGE_NAME,
                ],
                guid=string_to_large_int(target + "image"),
            )
            notes.append(note)

    # Sort and shuffle notes
    shuffle(notes)
    notes.sort(
        key=lambda note: len(note.fields[3].split())
    )  # Sort by English text length

    # Add notes to deck
    for note in tqdm(notes, desc="adding notes to deck"):
        deck.add_note(note)

    # Create and save the package
    package = genanki.Package(deck)
    package.media_files = [os.path.join(output_dir, file) for file in media_files]
    output_filename = os.path.join(output_dir, f"{story_name}_anki_deck.apkg")
    package.write_to_file(output_filename)
    print(f"Anki deck exported to {output_filename}")

    # Clean up temporary files
    for media_file in tqdm(media_files, desc="deleting temp files"):
        file_path = os.path.join(output_dir, media_file)
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

    print("Cleanup of temporary files completed.")
