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
from PIL import Image
from pydub import AudioSegment
from tqdm import tqdm

from src.config_loader import config
from src.dialogue_generation import update_vocab_usage
from src.generate import add_audio, add_translations
from src.translation import tokenize_text, translate_from_english
from src.utils import (
    add_image_paths,
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


class AnkiCollectionReader:
    def __init__(self, collection_path: str):
        self.collection_path = collection_path
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
        if not self.col:
            raise RuntimeError("Not connected to collection")

        decks = {}
        for deck_id, deck in self.col.decks.all():
            decks[deck_id] = deck["name"]
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


def print_deck_contents(collection_path: str, deck_name: str):
    """Print contents of a specific deck

    Args:
        collection_path: Path to the .anki2 collection file
        deck_name: Name of the deck to analyze
    """
    try:
        with AnkiCollectionReader(collection_path) as reader:
            print(f"\nAnalyzing deck: {deck_name}")

            # Get notes from the deck
            notes = reader.get_notes_for_deck(deck_name)
            print(f"\nFound {len(notes)} notes")

            # Print the first few notes in detail
            print("\nFirst few notes:")
            for i, note in enumerate(notes[:5]):
                print(f"\nNote {i+1}:")
                print(f"Model: {note['model_name']}")
                print("Fields:")
                for field_name, field_value in note["fields"].items():
                    # Truncate very long field values
                    display_value = (
                        field_value[:100] + "..."
                        if len(field_value) > 100
                        else field_value
                    )
                    print(f"  {field_name}: {display_value}")
                print("Tags:", " ".join(note["tags"]))

    except Exception as e:
        print(f"Error: {e}")
        raise


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
        deck_id = string_to_large_int("english_learning_" + config.language_name)
        deck_name = f"English Learning - {config.language_name} speakers"
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
        native_text = translate_from_english(english_phrase, config.TARGET_LANGUAGE)

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
    batch_size: int = 50,
    output_dir="../outputs/longman",
    image_dir: str = None,
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
            export_to_anki(
                translated_phrases_dict_audio,
                output_dir,
                f"{anki_filename_prefix}_{from_index}",
                deck_name=deck_name,
            )
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


def generate_wiktionary_links(
    phrase: str,
    language_name: str = config.language_name,
    language_code: str = config.TARGET_LANGUAGE,
) -> str:
    words = tokenize_text(text=phrase, language_code=language_code)
    links: List[str] = []

    for word in words:
        clean_word = "".join(char for char in word if char.isalnum())
        if clean_word:
            # Lowercase the word for URL and search, but keep original for display
            lowercase_word = clean_word.lower()
            # URL encode the lowercase word to handle non-ASCII characters
            encoded_word = urllib.parse.quote(lowercase_word)
            url = f"https://en.wiktionary.org/wiki/{encoded_word}"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    # Look for the language section using h2 tag
                    language_section = soup.find("h2", {"id": language_name})

                    if language_section:
                        # If found, create a link with the anchor to the specific language section
                        # Use the original word (with original capitalization) for display
                        link = f'<a href="{url}#{language_name}">{word}</a>'
                        links.append(link)
                    else:
                        # If not found, just add the original word without a link
                        links.append(word)
                else:
                    links.append(word)
            except requests.RequestException:
                links.append(word)
        else:
            links.append(word)

    return " ".join(links)


COMMON_CSS = """
    .card {
        font-family: Arial, sans-serif;
        font-size: 20px;
        text-align: center;
        color: black;
        background-color: white;
    }

    .target-text {
  font-size: 28px;
  margin: 20px 0;
  font-weight: bold;
  cursor: pointer;
  position: relative;
}

.target-text::after {
  content: 'Copied!';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  background-color: #4CAF50;
  color: white;
  padding: 5px 10px;
  border-radius: 5px;
  font-size: 14px;
  opacity: 0;
  transition: opacity 0.3s;
}

.target-text.copied::after {
  opacity: 1;
}
    .english-text {
        font-size: 22px;
        margin: 15px 0;
        font-weight: bold;
    }
    .wiktionary-links {
        margin-top: 20px;
    }
    .replay-button svg {
  width: 60px;
  height: 60px;
}
.replay-button svg circle {
  fill: #4CAF50;
}
.replay-button svg path {
  fill: white;
  stroke: none;
}
    .wiktionary-links a {
        display: inline-block;
        margin: 5px;
        padding: 10px 15px;
        background-color: #f0f0f0;
        border-radius: 5px;
        text-decoration: none;
        color: #333;
    }
    """


def export_to_anki(
    story_data_dict: Dict[str, Dict],
    output_dir: str,
    story_name: str,
    deck_name: str = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Common CSS for all card types

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

    language_practice_model = genanki.Model(
        1607392312,
        "Language Practice",
        fields=[
            {"name": "TargetText"},
            {"name": "TargetAudio"},
            {"name": "TargetAudioSlow"},  # New field for slow audio
            {"name": "EnglishText"},
            {"name": "WiktionaryLinks"},
        ],
        # ... (rest of the model definition)
        templates=[
            {
                "name": "Listening Card",
                "qfmt": f"""<div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div>
                Normal speed:
                <br>
                {{{{TargetAudio}}}}
            </div>
            <div>
                Slow speed:
                <br>
                {{{{TargetAudioSlow}}}}
            </div>
        </div>""",
                "afmt": f"""
        <hr id="answer">
        <div class="target-text" onclick="copyText(this)">{{{{TargetText}}}}</div>
        <div class="english-text">{{{{EnglishText}}}}</div>
        <div>
            Normal speed: {{{{TargetAudio}}}}
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
        <div class="target-text" onclick="copyText(this)">{{TargetText}}</div>
        """,
                "afmt": f"""
        {{{{FrontSide}}}}
        <hr id="answer">
        <div class="english-text">{{{{EnglishText}}}}</div>
        <div>
            {{{{TargetAudio}}}}
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
        <div class="english-text">{{EnglishText}}</div>
        """,
                "afmt": f"""
        {{{{FrontSide}}}}
        <hr id="answer">
        <div class="target-text" onclick="copyText(this)">{{{{TargetText}}}}</div>
        <div>
            {{{{TargetAudio}}}}
        </div>
        <div class="wiktionary-links">
        {{{{WiktionaryLinks}}}}
        </div>
        {card_back}
        """,
            },
        ],
        css=COMMON_CSS,
    )

    media_files = []
    notes = []
    if deck_name is None:
        deck_id = string_to_large_int(config.language_name)
        deck_name = f"{config.language_name} - phrases"
    else:
        deck_id = string_to_large_int(deck_name)
    deck = genanki.Deck(deck_id, deck_name)

    for _, data in story_data_dict.items():
        for (english, target), audio_segments in zip(
            data["translated_phrase_list"], data["translated_phrase_list_audio"]
        ):
            # Generate unique filename for audio
            target_audio_normal = f"{uuid.uuid4()}.mp3"
            target_audio_slow = f"{uuid.uuid4()}.mp3"

            # Export audio segment
            if isinstance(audio_segments, AudioSegment):
                target_normal_audio_segment = audio_segments
                target_slow_audio_segment = audio_segments
            elif isinstance(audio_segments, List) and len(audio_segments) > 2:
                target_normal_audio_segment = audio_segments[2]
                target_slow_audio_segment = audio_segments[1]
            else:
                raise Exception(f"Unexpected audio format: {audio_segments}")

            target_normal_audio_segment.export(
                os.path.join(output_dir, target_audio_normal), format="mp3"
            )
            target_slow_audio_segment.export(
                os.path.join(output_dir, target_audio_slow), format="mp3"
            )

            # Add to media files list
            media_files.extend([target_audio_normal, target_audio_slow])

            # Generate Wiktionary links
            wiktionary_links = generate_wiktionary_links(target, config.language_name)

            # Create notes for each card type
            note = genanki.Note(
                model=language_practice_model,
                fields=[
                    target,
                    f"[sound:{target_audio_normal}]",
                    f"[sound:{target_audio_slow}]",
                    english,
                    wiktionary_links,
                ],
                guid=string_to_large_int(target),
            )

            notes.append(note)
            # Add notes to the deck

    def sort_key(note):
        # Find the English field based on the note's model
        return 2  # english is the 2nd field (0 index)

        # Count words in the English phrase
        return len(note.fields[english_field_index].split())

    # shuffle the notes
    shuffle(notes)
    notes.sort(key=sort_key)
    for note in notes:
        deck.add_note(note)

    # Create a package with all decks
    # package = genanki.Package(list(decks.values()))
    package = genanki.Package(deck)
    package.media_files = [os.path.join(output_dir, file) for file in media_files]

    # Write the package to a file
    output_filename = os.path.join(output_dir, f"{story_name}_anki_deck.apkg")
    package.write_to_file(output_filename)

    print(f"Anki deck exported to {output_filename}")

    # Clean up temporary MP3 files
    for media_file in media_files:
        file_path = os.path.join(output_dir, media_file)
        try:
            os.remove(file_path)
            # print(f"Deleted temporary file: {file_path}")
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

    print("Cleanup of temporary MP3 files completed.")


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
        1607392313,
        "Language Practice With Images",
        fields=[
            {"name": "TargetText"},
            {"name": "TargetAudio"},
            {"name": "TargetAudioSlow"},
            {"name": "EnglishText"},
            {"name": "WiktionaryLinks"},
            {"name": "Picture"},
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
        # css=get_common_styles(),
    )

    media_files = []
    notes = []

    # Set up deck name and ID
    if deck_name is None:
        deck_id = string_to_large_int(config.language_name + "image")
        deck_name = f"{config.language_name} - phrases with images"
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
            wiktionary_links = generate_wiktionary_links(target, config.language_name)

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
