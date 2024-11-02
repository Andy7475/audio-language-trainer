import csv
import io
import json
import os
import shutil
import sqlite3
import tempfile
import urllib.parse
import uuid
import zipfile
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple

import genanki
import requests
import spacy
from bs4 import BeautifulSoup
from google.cloud import texttospeech
from pydub import AudioSegment

from anki.collection import Collection
from anki.models import NotetypeDict
from src.audio_generation import async_process_phrases
from src.config_loader import config
from src.dialogue_generation import update_vocab_usage
from src.generate import add_audio, add_translations
from src.utils import clean_filename, create_test_story_dict, string_to_large_int
import pysnooper


class AnkiCollectionReader:
    def __init__(self, collection_path: str):
        """Initialize connection to an Anki collection using the official anki package

        Args:
            collection_path: Path to the .anki2 collection file
        """
        self.collection_path = collection_path
        if not os.path.exists(self.collection_path):
            raise FileNotFoundError(
                f"Anki collection not found at: {self.collection_path}"
            )
        self.col: Optional[Collection] = None

    def connect(self):
        """Connect to the Anki collection"""
        self.col = Collection(self.collection_path)
        return self

    def close(self):
        """Close the collection connection"""
        if self.col:
            self.col.close()
            self.col = None

    def get_deck_names(self) -> Dict[int, str]:
        """Get all deck names and their IDs"""
        if not self.col:
            raise RuntimeError("Not connected to collection")

        decks = {}
        for deck_id, deck in self.col.decks.all():
            decks[deck_id] = deck["name"]
        return decks

    def get_notes_for_deck(self, deck_name: str) -> List[Dict[str, Any]]:
        """Get all notes for a specific deck

        Args:
            deck_name: Name of the deck to get notes from

        Returns:
            List of dictionaries containing note data
        """
        if not self.col:
            raise RuntimeError("Not connected to collection")

        # Get deck ID
        deck = self.col.decks.by_name(deck_name)
        if not deck:
            raise ValueError(f"Deck '{deck_name}' not found")

        # Find cards in deck
        card_ids = self.col.find_cards(f"deck:{deck_name}")

        # Get unique note IDs from cards
        note_ids = set()
        for card_id in card_ids:
            card = self.col.get_card(card_id)
            note_ids.add(card.nid)

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

    def get_media_files(self) -> Dict[str, str]:
        """Get the media files mapping"""
        if not self.col:
            raise RuntimeError("Not connected to collection")

        media_dir = os.path.dirname(self.collection_path)
        media_file = os.path.join(media_dir, "media")
        if os.path.exists(media_file):
            with open(media_file, "r") as f:
                return json.load(f)
        return {}

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


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


async def create_anki_deck_from_english_phrase_list(
    phrase_list: List[str],
    deck_name: str,
    anki_filename_prefix: str,
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
        translated_phrases_dict_audio = await add_audio(partial_dict)

        export_to_anki(
            translated_phrases_dict_audio,
            output_dir,
            f"{anki_filename_prefix}_{from_index}",
            deck_name=deck_name,
        )


def generate_wiktionary_links(
    phrase: str, language_name: str = config.language_name
) -> str:
    words = phrase.split()
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


def export_to_anki_with_images(
    story_data_dict: Dict[str, Dict],
    output_dir: str,
    story_name: str,
    deck_name: str = None,
):
    """
    Export story data to an Anki deck, including images for each card.
    """
    os.makedirs(output_dir, exist_ok=True)

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

    # (Card templates and styles remain the same...)
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
                "qfmt": f"""
                <div class="picture-container">{{{{Picture}}}}</div>
                <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
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
                <div class="picture-container">{{{{Picture}}}}</div>
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
                <div class="picture-container">{{Picture}}</div>
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
                <div class="picture-container">{{Picture}}</div>
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
        css=COMMON_CSS
        + """
        .picture-container {
            margin-bottom: 20px;
            text-align: center;
        }
        .picture-container img {
            max-width: 90%;
            max-height: 300px;
            object-fit: contain;
        }
        """,
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
        for (english, target), audio_segment, image_path in zip(
            phrase_pairs, audio_segments, image_paths
        ):
            # Handle image if it exists
            image_filename = None
            if image_path is not None:
                try:
                    image_filename = f"{uuid.uuid4()}.png"
                    shutil.copy2(image_path, os.path.join(output_dir, image_filename))
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
    for note in notes:
        deck.add_note(note)

    # Create and save the package
    package = genanki.Package(deck)
    package.media_files = [os.path.join(output_dir, file) for file in media_files]
    output_filename = os.path.join(output_dir, f"{story_name}_anki_deck.apkg")
    package.write_to_file(output_filename)
    print(f"Anki deck exported to {output_filename}")

    # Clean up temporary files
    for media_file in media_files:
        file_path = os.path.join(output_dir, media_file)
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

    print("Cleanup of temporary files completed.")
