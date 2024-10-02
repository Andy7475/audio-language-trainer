import csv
import io
import os
import urllib.parse
import uuid
from random import shuffle
from typing import Dict, List, Tuple

import genanki
import requests
import spacy
from bs4 import BeautifulSoup
from google.cloud import texttospeech
from pydub import AudioSegment

from src.audio_generation import async_process_phrases
from src.config_loader import config
from src.dialogue_generation import update_vocab_usage
from src.utils import string_to_large_int


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


def inspect_anki_deck(filename: str):
    separator = "\t"  # Default separator

    with open(filename, "r", encoding="utf-8") as f:
        # Process header information
        for line in f:
            if line.startswith("#"):
                if line.startswith("#separator:"):
                    separator = line.split(":")[1].strip()
                    if separator == "tab":
                        separator = "\t"
                print(f"Header: {line.strip()}")
            else:
                break  # Exit loop when we reach the data

        # Create a CSV reader with the determined separator
        reader = csv.reader(f, delimiter=separator)

        # Read the first data row
        first_row = next(reader)

        print("\nField inspection:")
        for i, field in enumerate(first_row):
            # Truncate long fields for display
            display_field = field[:50] + "..." if len(field) > 50 else field
            print(f"{i}: {display_field}")

        print(f"\nTotal number of fields: {len(first_row)}")


def import_anki_deck(
    file_path: str, english_field_index: int, target_field_index: int
) -> Tuple[List[str], List[str], List[Dict]]:
    """inspect the deck first to select the correct field_index values (inspect_anki_deck)
    Then we will get back a list of english_phrases and translated_phrases that we can feed into
    process_anki_deck function to generate the audio"""
    english_phrases = []
    target_phrases = []
    all_fields = []
    headers = []
    separator = "\t"  # Default separator

    with open(file_path, "r", encoding="utf-8") as f:
        # Process header information
        for line in f:
            if line.startswith("#"):
                if line.startswith("#separator:"):
                    separator = line.split(":")[1].strip()
                    if separator == "tab":
                        separator = "\t"
                # Process other header information if needed
            else:
                break  # Exit loop when we reach the data

        # Create a CSV reader with the determined separator
        reader = csv.reader(f, delimiter=separator)

        # Infer the number of columns from the first data row
        first_row = next(reader)
        num_columns = len(first_row)
        headers = [f"Field_{i}" for i in range(num_columns)]

        # Process the first row and continue with the rest
        process_row(
            first_row,
            english_field_index,
            target_field_index,
            english_phrases,
            target_phrases,
            all_fields,
            headers,
        )

        for row in reader:
            process_row(
                row,
                english_field_index,
                target_field_index,
                english_phrases,
                target_phrases,
                all_fields,
                headers,
            )

    return english_phrases, target_phrases, all_fields


def process_row(
    row: List[str],
    english_field_index: int,
    target_field_index: int,
    english_phrases: List[str],
    target_phrases: List[str],
    all_fields: List[Dict],
    headers: List[str],
):
    english_phrases.append(row[english_field_index])
    target_phrases.append(row[target_field_index])
    all_fields.append(dict(zip(headers, row)))


def export_to_anki(story_data_dict: Dict[str, Dict], output_dir: str, story_name: str):
    os.makedirs(output_dir, exist_ok=True)

    # Common CSS for all card types
    common_css = """
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
        1607392310,
        "Language Practice",
        fields=[
            {"name": "TargetText"},  # top field is the sort field
            {"name": "TargetAudio"},
            {"name": "EnglishText"},
            {"name": "WiktionaryLinks"},
        ],
        templates=[
            {
                "name": "Listening Card",
                "qfmt": "{{TargetAudio}}",
                "afmt": f"""
            {{{{FrontSide}}}}
            <hr id="answer">
            <div class="target-text" onclick="copyText(this)">{{{{TargetText}}}}</div>
            <div class="english-text">{{{{EnglishText}}}}</div>
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
                {{{{TargetAudio}}}}
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
                {{{{TargetAudio}}}}
                <div class="wiktionary-links">
                {{{{WiktionaryLinks}}}}
                </div>
                {card_back}
                """,
            },
        ],
        css=common_css,
    )

    media_files = []
    notes = []
    deck_id = string_to_large_int(config.language_name)
    deck = genanki.Deck(deck_id, f"{config.language_name} - phrases")

    for _, data in story_data_dict.items():
        for (english, target), audio_segments in zip(
            data["translated_phrase_list"], data["translated_phrase_list_audio"]
        ):
            # Generate unique filename for audio
            target_audio_normal = f"{uuid.uuid4()}.mp3"

            # Export audio segment
            if isinstance(audio_segments, AudioSegment):
                target_normal_audio_segment = audio_segments
            elif isinstance(audio_segments, List) and len(audio_segments) > 2:
                target_normal_audio_segment = audio_segments[2]
            else:
                raise Exception(f"Unexpected audio format: {audio_segments}")

            target_normal_audio_segment.export(
                os.path.join(output_dir, target_audio_normal), format="mp3"
            )

            # Add to media files list
            media_files.append(target_audio_normal)

            # Generate Wiktionary links
            wiktionary_links = generate_wiktionary_links(target, config.language_name)

            # Create notes for each card type
            note = genanki.Note(
                model=language_practice_model,
                fields=[
                    target,
                    f"[sound:{target_audio_normal}]",
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
            print(f"Deleted temporary file: {file_path}")
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")

    print("Cleanup of temporary MP3 files completed.")
