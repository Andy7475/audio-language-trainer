import os
import re
import tempfile
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import genanki
import pandas as pd
from anki.collection import Collection
from dotenv import load_dotenv
from PIL import Image
from pydub import AudioSegment
from tqdm import tqdm

from src.config_loader import config
from src.generate import add_audio, add_translations
from src.images import add_image_paths
from src.utils import (
    create_test_story_dict,
    string_to_large_int,
    load_template,
)
from src.wiktionary import generate_wiktionary_links


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


def get_deck_contents(
    deck_name: str,
    collection_path: Optional[str] = None,
    fields_to_extract: Optional[list[str]] = None,
    include_stats: bool = True,
) -> pd.DataFrame:
    """
    Get contents of a specific Anki deck as a pandas DataFrame.

    Args:
        deck_name: Name of the deck to analyze
        collection_path: Optional path to collection file. Uses ANKI_COLLECTION_PATH from env if None
        fields_to_extract: Optional list of field names to extract. If None, extracts all fields
        include_stats: Whether to include card statistics (ease, intervals, etc.)

    Returns:
        pd.DataFrame with columns for each note field plus statistics
    """
    try:
        with AnkiCollectionReader(collection_path) as reader:
            deck_id = reader.col.decks.id_for_name(deck_name)
            notes = reader.get_notes_for_deck(deck_name)

            if not notes:
                return pd.DataFrame()

            note_data = []

            for note in notes:
                # Basic note info
                note_dict = {
                    "note_id": note["id"],
                    "model_name": note["model_name"],
                    "tags": " ".join(note["tags"]),
                }

                # Add fields
                fields = note["fields"]
                if fields_to_extract:
                    for field in fields_to_extract:
                        note_dict[field] = fields.get(field, "")
                else:
                    note_dict.update(fields)

                if include_stats:
                    # Get all cards for this note - no need for fetchall()
                    cards = reader.col.db.execute(
                        "SELECT id, type, queue, due, ivl, factor, reps, lapses "
                        "FROM cards WHERE nid = ? AND did = ?",
                        note["id"],
                        deck_id,
                    )

                    if cards:  # cards is now directly the list of results
                        # Calculate average stats across all cards for this note
                        n_cards = len(cards)
                        total_ease = (
                            sum(card[5] for card in cards) / 10.0
                        )  # factor is stored as integer
                        total_reps = sum(card[6] for card in cards)
                        total_lapses = sum(card[7] for card in cards)
                        avg_interval = sum(card[4] for card in cards)  # ivl

                        note_dict.update(
                            {
                                "n_cards": n_cards,
                                "avg_ease": round(total_ease / n_cards, 1),
                                "total_reps": total_reps,
                                "avg_reps": round(total_reps / n_cards, 1),
                                "total_lapses": total_lapses,
                                "avg_lapses": round(total_lapses / n_cards, 1),
                                "avg_interval": round(avg_interval / n_cards, 1),
                            }
                        )
                    else:
                        # No cards found for this note
                        note_dict.update(
                            {
                                "n_cards": 0,
                                "avg_ease": None,
                                "total_reps": 0,
                                "avg_reps": 0,
                                "total_lapses": 0,
                                "avg_lapses": 0,
                                "avg_interval": 0,
                            }
                        )

                note_data.append(note_dict)

            # Create DataFrame
            df = pd.DataFrame(note_data)

            # Reorder columns
            first_cols = ["note_id", "model_name", "tags"]
            if include_stats:
                stat_cols = [
                    "n_cards",
                    "avg_ease",
                    "total_reps",
                    "avg_reps",
                    "total_lapses",
                    "avg_lapses",
                    "avg_interval",
                ]
                first_cols.extend(stat_cols)
            other_cols = [col for col in df.columns if col not in first_cols]
            df = df[first_cols + other_cols]

            if include_stats:
                df = add_knowledge_score(df)

            reader.close()
            return df

    except Exception as e:
        raise Exception(f"Error getting deck contents: {str(e)}")


def calculate_knowledge_score(row: pd.Series) -> float:
    """
    Calculate a knowledge score (0-1) for a single card based on its statistics.

    Components:
    - Interval (40%): max 365 days
    - Ease (20%): range 1300-3100
    - Success (30%): based on lapses vs reps
    - Efficiency (10%): interval gained per rep

    Args:
        row: Series containing card statistics with columns:
            avg_interval, avg_ease, total_reps, total_lapses

    Returns:
        float: Knowledge score between 0 and 1
    """
    # Return 0 if card has never been reviewed
    if row.total_reps == 0:
        return 0.0

    # 1. Interval score (40%)
    max_interval = 365
    interval_score = min(row.avg_interval / max_interval, 1) * 0.4

    # 2. Ease score (20%)
    min_ease = 1300
    max_ease = 3100
    ease_score = min(max(0, (row.avg_ease - min_ease) / (max_ease - min_ease)), 1) * 0.2

    # 3. Success score (30%)
    success_rate = 1 - (row.total_lapses / row.total_reps)
    success_score = max(0, success_rate) * 0.3

    # 4. Efficiency score (10%)
    days_per_rep = row.avg_interval / row.total_reps
    efficiency_score = min(days_per_rep / 30, 1) * 0.1  # Cap at 30 days per rep

    return round(interval_score + ease_score + success_score + efficiency_score, 3)


def add_knowledge_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add knowledge score column to DataFrame using apply."""
    result_df = df.copy()
    result_df["knowledge_score"] = df.apply(calculate_knowledge_score, axis=1)
    return result_df


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
            from_index=from_index,
            dialogue_entries=2,
            num_phrases=batch_size,
        )
        translated_phrases_dict_audio = add_audio(partial_dict)

        if image_dir:
            translated_phrases_dict_audio = add_image_paths(
                translated_phrases_dict_audio, image_dir
            )
            export_to_anki_with_images(
                story_data_dict=translated_phrases_dict_audio,
                output_dir=output_dir,
                story_name=f"{anki_filename_prefix}_{from_index}",
                deck_name=deck_name,
                index_position=from_index,
            )
        else:
            raise ValueError("Missing an image directory (image_dir)")
    return translated_phrases_dict_audio


def get_sort_field(order: int, target_text: str) -> str:
    """Create a unique sort field using order and truncated guid.

    Args:
        order: Integer for primary sort order (0-9999)
        target_text: The target language text used to generate guid

    Returns:
        String like '0001-3f4a9' that will sort correctly in Anki
    """
    guid = string_to_large_int(target_text + "image")
    # Take first 5 chars of hex representation (20 bits)
    truncated_guid = hex(guid)[2:10]
    return f"{order:04d}-{truncated_guid}"


def export_to_anki_with_images(
    story_data_dict: Dict[str, Dict],
    output_dir: str,
    story_name: str,
    deck_name: str = None,
    index_position: int = 0,
) -> None:
    """
    Export story data to an Anki deck, including images for each card. Use add_image_paths
    with story_data_dict first to get image data.

    The story_name is used as a prefix for the anki file only
    if you want these merged with other decks, ensure the deck_name matches exactly
    as this is used to generate the deck id.

    index_position is used to correctly allocate the right card_position index, so that
    the cards are served in the order they were generated, which will be an optimised order
    based on evening out the number of words per phrase, otherwise Anki will sort alphabetically on the
    first field in the list
    """
    os.makedirs(output_dir, exist_ok=True)

    language_practice_model = genanki.Model(
        1607392313 + 121 + 999,  # adding 999 to try sort order
        "Language Practice With Images Sort Order",
        fields=[
            {"name": "SortOrder"},
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
                        img = img.resize((400, 400))
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
                    get_sort_field(index_position, target),
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
            index_position += 1

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
