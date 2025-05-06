import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import genanki
import pandas as pd
from anki.collection import Collection
from dotenv import load_dotenv
from PIL import Image
from pydub import AudioSegment
from tqdm import tqdm

from src.config_loader import config
from src.convert import clean_filename, get_deck_name, string_to_large_int
from src.gcs_storage import get_story_collection_path, read_from_gcs
from src.generate import add_audio, add_translations
from src.images import add_image_paths
from src.phrase import build_phrase_dict_from_gcs, get_phrase_keys
from src.utils import (
    create_test_story_dict,
    load_template,
)
from src.wiktionary import generate_wiktionary_links


def import_anki_packages(
    package_paths: List[str],
    collection_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Simple wrapper to import multiple Anki packages using AnkiCollectionReader.
    Uses a context manager for proper cleanup.

    Args:
        package_paths: List of paths to .apkg files to import
        collection_path: Path to Anki collection (defaults to ANKI_COLLECTION_PATH env)
        verbose: Whether to print status updates during import

    Returns:
        Dictionary containing import results
    """

    # Validate inputs
    if not package_paths:
        return {
            "total_imported": 0,
            "successful": [],
            "failed": {"error": "No package paths provided"},
        }

    # Filter for valid .apkg files
    valid_paths = []
    invalid_paths = {}

    for path in package_paths:
        if not os.path.exists(path):
            invalid_paths[path] = "File not found"
        elif not path.lower().endswith(".apkg"):
            invalid_paths[path] = "Not an .apkg file"
        else:
            valid_paths.append(path)

    if not valid_paths:
        return {"total_imported": 0, "successful": [], "failed": invalid_paths}

    # Use the AnkiCollectionReader with context manager
    try:
        with AnkiCollectionReader(collection_path) as reader:
            if verbose:
                print(f"Connected to Anki collection: {reader.collection_path}")
                print(f"Importing {len(valid_paths)} package(s)...")

            total_imported = 0
            successful = []
            failed = {}

            for path in valid_paths:
                try:
                    # Import the package using Anki's backend method
                    from anki.collection import ImportAnkiPackageRequest

                    request = ImportAnkiPackageRequest(package_path=path)
                    imported = reader.col.import_anki_package(request)
                    notes_imported = imported.log.found_notes
                    total_imported += notes_imported
                    successful.append({"path": path, "notes_imported": notes_imported})
                except Exception as e:
                    failed[path] = str(e)

            # Save changes after all imports
            if successful:
                reader.col.save()

            # Add the invalid paths to the failed results
            failed.update(invalid_paths)

            # Print summary if verbose
            if verbose:
                print("Import summary:")
                print(f"Total notes imported: {total_imported}")
                print(f"Successfully imported packages: {len(successful)}")

                if successful:
                    print("\nSuccessful imports:")
                    for success in successful:
                        print(f"- {success['path']}: {success['notes_imported']} notes")

                if failed:
                    print("\nFailed imports:")
                    for path, error in failed.items():
                        print(f"- {path}: {error}")

            return {
                "total_imported": total_imported,
                "successful": successful,
                "failed": failed,
            }

    except Exception as e:
        # Handle any exceptions during import
        error_msg = str(e)
        if verbose:
            print(f"Error during import: {error_msg}")

        return {
            "total_imported": 0,
            "successful": [],
            "failed": {p: error_msg for p in valid_paths},
        }


def update_language_model_templates(model_id: int, model_name: str):
    try:
        # Import genanki here to keep it optional
        import genanki

        # Create genanki model (this mimics your existing code)
        language_practice_model = genanki.Model(
            model_id,  # Model ID
            model_name,
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

        # Connect to Anki collection and update templates
        with AnkiCollectionReader() as reader:
            print(f"Connected to Anki collection: {reader.collection_path}")

            # Update ONLY the templates and CSS (safer)
            success = reader.update_model_templates_only(language_practice_model)

            if success:
                print("Successfully updated templates")
                # Print details for verification
                reader.print_model_details("Language Practice With Images")
            else:
                print("Failed to update templates")

    except ImportError:
        print("genanki module not found. Install it with: pip install genanki")
    except Exception as e:
        print(f"Error: {str(e)}")


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
        if not self.collection_path or not os.path.exists(self.collection_path):
            raise FileNotFoundError(
                f"Anki collection not found at: {self.collection_path}"
            )
        self.col: Optional[Collection] = None

    def connect(self):
        """Connect to the Anki collection."""
        self.col = Collection(self.collection_path)
        return self

    def close(self):
        """Close the connection to the collection."""
        if self.col:
            self.col.close()
            self.col = None

    def get_deck_names(self) -> Dict[int, str]:
        """Returns a dictionary of deck_id : deck_name"""
        if not self.col:
            raise RuntimeError("Not connected to collection")

        decks = {}
        for deck in self.col.decks.all():
            decks[deck["id"]] = deck["name"]
        return decks

    def get_notes_for_deck(self, deck_name: str) -> List[Dict[str, Any]]:
        """Get all notes for a specific deck."""
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

    def import_apkg(self, apkg_path: str) -> Tuple[int, List[str]]:
        """
        Import an Anki package (.apkg) file into the collection.

        Args:
            apkg_path: Path to the .apkg file

        Returns:
            Tuple containing:
            - Number of notes imported
            - List of error messages, if any

        Raises:
            FileNotFoundError: If the .apkg file doesn't exist
            RuntimeError: If not connected to collection
        """
        if not self.col:
            raise RuntimeError("Not connected to collection")

        if not os.path.exists(apkg_path):
            raise FileNotFoundError(f"APKG file not found: {apkg_path}")

        # Ensure the file is an APKG
        if not apkg_path.lower().endswith(".apkg"):
            raise ValueError(f"File must be an .apkg file: {apkg_path}")

        try:
            # Import the package using Anki's built-in method
            imported = self.col.import_anki_package(apkg_path)

            # Save changes
            self.col.save()

            return imported.num_notes, []
        except Exception as e:
            return 0, [f"Error importing package: {str(e)}"]

    def import_anki_packages(self, package_paths: List[str]) -> Dict[str, Any]:
        """
        Import multiple Anki package (.apkg) files into the collection.

        Args:
            package_paths: List of paths to .apkg files

        Returns:
            Dictionary containing:
            - 'total_imported': Total number of notes imported
            - 'successful': List of successfully imported packages
            - 'failed': Dict mapping failed package paths to error messages

        Raises:
            RuntimeError: If not connected to collection
        """
        if not self.col:
            raise RuntimeError("Not connected to collection")

        total_imported = 0
        successful = []
        failed = {}

        for package_path in package_paths:
            try:
                # Check if file exists
                if not os.path.exists(package_path):
                    failed[package_path] = f"File not found: {package_path}"
                    continue

                # Check file extension
                if not package_path.lower().endswith(".apkg"):
                    failed[package_path] = f"File must be an .apkg file: {package_path}"
                    continue

                # Import the package
                imported = self.col.import_anki_package(package_path)

                # Update counters
                total_imported += imported.num_notes
                successful.append(
                    {"path": package_path, "notes_imported": imported.num_notes}
                )

            except Exception as e:
                failed[package_path] = f"Error: {str(e)}"

        # Save changes after all imports
        if successful:
            self.col.save()

        # Return summary
        return {
            "total_imported": total_imported,
            "successful": successful,
            "failed": failed,
        }

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific note type (model).

        Args:
            model_name: Name of the model

        Returns:
            Dictionary containing model data if found, None otherwise
        """
        if not self.col:
            raise RuntimeError("Not connected to collection")

        model = self.col.models.by_name(model_name)
        if not model:
            return None

        return dict(model)

    def update_model_from_genanki(
        self, genanki_model, update_existing: bool = True
    ) -> bool:
        """
        Update or create a model in Anki from a genanki.Model object using to_json().

        Args:
            genanki_model: A genanki.Model instance
            update_existing: If True, update existing model if found; if False, fail if model exists

        Returns:
            True if successful, False otherwise
        """
        if not self.col:
            raise RuntimeError("Not connected to collection")

        # Check if model with this name already exists
        existing_model = self.col.models.by_name(genanki_model.name)

        if existing_model and not update_existing:
            print(
                f"Model '{genanki_model.name}' already exists and update_existing=False"
            )
            return False

        # Get a default deck ID for the model.to_json() method
        default_deck = self.col.decks.all()[0]
        default_deck_id = default_deck["id"]

        # Get the current timestamp for the model.to_json() method
        timestamp = time.time()

        # Convert genanki model to Anki format using to_json()
        try:
            # Use the provided to_json() method
            model_json = genanki_model.to_json(
                timestamp=timestamp, deck_id=default_deck_id
            )

            if existing_model:
                # Update existing model
                # We need to preserve the existing model ID
                model_json["id"] = existing_model["id"]

                # Update fields - careful with this as changing field structure can break notes
                # For safety, we'll check if field structures match
                if len(model_json["flds"]) != len(existing_model["flds"]):
                    print("Warning: Different number of fields between models.")
                    # Need to handle field mapping here if changing structure

                # Update templates
                existing_model["tmpls"] = model_json["tmpls"]

                # Update CSS
                existing_model["css"] = model_json["css"]

                # Save the updated model
                self.col.models.save(existing_model)
                self.col.models.update(existing_model)

                print(f"Updated existing model: {genanki_model.name}")
                return True
            else:
                # Create new model
                # We need to convert the model JSON to an Anki model object
                new_model = self.col.models.new(genanki_model.name)

                # Add fields
                for field in model_json["flds"]:
                    field_name = field["name"]
                    self.col.models.add_field(
                        new_model, self.col.models.new_field(field_name)
                    )

                # Add templates
                for template in model_json["tmpls"]:
                    t = self.col.models.new_template(template["name"])
                    t["qfmt"] = template["qfmt"]
                    t["afmt"] = template["afmt"]
                    self.col.models.add_template(new_model, t)

                # Set CSS
                new_model["css"] = model_json["css"]

                # Set sort field
                new_model["sortf"] = model_json["sortf"]

                # Save the new model
                self.col.models.save(new_model)
                self.col.models.update(new_model)

                print(f"Created new model: {genanki_model.name}")
                return True

        except Exception as e:
            print(f"Error updating model: {str(e)}")
            return False

    def update_model_templates_only(self, genanki_model) -> bool:
        """
        Update only the templates and CSS of an existing model, preserving field structure.
        This is safer than full model updates when you just want to change the templates.

        Args:
            genanki_model: A genanki.Model instance

        Returns:
            True if successful, False otherwise
        """
        if not self.col:
            raise RuntimeError("Not connected to collection")

        # Check if model exists
        existing_model = self.col.models.by_name(genanki_model.name)
        if not existing_model:
            print(
                f"Model '{genanki_model.name}' does not exist - cannot update templates"
            )
            return False

        try:
            # Get default deck ID (needed for to_json())
            default_deck_id = self.col.decks.all()[0]["id"]

            # Get model JSON
            model_json = genanki_model.to_json(
                timestamp=time.time(), deck_id=default_deck_id
            )

            # Update templates
            # Check if template count matches
            if len(model_json["tmpls"]) != len(existing_model["tmpls"]):
                print(
                    f"Warning: Template count mismatch. Existing: {len(existing_model['tmpls'])}, New: {len(model_json['tmpls'])}"
                )

                # If more templates in new model, add them
                if len(model_json["tmpls"]) > len(existing_model["tmpls"]):
                    for i in range(
                        len(existing_model["tmpls"]), len(model_json["tmpls"])
                    ):
                        template = model_json["tmpls"][i]
                        t = self.col.models.new_template(template["name"])
                        t["qfmt"] = template["qfmt"]
                        t["afmt"] = template["afmt"]
                        self.col.models.add_template(existing_model, t)

                # If fewer templates in new model, we leave the extras alone

            # Update content of existing templates
            for i in range(min(len(existing_model["tmpls"]), len(model_json["tmpls"]))):
                existing_model["tmpls"][i]["name"] = model_json["tmpls"][i]["name"]
                existing_model["tmpls"][i]["qfmt"] = model_json["tmpls"][i]["qfmt"]
                existing_model["tmpls"][i]["afmt"] = model_json["tmpls"][i]["afmt"]

            # Update CSS
            existing_model["css"] = model_json["css"]

            # Save changes
            self.col.models.save(existing_model)
            self.col.models.update(existing_model)

            print(f"Successfully updated templates for model: {genanki_model.name}")
            return True

        except Exception as e:
            print(f"Error updating templates: {str(e)}")
            return False

    def print_model_details(self, model_name: str) -> None:
        """Print details about a model for debugging purposes."""
        if not self.col:
            raise RuntimeError("Not connected to collection")

        model = self.col.models.by_name(model_name)
        if not model:
            print(f"Model '{model_name}' not found")
            return

        print(f"Model: {model_name}")
        print(f"ID: {model['id']}")
        print(f"Fields: {[f['name'] for f in model['flds']]}")
        print(f"Templates: {[t['name'] for t in model['tmpls']]}")
        print(f"CSS Length: {len(model['css'])} characters")

        for i, template in enumerate(model["tmpls"]):
            print(f"\nTemplate {i+1}: {template['name']}")
            print(f"Question format (first 100 chars): {template['qfmt'][:100]}...")
            print(f"Answer format (first 100 chars): {template['afmt'][:100]}...")

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


def create_anki_deck_from_gcs(
    story_name: Optional[str | list[str]] = None,
    deck_name: Optional[str] = None,
    output_dir: Optional[str] = "../outputs/flashcards",
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
) -> None:
    """
    Create an Anki deck from translated phrase data stored in GCS.

    Args:
        deck_name: Name of the Anki deck
        output_dir: Directory to save the Anki deck
        story_name: Optional name of the story to filter phrases by. Can be a single story name or a list of story names.
        collection: Collection name (default: "LM1000")
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)
        image_dir: Optional directory containing images for the deck
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Get the story collection to map phrases to their keys
    collection_path = get_story_collection_path(collection)
    try:
        collection_data = read_from_gcs(bucket_name, collection_path, "json")
    except Exception as e:
        raise ValueError(f"Failed to read story collection: {str(e)}")

    # Convert single story_name to list for consistent handling
    story_names = [story_name] if isinstance(story_name, str) else story_name

    # If no specific stories requested, get all stories from collection
    if not story_names:
        story_names = list(collection_data.keys())

    # Process each story
    for story_position, current_story_name in enumerate(story_names, start=1):
        # Get phrase keys for this story
        phrase_keys = get_phrase_keys(current_story_name, collection)
        if not phrase_keys:
            print(f"No phrases found for story: {current_story_name}")
            continue

        # Build the phrase dictionary from GCS with filtering
        phrase_dict = build_phrase_dict_from_gcs(
            collection=collection, bucket_name=bucket_name, phrase_keys=phrase_keys
        )

        if not phrase_dict:
            print(f"No matching phrases found for story: {current_story_name}")
            continue

        # Create Anki deck for this story
        export_phrases_to_anki(
            phrase_dict=phrase_dict,
            output_dir=output_dir,
            deck_name=deck_name,
            story_name=current_story_name,
            collection=collection,
            story_position=story_position,
        )


def export_phrases_to_anki(
    phrase_dict: Dict[str, Dict[str, Any]],
    output_dir: str = "../outputs/flashcards",
    language: str = config.TARGET_LANGUAGE_NAME.lower(),
    story_name: Optional[str] = None,
    deck_name: Optional[str] = None,
    collection: str = "LM1000",
    story_position: Optional[int] = None,
) -> None:
    """
    Export phrase data to an Anki deck.

    Args:
        phrase_dict: Dictionary of phrase data in format:
            {
                "phrase_key": {
                    "english_text": str,
                    "target_text": str,
                    "audio_normal": AudioSegment or None,
                    "audio_slow": AudioSegment or None,
                    "image": Image or None,
                    "wiktionary_links": str or None
                },
                ...
            }
        output_dir: Directory to save the Anki deck
        deck_name: Name of the Anki deck (if left out one will be generated based off language, collection and story_name)
        language: Target language name (e.g. 'french')
        story_name: Optional name of the story (e.g. 'story_community_park')
        collection: Collection name (default: "LM1000")
        story_position: Optional position of story in sequence (e.g. 1 becomes "01")
    """
    os.makedirs(output_dir, exist_ok=True)
    language_cap = language.title()
    # Format deck name with proper hierarchy
    if story_name:
        formatted_deck_name = get_deck_name(
            story_name, collection, story_position, language
        )
    else:
        formatted_deck_name = f"{language_cap}::{collection}::Phrases"

    # Create package filename in snake case
    package_name_parts = [language.lower()]
    if collection != "LM1000":
        package_name_parts.append(collection.lower())
    if story_name:
        package_name_parts.append(story_name.replace("story_", ""))
    package_filename = f"{'_'.join(package_name_parts)}_anki_deck.apkg"

    # Create Anki model
    language_practice_model = genanki.Model(
        1607392313,  # Model ID
        "Language Practice With Images",
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

    if deck_name is not None:
        formatted_deck_name = deck_name
    # Set up deck
    deck_id = string_to_large_int(formatted_deck_name + "TurboPhrase")
    deck = genanki.Deck(deck_id, formatted_deck_name)

    # Get the original phrase keys in order from the story collection
    if story_name:
        phrase_keys = get_phrase_keys(story_name, collection)
    else:
        phrase_keys = list(phrase_dict.keys())

    # Process each phrase in the original order
    for index, phrase_key in enumerate(phrase_keys):
        if phrase_key not in phrase_dict:
            continue

        phrase_data = phrase_dict[phrase_key]
        try:
            # Handle image
            image_filename = None
            if phrase_data["image"] is not None:
                try:
                    image_filename = f"{uuid.uuid4()}.png"
                    output_path = os.path.join(output_dir, image_filename)
                    # Resize and save the PIL Image directly
                    resized_img = phrase_data["image"].resize((400, 400))
                    resized_img.save(output_path, "PNG", optimize=True)
                    media_files.append(image_filename)
                except Exception as e:
                    print(f"Error processing image for {phrase_key}: {str(e)}")
                    image_filename = None

            # Handle audio
            target_audio_normal = f"{uuid.uuid4()}.mp3"
            target_audio_slow = f"{uuid.uuid4()}.mp3"

            if phrase_data["audio_normal"] is not None:
                phrase_data["audio_normal"].export(
                    os.path.join(output_dir, target_audio_normal), format="mp3"
                )
                media_files.append(target_audio_normal)

            if phrase_data["audio_slow"] is not None:
                phrase_data["audio_slow"].export(
                    os.path.join(output_dir, target_audio_slow), format="mp3"
                )
                media_files.append(target_audio_slow)

            # Create note
            note = genanki.Note(
                model=language_practice_model,
                fields=[
                    get_sort_field(index, phrase_data["target_text"]),
                    phrase_data["target_text"],
                    (
                        f"[sound:{target_audio_normal}]"
                        if phrase_data["audio_normal"]
                        else ""
                    ),
                    f"[sound:{target_audio_slow}]" if phrase_data["audio_slow"] else "",
                    phrase_data["english_text"],
                    phrase_data["wiktionary_links"] or "",
                    f'<img src="{image_filename}">' if image_filename else "",
                    language_cap,
                ],
                guid=string_to_large_int(phrase_data["target_text"] + "image"),
            )
            notes.append(note)

        except Exception as e:
            print(f"Error processing phrase {phrase_key}: {str(e)}")
            continue

    # Add notes to deck
    for note in tqdm(notes, desc="adding notes to deck"):
        deck.add_note(note)

    # Create and save the package
    package = genanki.Package(deck)
    package.media_files = [os.path.join(output_dir, file) for file in media_files]
    output_filename = os.path.join(output_dir, package_filename)
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
