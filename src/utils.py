import inspect
import itertools
import json
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from anthropic import AnthropicVertex
from dotenv import load_dotenv
from tqdm import tqdm
from src.config_loader import config
from src.gcs_storage import get_story_collection_path, read_from_gcs

load_dotenv()  # so we can use environment variables for various global settings


def get_first_n_items(d: dict, n: int) -> dict:
    """
    Get the first n items from a dictionary.

    Args:
        d: Dictionary to slice
        n: Number of items to take

    Returns:
        A new dictionary containing the first n items
    """
    return dict(itertools.islice(d.items(), n))


def create_test_story_dict(
    story_data_dict: Dict[str, Dict],
    story_parts: int = 2,
    from_index: int = 0,
    dialogue_entries: int = 2,
    num_phrases: Optional[int] = None,
    fast_audio_fraction: Optional[float] = None,
) -> Dict[str, Dict]:
    """
    Create a smaller version of the story_data_dict for testing purposes.

    Args:
        story_data_dict (Dict[str, Dict]): The original story data dictionary.
        story_parts (int): Number of story parts to include in the test dictionary.
        from_index (int): Starting index for entries to include.
        dialogue_entries (int): Number of dialogue entries to include in each story part.
        num_phrases (int, optional): Number of phrases to include from corrected_phrase_list
            and related lists. If None, includes all phrases.
        fast_audio_fraction (float, optional): If provided, clips the fast audio to this
            fraction of its original length (e.g., 0.1 for 10% of length).

    Returns:
        Dict[str, Dict]: A smaller version of the story data dictionary for testing.
    """
    test_dict = {}

    for i, (part_key, part_data) in enumerate(story_data_dict.items()):
        if i >= story_parts:
            break

        test_dict[part_key] = {}

        # Handle phrase-related lists with num_phrases
        phrase_related_fields = [
            "corrected_phrase_list",
            "translated_phrase_list",
            "translated_phrase_list_audio",
            "image_path",
        ]

        for field in phrase_related_fields:
            if field in part_data:
                original_list = part_data[field]
                if num_phrases is not None:
                    end_index = min(from_index + num_phrases, len(original_list))
                    test_dict[part_key][field] = original_list[from_index:end_index]
                else:
                    test_dict[part_key][field] = original_list

        # Handle dialogue-related fields
        dialogue_fields = [
            "dialogue",
            "translated_dialogue",
            "translated_dialogue_audio",
        ]

        for field in dialogue_fields:
            if field in part_data:
                original_list = part_data[field]
                end_index = min(from_index + dialogue_entries, len(original_list))
                test_dict[part_key][field] = original_list[from_index:end_index]

        # Handle fast dialogue audio with optional clipping
        if "translated_dialogue_audio_fast" in part_data:
            fast_audio = part_data["translated_dialogue_audio_fast"]
            if fast_audio_fraction is not None and 0 < fast_audio_fraction <= 1:
                # Calculate length in milliseconds
                total_length = len(fast_audio)
                clip_length = int(total_length * fast_audio_fraction)
                fast_audio = fast_audio[:clip_length]
            test_dict[part_key]["translated_dialogue_audio_fast"] = fast_audio

        # Copy any other fields that might be present
        other_fields = (
            set(part_data.keys())
            - set(phrase_related_fields)
            - set(dialogue_fields)
            - {"translated_dialogue_audio_fast"}
        )
        for field in other_fields:
            test_dict[part_key][field] = part_data[field]

    return test_dict


def filter_longman_words(
    data: List[Dict], category: Literal["S1", "S2", "S3", "W1", "W2", "W3"]
) -> Dict[str, List[str]]:
    """This will only work with the specific format of longman data in a nested JSON structure from: https://github.com/healthypackrat/longman-communication-3000.
    S1 means part of the first 1000 vocab list for speech, W3 means part of the 3000 words (i.e. the third '1000' chunk) for writing
    """
    s1_words = defaultdict(list)
    for entry in data:
        if category in entry.get("frequencies", []):
            for word_class in entry.get("word_classes", []):
                s1_words[word_class].append(entry["word"])
    return dict(s1_words)


def get_longman_verb_vocab_dict(
    longman_file_path, category: Literal["S1", "S2", "S3", "W1", "W2", "W3"]
) -> Dict[str, List[str]]:
    """Returns a vocabulary dict with keys 'verbs' and 'vocab' for verbs and all other parts-of-speech. This is now in the
    same format as the known_vocab_list.json as used in the rest of the code."""
    data = load_json(longman_file_path)
    category_words = filter_longman_words(data, category=category)
    words_dict = defaultdict(list)
    for pos in category_words.keys():
        if pos in ["v", "auxillary"]:
            words_dict["verbs"].extend([word.lower() for word in category_words[pos]])
        else:
            words_dict["vocab"].extend([word.lower() for word in category_words[pos]])

    return words_dict


def save_pickle(data: Any, file_path: str) -> None:
    """
    Save data to a pickle file, with special handling for AudioSegment objects.

    Args:
        data: Any Python object that can be pickled, including those containing AudioSegment objects
        file_path: Path where the pickle file will be saved
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save with highest protocol for better compatibility
        with open(file_path, "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        print(f"Error saving pickle file {file_path}: {str(e)}")
        raise


def load_pickle(file_path: str, default_value: Any = None) -> Any:
    """
    Load data from a pickle file, with proper error handling.

    Args:
        file_path: Path to the pickle file
        default_value: Value to return if file doesn't exist or loading fails (default: None)

    Returns:
        The unpickled data, or default_value if loading fails
    """
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return default_value

    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        print(f"Error loading pickle file {file_path}: {str(e)}")
        return default_value


def load_text_file(file_path) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def save_text_file(lines: List[str], file_path: str) -> None:
    """Save a list of strings to a text file, one per line.

    Args:
        lines: List of strings to save
        file_path: Path where the file will be saved
    """
    with open(file_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def load_json(file_path) -> dict:
    """Returns {} if JSON does not exist"""
    with open(file_path, "r") as file:
        return json.load(file)


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)


def load_template(filename, parent_path: str = "../src/templates"):
    """Load a template file from the templates directory.

    Args:
        filename: Name of the template file
        parent_path: Parent directory containing the templates folder

    Returns:
        str: Contents of the template file
    """
    # Handle CSS files
    if filename.endswith(".css"):
        parent_path = os.path.join(parent_path, "styles")

    filename = os.path.join(parent_path, filename)
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def get_caller_name():
    """Method 1: Using inspect.stack()"""
    # Get the frame 2 levels up (1 would be this function, 2 is the caller)
    caller_frame = inspect.stack()[2]
    return caller_frame.function


def ok_to_query_api() -> bool:
    """Check if enough time has passed since the last API call.
    If not enough time has passed, wait for the remaining time.

    Returns:
        bool: True when it's ok to proceed with the API call
    """
    time_since_last_call = config.get_time_since_last_api_call()

    if time_since_last_call >= config.API_DELAY_SECONDS:
        config.update_api_timestamp()
        return True

    # Calculate how long we need to wait
    wait_time = int(config.API_DELAY_SECONDS - time_since_last_call)

    if time_since_last_call == 0:
        # first generation
        wait_time = 1

    # Show progress bar for waiting time
    pbar = tqdm(
        range(wait_time), desc="Waiting for API cooldown", ncols=75, colour="blue"
    )

    for sec in pbar:
        time.sleep(1)
        pbar.refresh()

    config.update_api_timestamp()
    return True


def anthropic_generate(prompt: str, max_tokens: int = 1024, model: str = None) -> str:
    """given a prompt generates an LLM response. The default model is specified in the config file.
    Most likely the largest Anthropic model. The region paramater in the config will have to match where that model
    is available"""
    print(
        f"Function that called this one: {get_caller_name()}. Sleeping for 20 seconds"
    )
    ok_to_query_api()

    client = AnthropicVertex(
        region=config.ANTHROPIC_REGION, project_id=config.PROJECT_ID
    )

    if model is None:
        model = config.ANTHROPIC_MODEL_NAME
    message = client.messages.create(
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    response_json = message.model_dump_json(indent=2)

    response = json.loads(response_json)
    return response["content"][0]["text"]


def extract_json_from_llm_response(response):
    """
    Extract JSON from an LLM response.

    :param response: String containing the LLM's response
    :return: Extracted JSON as a Python object, or None if no valid JSON is found
    """
    # Try to find JSON-like structure in the response
    json_pattern = (
        r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}))*\}"
    )
    json_match = re.search(json_pattern, response)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Found JSON-like structure, but it's not valid JSON")
            return None
    else:
        print("No JSON-like structure found in the response")
        return None


def get_story_position(
    story_name: str, collection: str = "LM1000", bucket_name: Optional[str] = None
) -> int:
    """
    Get the position of a story in the collection.

    Since Python 3.7+, dictionaries maintain insertion order, so we can safely
    get the position of a story by iterating through the keys.

    Args:
        story_name: Name of the story to find position for
        collection: Collection name (default: "LM1000")
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)

    Returns:
        int: 1-based position of the story in the collection (1 for first story)

    Raises:
        ValueError: If story_name is not found in collection data
    """

    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Get collection data from GCS
    collection_path = get_story_collection_path(collection)
    collection_data = read_from_gcs(bucket_name, collection_path, "json")

    # Find position in collection
    for i, key in enumerate(collection_data.keys(), start=1):
        if key == story_name:
            return i
    raise ValueError(f"Story '{story_name}' not found in collection {collection}")


def change_phrase(
    old_phrase: str,
    new_phrase: str,
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
    keep_image: bool = False,
    generate_audio: bool = True,
    review_translations: bool = True,
) -> None:
    """
    Change a phrase and update all related files in GCS.
    This includes updating the raw phrases list, translations, story collections,
    and handling multimedia files (audio and images).

    Args:
        old_phrase: The original phrase to be changed
        new_phrase: The new phrase to replace it with
        collection: Collection name (default: "LM1000")
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)
        keep_image: If True, renames the image file to use the new phrase key instead of preserving the old one
        generate_audio: If True, generates new audio for the new phrase
        review_translations: If True, uses Anthropic to review the translation
    """
    from src.convert import clean_filename
    from src.gcs_storage import (
        get_phrase_path,
        get_translated_phrases_path,
        get_story_collection_path,
        get_phrase_audio_path,
        get_phrase_image_path,
        get_phrase_index_path,
        get_story_index_path,
        read_from_gcs,
        upload_to_gcs,
    )
    from src.audio_generation import upload_phrases_audio_to_gcs
    from src.translation import (
        review_translations_with_anthropic,
        translate_from_english,
    )
    from src.wiktionary import generate_wiktionary_links
    from google.cloud import storage

    print(f"\n{'='*80}")
    print(f"Starting phrase change from '{old_phrase}' to '{new_phrase}'")
    print(f"{'='*80}\n")

    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Get the phrase keys for both old and new phrases
    old_phrase_key = clean_filename(old_phrase)
    new_phrase_key = clean_filename(new_phrase)
    print(f"Generated phrase keys:")
    print(f"  Old: {old_phrase_key}")
    print(f"  New: {new_phrase_key}\n")

    # 1. Update raw phrases list
    print("1. Updating raw phrases list...")
    phrases_path = get_phrase_path(collection)
    try:
        phrases = read_from_gcs(bucket_name, phrases_path, "json")
        if old_phrase not in phrases:
            raise ValueError(f"Old phrase '{old_phrase}' not found in phrases list")

        # Replace the old phrase with the new one
        phrases = [new_phrase if p == old_phrase else p for p in phrases]
        upload_to_gcs(phrases, bucket_name, phrases_path)
        print("  ✓ Raw phrases list updated successfully\n")
    except Exception as e:
        raise ValueError(f"Failed to update phrases list: {str(e)}")

    # 2. Update translations
    print("2. Updating translations...")
    translations_path = get_translated_phrases_path(collection)
    try:
        translations = read_from_gcs(bucket_name, translations_path, "json")
        if old_phrase_key in translations:
            # Get the old translation data
            old_translation_data = translations.pop(old_phrase_key)

            # Get new translation
            new_translation = translate_from_english([new_phrase])[0]

            # Create new translation data with correct format
            new_translation_data = {
                "english": new_phrase,
                config.TARGET_LANGUAGE_NAME.lower(): new_translation,
            }

            # If review_translations is True, use Anthropic to review the translation
            if review_translations:
                print("  ℹ Reviewing translation with Anthropic...")
                reviewed = review_translations_with_anthropic(
                    [{"english": new_phrase, "translation": new_translation}]
                )
                if reviewed and reviewed[0].get("modified", False):
                    new_translation_data[config.TARGET_LANGUAGE_NAME.lower()] = (
                        reviewed[0]["translation"]
                    )
                    print("  ✓ Translation reviewed and improved")
                else:
                    print("  ℹ Translation was good as is")

            # Generate new Wiktionary links
            print("  ℹ Generating Wiktionary links...")
            wiktionary_links = generate_wiktionary_links(
                new_translation_data[config.TARGET_LANGUAGE_NAME.lower()]
            )
            new_translation_data["wiktionary_links"] = wiktionary_links
            print("  ✓ Wiktionary links generated")

            # Add the new translation data
            translations[new_phrase_key] = new_translation_data
            upload_to_gcs(translations, bucket_name, translations_path)
            print("  ✓ Translations updated successfully\n")
        else:
            print("  ℹ No translation found for old phrase\n")
    except Exception as e:
        print(f"  ⚠ Warning: Failed to update translations: {str(e)}\n")

    # 3. Update story collections
    print("3. Updating story collections...")
    collection_path = get_story_collection_path(collection)
    try:
        collection_data = read_from_gcs(bucket_name, collection_path, "json")
        updated_stories = []
        for story_name, phrases in collection_data.items():
            for phrase_data in phrases:
                if phrase_data["phrase"] == old_phrase:
                    phrase_data["phrase"] = new_phrase
                    updated_stories.append(story_name)
        if updated_stories:
            upload_to_gcs(collection_data, bucket_name, collection_path)
            print(f"  ✓ Updated phrase in {len(updated_stories)} stories:")
            for story in updated_stories:
                print(f"    - {story}")
            print()
        else:
            print("  ℹ Phrase not found in any stories\n")
    except Exception as e:
        print(f"  ⚠ Warning: Failed to update story collections: {str(e)}\n")

    # 4. Handle multimedia files
    print("4. Handling multimedia files...")
    # First, get all files associated with the old phrase key
    old_audio_normal = get_phrase_audio_path(old_phrase_key, "normal")
    old_audio_slow = get_phrase_audio_path(old_phrase_key, "slow")
    old_image = get_phrase_image_path(old_phrase_key)
    new_image = get_phrase_image_path(new_phrase_key)

    # Delete old audio files
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    deleted_files = []
    for path in [old_audio_normal, old_audio_slow]:
        blob = bucket.blob(path)
        if blob.exists():
            blob.delete()
            deleted_files.append(path)

    if deleted_files:
        print("  ✓ Deleted audio files:")
        for file in deleted_files:
            print(f"    - {file}")
    else:
        print("  ℹ No audio files found to delete")

    # Handle image file
    old_image_blob = bucket.blob(old_image)
    if old_image_blob.exists():
        if keep_image:
            # Copy the image to the new location and delete the old one
            new_image_blob = bucket.blob(new_image)
            new_image_blob.rewrite(old_image_blob)
            old_image_blob.delete()
            print(f"  ✓ Renamed image file from {old_image} to {new_image}")
        else:
            print(f"  ℹ Image file preserved at: {old_image}")
    else:
        print("  ℹ No image file found")
    print()

    # 5. Generate new audio if requested
    if generate_audio:
        print("5. Generating new audio...")
        try:
            # Create phrase dict for audio generation
            phrase_dict = {
                new_phrase_key: {
                    "english": new_phrase,
                    config.TARGET_LANGUAGE_NAME.lower(): translations[new_phrase_key][
                        config.TARGET_LANGUAGE_NAME.lower()
                    ],
                }
            }

            # Generate and upload new audio
            result = upload_phrases_audio_to_gcs(
                phrase_dict=phrase_dict,
                bucket_name=bucket_name,
                upload_english_audio=True,
                overwrite=True,
            )

            if result and new_phrase_key in result:
                print("  ✓ New audio generated and uploaded successfully")
                print(
                    f"    - Normal speed: {result[new_phrase_key]['audio_urls']['normal']}"
                )
                print(
                    f"    - Slow speed: {result[new_phrase_key]['audio_urls']['slow']}"
                )
                print(
                    f"    - English: {result[new_phrase_key]['audio_urls']['english']}"
                )
            else:
                print("  ⚠ Warning: Failed to generate new audio")
            print()
        except Exception as e:
            print(f"  ⚠ Warning: Failed to generate new audio: {str(e)}\n")

    # 6. Rebuild indexes
    print("6. Rebuilding indexes...")
    try:
        # Rebuild phrase index
        phrase_index_path = get_phrase_index_path(collection)
        phrase_index = read_from_gcs(bucket_name, phrase_index_path, "json")
        updated_words = []
        for word, phrases in phrase_index.items():
            if old_phrase in phrases:
                phrases.remove(old_phrase)
                phrases.append(new_phrase)
                updated_words.append(word)
        if updated_words:
            upload_to_gcs(phrase_index, bucket_name, phrase_index_path)
            print(f"  ✓ Updated phrase index for {len(updated_words)} words:")
            for word in updated_words:
                print(f"    - {word}")
        else:
            print("  ℹ No updates needed in phrase index")

        # Rebuild story index
        story_index_path = get_story_index_path(collection)
        story_index = read_from_gcs(bucket_name, story_index_path, "json")
        updated_words = []
        for word, stories in story_index.items():
            if old_phrase in stories:
                stories.remove(old_phrase)
                stories.append(new_phrase)
                updated_words.append(word)
        if updated_words:
            upload_to_gcs(story_index, bucket_name, story_index_path)
            print(f"  ✓ Updated story index for {len(updated_words)} words:")
            for word in updated_words:
                print(f"    - {word}")
        else:
            print("  ℹ No updates needed in story index")
        print()
    except Exception as e:
        print(f"  ⚠ Warning: Failed to rebuild indexes: {str(e)}\n")

    print(f"{'='*80}")
    print(f"Successfully changed phrase from '{old_phrase}' to '{new_phrase}'")
    if not keep_image:
        print(
            f"Note: Image file at {old_image} was preserved. You may want to update it manually if needed."
        )
    print(f"{'='*80}\n")



