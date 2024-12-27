import base64
import hashlib
import inspect
import io
import json
import os
import pickle
import re
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from anthropic import AnthropicVertex
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from pydub import AudioSegment
from tqdm import tqdm

from src.config_loader import config

load_dotenv()  # so we can use environment variables for various global settings


def sanitize_path_component(s: str) -> str:
    """
    Sanitize a string for use in GCS paths.
    Replaces spaces with underscores, removes special characters, and converts to lowercase.
    """
    # Replace one or more spaces with a single underscore
    s = re.sub(r"\s+", "_", s)
    # Remove any characters that aren't alphanumeric, underscore, or hyphen
    s = "".join(c for c in s if c.isalnum() or c in "_-")
    # Convert to lowercase for consistency
    return s.lower()


def construct_gcs_path(
    story_name: str,
    language_name: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> str:
    """
    Construct the Google Cloud Storage path for a story HTML file.

    Args:
        story_name: Name of the story (without .html extension)
        language_name: Optional language name (defaults to config.TARGET_LANGUAGE_NAME)
        bucket_name: Optional bucket name (defaults to config.GCS_PUBLIC_BUCKET)

    Returns:
        str: Full GCS path in format: gs://<bucket>/<language>/<story_name>/story_name.html
    """
    # Get defaults from config if not provided
    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET
    if language_name is None:
        language_name = config.TARGET_LANGUAGE_NAME

    # Sanitize paths
    language_folder = sanitize_path_component(language_name)
    story_name = sanitize_path_component(Path(story_name).stem)

    # Construct path
    return f"gs://{bucket_name}/{language_folder}/{story_name}/{story_name}.html"


def upload_to_gcs(
    html_file_path: str,
    language_name: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> str:
    """
    Upload an HTML file to Google Cloud Storage with organized folder structure.

    Args:
        html_file_path: Path to the HTML file to upload
        language_name: Language name (e.g. "Swedish", "French")
        bucket_name: Optional bucket name (defaults to config.GCS_PUBLIC_BUCKET)

    Returns:
        str: Public URL of the uploaded file

    Raises:
        FileNotFoundError: If the HTML file doesn't exist
        google.cloud.exceptions.NotFound: If the bucket doesn't exist
    """
    # Check if file exists
    if not os.path.exists(html_file_path):
        raise FileNotFoundError(f"HTML file not found: {html_file_path}")

    # Initialize storage client
    storage_client = storage.Client()

    # Get bucket
    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET
    bucket = storage_client.bucket(bucket_name)

    # Get story name and construct GCS path
    story_name = Path(html_file_path).stem
    gcs_path = construct_gcs_path(story_name, language_name, bucket_name)

    # Remove gs:// prefix and bucket name for blob path
    blob_path = "/".join(gcs_path.split("/")[3:])

    # Create and upload blob
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(html_file_path, content_type="text/html")

    # Construct the public URL (since bucket is public)
    return f"https://storage.googleapis.com/{bucket_name}/{blob_path}"


def clean_filename(phrase: str) -> str:
    """Convert a phrase to a clean filename-safe string."""
    # Convert to lowercase
    clean = phrase.lower()
    # Replace any non-alphanumeric characters (except spaces) with empty string
    clean = re.sub(r"[^a-z0-9\s]", "", clean)
    # Replace spaces with underscores
    clean = clean.replace(" ", "_")
    # Remove any double underscores
    clean = re.sub(r"_+", "_", clean)
    # Trim any leading/trailing underscores
    clean = clean.strip("_")
    return clean


def string_to_large_int(s: str) -> int:
    """Notes in Anki have a unique ID, and so to create the note ID and ensure
    it correlates with the content we can pass in the translated phrase as a string
    and get back a large interger (a bit like a hash function).

    So this can be used to create a numerical ID from a given phrase.

    Args:
        s (str): The string to convert (usually the translated phrase)

    Returns:
        int: A large interger (equivalent to a hash)
    """
    # Encode the string to bytes
    encoded = s.encode("utf-8")
    # Create a SHA-256 hash
    hash_object = hashlib.sha256(encoded)
    # Get the hexadecimal representation
    hex_dig = hash_object.hexdigest()
    # Take the first 16 characters (64 bits) of the hex string
    truncated_hex = hex_dig[:16]
    # Convert hex to integer
    large_int = int(truncated_hex, 16)
    # Ensure the value is positive and within SQLite's signed 64-bit integer range
    return large_int & 0x7FFFFFFFFFFFFFFF


def create_test_story_dict(
    story_data_dict: Dict[str, Dict],
    story_parts: int = 2,
    from_index: int = 0,
    dialogue_entries: int = 2,
    fast_audio_fraction: Optional[float] = None,
) -> Dict[str, Dict]:
    """
    Create a smaller version of the story_data_dict for testing purposes.

    Args:
        story_data_dict (Dict[str, Dict]): The original story data dictionary.
        story_parts (int): Number of story parts to include in the test dictionary.
        from_index (int): Starting index for dialogue entries to include.
        dialogue_entries (int): Number of dialogue entries to include in each story part.
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

        # Handle dialogue related fields
        if "dialogue" in part_data:
            end_index = min(from_index + dialogue_entries, len(part_data["dialogue"]))
            test_dict[part_key]["dialogue"] = part_data["dialogue"][
                from_index:end_index
            ]

        if "translated_dialogue" in part_data:
            end_index = min(
                from_index + dialogue_entries, len(part_data["translated_dialogue"])
            )
            test_dict[part_key]["translated_dialogue"] = part_data[
                "translated_dialogue"
            ][from_index:end_index]

        if "translated_dialogue_audio" in part_data:
            end_index = min(
                from_index + dialogue_entries,
                len(part_data["translated_dialogue_audio"]),
            )
            test_dict[part_key]["translated_dialogue_audio"] = part_data[
                "translated_dialogue_audio"
            ][from_index:end_index]

        # Handle fast dialogue audio with optional clipping
        if "translated_dialogue_audio_fast" in part_data:
            fast_audio = part_data["translated_dialogue_audio_fast"]
            if fast_audio_fraction is not None and 0 < fast_audio_fraction <= 1:
                # Calculate length in milliseconds
                total_length = len(fast_audio)
                clip_length = int(total_length * fast_audio_fraction)
                fast_audio = fast_audio[:clip_length]
            test_dict[part_key]["translated_dialogue_audio_fast"] = fast_audio

    return test_dict


def convert_defaultdict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict(v) for k, v in d.items()}
    return d


def save_defaultdict(d, filepath):
    normal_dict = convert_defaultdict(d)
    save_json(normal_dict, filepath)


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
    if not os.path.exists(file_path):
        print("file does not exist, returning empty dict")
        return {}
    with open(file_path, "r") as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)


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
