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

load_dotenv()  # so we can use environment variables for various global settings


def list_story_folders(base_dir: str = "../outputs/stories"):
    """List all story_ folders in the specified directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    return [
        folder.name
        for folder in base_path.iterdir()
        if folder.is_dir() and folder.name.startswith("story_")
    ]


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
    # print(os.listdir())
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


def get_story_position(story_name: str, collection_data: Dict[str, Any]) -> int:
    """
    Get the position of a story in the collection data.

    Since Python 3.7+, dictionaries maintain insertion order, so we can safely
    get the position of a story by iterating through the keys.

    Args:
        story_name: Name of the story to find position for
        collection_data: Dictionary from get_story_collection_path containing story data

    Returns:
        int: 1-based position of the story in the collection (1 for first story)

    Raises:
        ValueError: If story_name is not found in collection_data
    """
    for i, key in enumerate(collection_data.keys(), start=1):
        if key == story_name:
            return i
    raise ValueError(f"Story '{story_name}' not found in collection data")
