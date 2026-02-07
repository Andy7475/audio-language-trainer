import inspect
import itertools
import json
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

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


def render_html_content(data: dict, template_name: str) -> str:
    """Renders HTML content via Jinja2

    Args:
        data (dict): dictionary that will be fed directly into the the Jinja Template
        template_name (str): name of template  in src/templates

    Returns:
        str: html_content
    """
    loader = FileSystemLoader(
        [
            "templates",
            "../src/templates",
            "src/templates",
            "../src/templates/shop",
            "src/templates/shop",
            "templates/shop",
        ]
    )
    env = Environment(loader=loader, autoescape=False)
    template = env.get_template(template_name)
    html_content = template.render(data)
    return html_content


def normalize_lang_code_for_wiktionary(lang_code: str) -> str:
    """
    Normalize language codes to match Wiktionary's language code conventions.

    Args:
        lang_code: Base language code from langcodes (e.g., 'cmn', 'en', 'yue')

    Returns:
        str: Normalized language code for Wiktionary (e.g., 'zh', 'en')
    """
    # Mapping of language codes to Wiktionary codes
    lang_code_mapping = {
        "cmn": "zh",  # Mandarin Chinese -> Chinese
        "yue": "zh",  # Cantonese -> Chinese (if applicable)
        "nan": "zh",  # Min Nan -> Chinese (if applicable)
        "wuu": "zh",  # Wu Chinese -> Chinese (if applicable)
        "sr": "sh",  # Serbian -> Serbo-Croatian
        "hr": "sh",  # Croatian -> Serbo-Croatian (if needed)
        "bs": "sh",  # Bosnian -> Serbo-Croatian (if needed)
    }

    return lang_code_mapping.get(lang_code, lang_code)


def normalize_lang_code_for_challenges(lang_code: str) -> str:
    """
    Normalize language codes to match Wiktionary's language code conventions.

    Args:
        lang_code: Base language code from langcodes (e.g., 'cmn', 'en', 'yue')

    Returns:
        str: Normalized language code for Wiktionary (e.g., 'zh', 'en')
    """
    # Mapping of language codes to Wiktionary codes
    lang_code_mapping = {
        "cmn": "zh",  # Mandarin Chinese -> Chinese
        "yue": "zh",  # Cantonese -> Chinese (if applicable)
        "nan": "zh",  # Min Nan -> Chinese (if applicable)
        "wuu": "zh",  # Wu Chinese -> Chinese (if applicable)
        # "sr": "sh",  # Serbian -> Serbo-Croatian
        # "hr": "sh",  # Croatian -> Serbo-Croatian (if needed)
        # "bs": "sh",  # Bosnian -> Serbo-Croatian (if needed)
    }

    return lang_code_mapping.get(lang_code, lang_code)
