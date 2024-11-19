import base64
import copy
import hashlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import pycountry
import requests
import spacy
import vertexai
from anthropic import AnthropicVertex
from dotenv import load_dotenv
from PIL import Image
from pydub import AudioSegment
from tqdm import tqdm
from vertexai.preview.vision_models import ImageGenerationModel

from src.config_loader import config

load_dotenv()  # so we can use environment variables for various global settings


def convert_audio_to_base64(audio_segment: AudioSegment) -> str:
    """Convert an AudioSegment to a base64 encoded string."""
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3")
    buffer.seek(0)
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


def prepare_story_data_for_html(story_data_dict: Dict) -> Dict:
    """Process the story data dictionary to include base64 encoded audio."""
    prepared_data = {}

    for section_name, section_data in story_data_dict.items():
        prepared_data[section_name] = {
            "dialogue": section_data.get("dialogue", []),
            "translated_dialogue": section_data.get("translated_dialogue", []),
            "translated_phrase_list": section_data.get("translated_phrase_list", []),
            "audio_data": {"phrases": [], "dialogue": []},
        }

        # Process phrase audio
        if "translated_phrase_list_audio" in section_data:
            for audio_segments in section_data["translated_phrase_list_audio"]:
                if isinstance(audio_segments, list) and len(audio_segments) > 2:
                    normal_audio = convert_audio_to_base64(audio_segments[2])
                    slow_audio = convert_audio_to_base64(audio_segments[1])
                    prepared_data[section_name]["audio_data"]["phrases"].append(
                        {"normal": normal_audio, "slow": slow_audio}
                    )

        # Process dialogue audio
        if "translated_dialogue_audio" in section_data:
            for audio_segment in section_data["translated_dialogue_audio"]:
                audio_base64 = convert_audio_to_base64(audio_segment)
                prepared_data[section_name]["audio_data"]["dialogue"].append(
                    audio_base64
                )

    return prepared_data


def create_html_story(
    story_data_dict: Dict,
    output_path: str,
    component_path: str,
    title: Optional[str] = None,
    language: str = config.get_language_name(),
) -> None:
    """
    Create a standalone HTML file from the story data dictionary.

    Args:
        story_data_dict: Dictionary containing story data, translations, and audio
        output_path: Path where the HTML file should be saved
        component_path: Path to the React component file
        title: Optional title for the story
        language: Target language name for Wiktionary links
    """

    # Process the story data and convert audio to base64
    prepared_data = prepare_story_data_for_html(story_data_dict)

    # Read the React component
    with open(component_path, "r", encoding="utf-8") as f:
        react_component = f.read()

    # Convert the React component from JSX to pure JavaScript
    # Note: In practice, you'd want to use a proper JSX transformer like Babel
    # This is a simplified version that assumes the component is already in JS

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lucide/0.263.1/lucide.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
        <style>
            .audio-player {{
                display: none;
            }}
        </style>
    </head>
    <body>
        <div id="root"></div>
        <script>
            // Embed the story data
            const storyData = {story_data};
            const targetLanguage = "{language}";
            
            {react_component}
            
            // Render the app
            const root = ReactDOM.createRoot(document.getElementById('root'));
            root.render(React.createElement(StoryViewer, {{ 
                storyData: storyData,
                targetLanguage: targetLanguage,
                title: "{title}"
            }}));
        </script>
    </body>
    </html>
    """

    # Format the HTML template
    html_content = html_template.format(
        title=title or "Language Learning Story",
        story_data=json.dumps(prepared_data),
        language=language,
        react_component=react_component,
    )

    # Write the HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML story created at: {output_path}")


def test_image_reading(image_path):
    """
    Test reading and basic image properties from a path

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height) if successful
    """
    try:
        # Open image in binary mode, not text mode
        with Image.open(image_path) as img:
            print(f"Successfully opened image from: {image_path}")
            print(f"Image size: {img.size}")
            print(f"Image mode: {img.mode}")
            return img.size
    except FileNotFoundError:
        print(f"Image file not found at: {image_path}")
    except Exception as e:
        print(f"Error reading image: {str(e)}")


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
    phrases: int = 2,
    from_index: int = 0,
) -> Dict[str, Dict]:
    """
    Create a smaller version of the story_data_dict for testing purposes.

    Args:
    story_data_dict (Dict[str, Dict]): The original story data dictionary.
    story_parts (int): Number of story parts to include in the test dictionary.
    phrases (int): Number of phrases to include in each story part.

    Returns:
    Dict[str, Dict]: A smaller version of the story data dictionary for testing.
    """
    test_dict = {}

    for i, (part_key, part_data) in enumerate(story_data_dict.items()):
        if i >= story_parts:
            break

        test_dict[part_key] = {
            "translated_phrase_list": [],
            "translated_phrase_list_audio": [],
        }

        for j in range(
            from_index,
            min(phrases + from_index, len(part_data["translated_phrase_list"])),
        ):
            test_dict[part_key]["translated_phrase_list"].append(
                part_data["translated_phrase_list"][j]
            )

            # Check if audio data exists and is in the correct format
            try:
                audio_data = part_data["translated_phrase_list_audio"][j]
                test_dict[part_key]["translated_phrase_list_audio"].append(audio_data)
            except KeyError:
                pass

    return test_dict


def update_vocab_usage(used_words: Set[Tuple[str, str]], update_amount: int = 1):
    """Taking a list of (word, word_type) e.g. ('can', 'verbs') we update the vocab_usage
    list, if the word doesn't exist we add it to list. This is used for sampling vocab for subsequent
    lessons. words that haven't been used have a higher chance of being sampled.

    No return statement"""
    # Load the current usage

    vocab_usage = load_json(config.VOCAB_USAGE_PATH)
    # Update the usage count for each used word
    for word, pos in used_words:
        if pos in ["VERB", "AUX"]:
            if word in vocab_usage["verbs"]:
                vocab_usage["verbs"][word] += update_amount
            else:
                vocab_usage["verbs"][word] = update_amount
        else:
            if word in vocab_usage["vocab"]:
                vocab_usage["vocab"][word] += update_amount
            else:
                vocab_usage["vocab"][word] = update_amount

    # Save the updated usage dictionary
    save_json(vocab_usage, config.VOCAB_USAGE_PATH)


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


def load_text_file(file_path) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_json(file_path) -> dict:
    """Returns {} if JSON does not exist"""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)
    # print(f"Data saved to {file_path}")


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
