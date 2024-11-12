import base64
import copy
import hashlib
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
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from tqdm import tqdm
from vertexai.generative_models import HarmCategory, SafetySetting
from vertexai.preview.vision_models import ImageGenerationModel

from src.config_loader import config

load_dotenv()  # so we can use environment variables for various global settings

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")


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


def generate_story_image(story_plan):
    """
    Generate an image for a story using Google Cloud Vertex AI's Image Generation API.

    :param story_plan: A string containing the story plan
    :param project_id: Your Google Cloud project ID
    :param location: The location of your Vertex AI endpoint
    :return: Image data as bytes
    """
    # Initialize Vertex AI
    vertexai.init(project=config.PROJECT_ID, location=config.VERTEX_REGION)

    # Initialize the Image Generation model
    generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

    # Craft the prompt
    prompt = f"""
    Create a colorful, engaging image for a language learning story. 
    The image should be suitable as album art for an educational audio file.
    The story is about: {story_plan}
    The image should be family-friendly and appropriate for all ages.
    The style should be hand-painted (not a photo). It should not contain any people.
    """

    # Generate the image
    images = generation_model.generate_images(
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="1:1",
        # safety_filter_level="block_some",
        person_generation="don't allow",
    )

    # Get the first (and only) generated image
    generated_image = images[0]

    # Get the image bytes directly
    image_data = generated_image._image_bytes

    # Convert the image to PIL Image for potential resizing
    image = Image.open(io.BytesIO(image_data))

    # Resize the image if it's not 500x500
    if image.size != (500, 500):
        image = image.resize((500, 500))

        # If we resized, convert the resized image back to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        image_data = img_byte_arr.getvalue()

    return image_data


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


def add_image_paths(story_dict: Dict[str, Any], image_dir: str) -> Dict[str, Any]:
    """
    Add image paths to the story dictionary based on the English phrases.

    Args:
        story_dict: Dictionary containing story data with translated_phrase_list
        image_dir: Directory containing the images

    Returns:
        Updated dictionary with image_path added for each story part

    Note:
        For each story part, expects translated_phrase_list to be a list of tuples
        where each tuple is (english_text, target_text)
    """
    # Create a deep copy of the dictionary to avoid modifying nested structures
    updated_dict = copy.deepcopy(story_dict)

    for story_part, data in tqdm(updated_dict.items(), desc="Processing story parts"):
        # Initialize image_path list for this story part
        data["image_path"] = []

        # Get the phrases from translated_phrase_list
        phrase_list = data.get("translated_phrase_list", [])

        for eng_phrase, _ in tqdm(
            phrase_list, desc=f"Adding image paths for {story_part}", leave=False
        ):
            # Generate the expected image filename from English phrase
            clean_name = clean_filename(eng_phrase)
            image_filename = f"{clean_name}.png"
            full_path = os.path.join(image_dir, image_filename)

            # Check if the image exists and is readable
            if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                data["image_path"].append(full_path)
            else:
                print(f"Warning: Image not found or not readable: {full_path}")
                data["image_path"].append(None)

        # Verify lengths match
        if len(data["image_path"]) != len(phrase_list):
            raise ValueError(
                f"Mismatch in {story_part}: {len(data['image_path'])} images "
                f"vs {len(phrase_list)} phrases"
            )

    return updated_dict


def add_images_to_phrases(
    phrases: List[str],
    output_dir: str,
    image_format: str = "png",
    imagen_model: Literal[
        "imagen-3.0-fast-generate-001", "imagen-3.0-generate-001"
    ] = "imagen-3.0-generate-001",
    anthropic_model=config.ANTHROPIC_MODEL_NAME,
) -> Dict:
    """
    Process a list of phrases to create a dictionary with prompts and image paths.

    Args:
        phrases: List of English phrases
        output_dir: Directory where images will be saved
        generate_image_prompt: Function that takes a phrase and returns a prompt
        generate_image: Function that takes a prompt and returns image data
        image_format: Image file format (default: 'png')

    Returns:
        Dictionary containing phrases, prompts, and image paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results dictionary
    results = {}

    for phrase in tqdm(phrases):
        # Create a clean filename from the phrase
        clean_name = clean_filename(phrase)

        # Generate image filename
        image_filename = f"{clean_name}.{image_format}"
        image_path = os.path.join(output_dir, image_filename)

        if os.path.exists(image_path):
            print(f"Warning: Image already exists for '{phrase}', skipping generation")
            results[clean_name] = {
                "phrase": phrase,
                "prompt": None,
                "image_path": image_path,
            }
            continue
        # Generate prompt for the phrase
        prompt = create_image_generation_prompt(phrase, anthropic_model)

        # Generate and save the image
        try:
            image = generate_image_imagen(prompt)
            if image is None:
                image = generate_image_deepai(prompt)

            if image is None:
                print("Both image generation attempts failed, skipping")
                continue
            # Save image to file
            image.save(image_path)

            # Store results in dictionary
            results[clean_name] = {
                "phrase": phrase,
                "prompt": prompt,
                "image_path": image_path,
            }

            print(f"Successfully processed: {phrase}. Now sleeping.")
            pbar = tqdm(range(45), desc="Sleeping", ncols=75, colour="blue")
            for sec in pbar:
                time.sleep(1)
                pbar.refresh()

        except Exception as e:
            print(f"Error processing phrase '{phrase}': {str(e)}")
            continue

    return results


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


def create_image_generation_prompt(phrase, anthropic_model: str = None):
    """
    Create a specific image generation prompt based on a language learning phrase.

    :param phrase: The language learning phrase to visualize
    :return: A specific prompt for image generation
    """

    llm_prompt = f"""
    Given the following phrase for language learners: "{phrase}"
    
    Create a specific, detailed prompt for generating an image that will help learners remember this phrase.
    Focus on key nouns, verbs, or concepts that can be visually represented.
    The image should be memorable and directly related to the meaning of the phrase.
    
    Your prompt should:
    1. Ensure you consider every element from the phrase to visualize.
    2. Suggest a simple scene or composition that incorporates these elements. You can use your imagination to make it more memorable
    3. Include any relevant emotions, or atmosphere that would enhance memory retention.
    4. Limit your output to 1 - 2 sentences. Do not add details of the image style, this will be added later.
    
    Provide only the image generation prompt, without any explanations or additional text.

    Example phrase: "The bride watched the sunset from the balcony"
    Example Output: "A bride on a balcony, looking at sunset over the horizon, tropical island, villa"
    """

    base_style = "a children's book illustration, Axel Scheffler style, thick brushstrokes, colored pencil texture, expressive characters, bold outlines, textured shading, pastel color palette"

    # Use the anthropic_generate function to get the LLM's response
    image_prompt = anthropic_generate(llm_prompt, model=anthropic_model)
    image_prompt.strip('".')

    return image_prompt + f" in the style of {base_style}"


def generate_image_deepai(
    prompt: str,
    width: Union[str, int] = "512",
    height: Union[str, int] = "512",
    model: Literal["standard", "hd"] = "hd",
    negative_prompt: Optional[str] = None,
) -> Image.Image:
    """
    Generate an image using DeepAI's text2img API and return it as a PIL Image object.

    Args:
        prompt (str): The text prompt to generate the image from
        width (Union[str, int]): Image width (128-1536, default 512)
        height (Union[str, int]): Image height (128-1536, default 512)
        model (str): Model version ("standard" or "hd")
        negative_prompt (Optional[str]): Text describing what to remove from the image

    Returns:
        PIL.Image.Image: The generated image as a PIL Image object

    Raises:
        Exception: If there's an error in image generation or processing
        EnvironmentError: If DEEPAI_API_KEY environment variable is not set
    """
    # Get API key from environment variable
    api_key = os.getenv("DEEPAI_API_KEY")
    if not api_key:
        raise EnvironmentError("DEEPAI_API_KEY environment variable not set")

    try:
        # Convert width and height to strings if they're integers
        width = str(width)
        height = str(height)

        # Prepare the API request data
        data = {
            "text": prompt,
            "width": width,
            "height": height,
            "image_generator_version": model,
        }

        # Add negative prompt if provided
        if negative_prompt:
            data["negative_prompt"] = negative_prompt

        # Make the API request
        response = requests.post(
            "https://api.deepai.org/api/text2img",
            data=data,
            headers={"api-key": api_key},
        )

        # Check if the request was successful
        response.raise_for_status()

        # Get the URL of the generated image from the response
        result = response.json()
        if "output_url" not in result:
            raise Exception(f"Unexpected API response: {result}")

        # Download the image from the URL
        image_response = requests.get(result["output_url"])
        image_response.raise_for_status()

        # Convert to PIL Image
        image = Image.open(BytesIO(image_response.content))

        return image

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error making request to DeepAI API: {str(e)}")
    except Exception as e:
        raise Exception(f"Error generating image with DeepAI: {str(e)}")


def generate_image_stability(
    prompt: str,
    negative_prompt: str = "",
    style_preset: Optional[
        Literal[
            "3d-model",
            "analog-film",
            "anime",
            "cinematic",
            "comic-book",
            "digital-art",
            "enhance",
            "fantasy-art",
            "isometric",
            "line-art",
            "low-poly",
            "modeling-compound",
            "neon-punk",
            "origami",
            "photographic",
            "pixel-art",
            "tile-texture",
        ]
    ] = None,
    endpoint=config.STABILITY_ENDPOINT,
) -> Optional[Image.Image]:
    """
    Generate an image using Stability AI's core model API.

    Args:
        prompt (str): Text description of the desired image
        negative_prompt (str): Text description of what to avoid in the image
        style_preset (str, optional): Style preset to use for image generation

    Returns:
        bytes: Generated image data, or None if generation fails

    Raises:
        ValueError: If API key is missing
        Warning: If content is filtered (NSFW)
        requests.RequestException: If the API request fails
    """
    if not STABILITY_API_KEY:
        raise ValueError("STABILITY_API_KEY environment variable not set")

    # Prepare headers
    headers = {"Accept": "image/*", "Authorization": f"Bearer {STABILITY_API_KEY}"}

    # Prepare form data
    files = {
        "prompt": (None, prompt),
    }

    # Add optional parameters if provided
    if negative_prompt:
        files["negative_prompt"] = (None, negative_prompt)

    if style_preset:
        files["style_preset"] = (None, style_preset)

    try:
        # Make the API request
        response = requests.post(
            endpoint,
            headers=headers,
            files=files,
        )

        # Check if request was successful
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.content}")
            return None

        # Check for content filtering
        finish_reason = response.headers.get("finish-reason")
        if finish_reason == "CONTENT_FILTERED":
            raise Warning("Generation failed NSFW classifier")

        # Return the raw image content
        return Image.open(io.BytesIO(response.content))

    except requests.RequestException as e:
        print(f"Request failed: {str(e)}")
        return None
    except Warning as w:
        print(f"Content filtered: {str(w)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None


def generate_image_imagen(
    prompt: str,
    model: Literal[
        "imagen-3.0-fast-generate-001", "imagen-3.0-generate-001"
    ] = "imagen-3.0-generate-001",
) -> Optional[Image.Image]:
    """
    Generate an image using the Vertex AI Imagen model with retry logic.

    Args:
        prompt: The text prompt to generate the image from
        model: The Imagen model to use
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds between retries
        delay_multiplier: Factor to multiply delay by after each attempt

    Returns:
        Generated image bytes from the model

    Raises:
        Exception: If image generation fails after all retry attempts
    """
    vertexai.init(project=config.PROJECT_ID, location=config.VERTEX_REGION)
    generation_model = ImageGenerationModel.from_pretrained(model)

    try:
        # Generate the image
        images = generation_model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            # person_generation="allow_adult",
            # safety_filter_level="block_fewest",
        )

        if len(images.images) > 0:

            return Image.open(io.BytesIO(images.images[0]._image_bytes))
        else:
            print(f"No image generated using {model} with prompt: {prompt}")
            return None

    except Exception as e:
        print(f"Imagen generation failed with error {e}")
        return None


def resize_image(generated_image, height=500, width=500):

    # Get the image bytes directly
    image_data = generated_image._image_bytes

    # Convert the image to PIL Image for potential resizing
    image = Image.open(io.BytesIO(image_data))

    # Resize the image if it's not 500x500
    if image.size != (500, 500):
        image = image.resize((500, 500))

        # If we resized, convert the resized image back to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        image_data = img_byte_arr.getvalue()

    return image_data


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


def ensure_spacy_model(model_name="en_core_web_md"):
    try:
        spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model {model_name}...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])


def extract_vocab_and_pos(english_phrases: List[str]) -> List[Tuple[str, str]]:
    """Returns the (lemma and POS) for feeding into update_vocab_usage, as a list."""
    # Process vocabulary
    ensure_spacy_model()
    nlp = spacy.load("en_core_web_md")

    vocab_set = set()
    excluded_names = {"sam", "alex"}

    for phrase in english_phrases:
        doc = nlp(phrase)

        for token in doc:
            if (
                token.pos_ != "PUNCT"
                and token.ent_type_ != "PERSON"
                and token.text.lower() not in excluded_names
            ):
                vocab_set.add((token.lemma_.lower(), token.pos_))

    return vocab_set


def extract_substring_matches(
    new_phrases: List[str], target_phrases: Set[str]
) -> Set[str]:
    # Convert all new phrases to lowercase
    lowercase_phrases = [phrase.lower() for phrase in new_phrases]

    # Convert all target phrases to lowercase
    lowercase_targets = [target.lower() for target in target_phrases]

    # Initialize a set to store matched substrings
    matched_substrings = set()

    # Check each target phrase against each new phrase
    for target in lowercase_targets:
        # Create a regex pattern that matches the target as a whole word or phrase
        pattern = r"\b" + re.escape(target) + r"\b"
        for phrase in lowercase_phrases:
            if re.search(pattern, phrase):
                matched_substrings.add(target)
                break  # Move to the next target once a match is found

    return matched_substrings


def extract_spacy_lowercase_words(new_phrases: List[str]) -> Set[str]:
    # Ensure the spaCy model is loaded
    nlp = spacy.load("en_core_web_sm")

    # Initialize an empty set to store unique lowercase words
    lowercase_words = set()

    # Process each phrase with spaCy
    for phrase in new_phrases:
        doc = nlp(phrase)

        # Add the lowercase version of each token's text to the set
        lowercase_words.update(token.text.lower() for token in doc)

    return lowercase_words


def get_verb_and_vocab_lists(used_words: Set[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Separate the input set of (word, POS) tuples into verb and vocabulary lists.

    Args:
    used_words (Set[Tuple[str, str]]): A set of tuples containing (word, POS)

    Returns:
    Dict[str, List[str]]: A dictionary with 'verbs' and 'vocab' lists
    """
    verb_list = []
    vocab_list = []

    for word, pos in used_words:
        if pos in ["VERB", "AUX"]:
            verb_list.append(word)
        else:
            vocab_list.append(word)

    return {"verbs": verb_list, "vocab": vocab_list}


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


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)
    # print(f"Data saved to {file_path}")


def anthropic_generate(prompt: str, max_tokens: int = 1024, model: str = None) -> str:
    """given a prompt generates an LLM response. The default model is specified in the config file.
    Most likely the largest Anthropic model. The region paramater in the config will have to match where that model
    is available"""
    client = AnthropicVertex(region=config.ANTHROPIC_REGION, project_id=PROJECT_ID)

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


def download_font(url, font_name):
    response = requests.get(url)
    if response.status_code == 200:
        font_path = f"{font_name.replace(' ', '_')}.ttf"
        font_path = os.path.join("../fonts", font_path)
        with open(font_path, "wb") as f:
            f.write(response.content)
        return font_path
    return None


def create_pdf_booklet(story_data_dict, output_filename="pdf_booklet.pdf"):
    # Font setup
    font_name, font_path = setup_font(config.TARGET_LANGUAGE)

    # Define styles
    styles = define_styles(font_name)

    # Create document
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    elements = []

    # Add title
    elements.append(Paragraph("Comprehensive Story Translation", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Process story data
    for story_part, data in story_data_dict.items():
        elements.extend(process_story_part(story_part, data, styles))

    # Build the document
    doc.build(elements)


# Language to font mapping
LANGUAGE_FONT_MAP = {
    "ja": {"name": "HeiseiMin-W3", "google": False},
    "zh": {"name": "STSong-Light", "google": False},
    "ko": {"name": "HYSMyeongJoStd-Medium", "google": False},
    "ru": {"name": "DejaVuSerif", "google": False},
    "sr": {"name": "DejaVuSerif", "google": False},
    "en": {"name": "Roboto", "google": True},
    "sv": {"name": "Roboto", "google": True},
    # Add more languages as needed
}


def get_google_font(language_code):
    # Map language codes to Google Font names

    font_name = LANGUAGE_FONT_MAP.get(language_code, {"name": "Roboto"})[
        "name"
    ]  # Default to Roboto if language not found

    # URL for Google Fonts API
    api_url = f"https://fonts.googleapis.com/css?family={font_name.replace(' ', '+')}"

    response = requests.get(api_url)
    if response.status_code == 200:
        # Extract the TTF URL from the CSS
        for line in response.text.split("\n"):
            if ".ttf" in line:
                ttf_url = line.split("url(")[1].split(")")[0]
                return font_name, ttf_url

    return None, None


def setup_font(target_language):
    font_info = LANGUAGE_FONT_MAP.get(
        target_language, {"name": "Helvetica", "google": False}
    )
    font_name = font_info["name"]

    if font_info["google"]:
        font_name, font_url = get_google_font(target_language)
        if font_name and font_url:
            font_path = download_font(font_url, font_name)
            if font_path:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                print(f"Registered Google font: {font_name}")
                return font_name, font_path
    else:
        if font_name != "Helvetica":
            try:
                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
            except KeyError:
                pdfmetrics.registerFont(TTFont(font_name, "../fonts/DejaVuSans.ttf"))
            except Exception(f"Some font error with {font_name}"):
                font_name = "Helvetica"

            print(f"Registered CID font: {font_name}")
        else:
            print("Using default font: Helvetica")

    return font_name, None


def define_styles(font_name):
    styles = getSampleStyleSheet()

    # Modify existing styles
    styles["Title"].alignment = 1
    styles["Title"].fontName = font_name

    styles["Heading2"].alignment = 1
    styles["Heading2"].fontName = font_name

    styles.add(
        ParagraphStyle(
            name="Subtitle", parent=styles["Heading2"], alignment=1, fontName=font_name
        )
    )
    # Add new styles
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["Normal"],
            fontSize=14,
            fontName=font_name,
            textColor=colors.whitesmoke,
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["Normal"],
            fontSize=10,
            leading=12,
            alignment=TA_LEFT,
            fontName=font_name,
        )
    )
    return styles


def process_story_part(story_part, data, styles):
    elements = []
    elements.append(Paragraph(story_part.capitalize(), styles["Subtitle"]))
    elements.append(Spacer(1, 12))

    if "translated_phrase_list" in data:
        elements.extend(create_phrase_table(data["translated_phrase_list"], styles))

    if "dialogue" in data and "translated_dialogue" in data:
        elements.extend(
            create_dialogue_table(data["dialogue"], data["translated_dialogue"], styles)
        )

    elements.append(Spacer(1, 24))
    return elements


def create_phrase_table(phrases, styles):
    elements = []
    elements.append(Paragraph("Translated Phrases", styles["Subtitle"]))

    data = [["English", "Target Language"]]
    for eng, target in phrases:
        data.append(
            [
                Paragraph(eng, styles["TableCell"]),
                Paragraph(target, styles["TableCell"]),
            ]
        )

    table = Table(data, colWidths=[250, 250])
    table.setStyle(create_table_style(styles["TableCell"].fontName))

    elements.append(table)
    elements.append(Spacer(1, 12))
    return elements


def create_dialogue_table(dialogue, translated_dialogue, styles):
    elements = []
    elements.append(Paragraph("Dialogue", styles["Subtitle"]))

    data = [["English", "Target Language"]]
    for eng, target in zip(dialogue, translated_dialogue):
        data.append(
            [
                Paragraph("{}".format(eng["text"]), styles["TableCell"]),
                Paragraph("{}".format(target["text"]), styles["TableCell"]),
            ]
        )

    table = Table(data, colWidths=[250, 250])
    table.setStyle(create_table_style(styles["TableCell"].fontName))

    elements.append(table)
    return elements


def create_table_style(font_name):
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, 0), 14),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("TOPPADDING", (0, 1), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
