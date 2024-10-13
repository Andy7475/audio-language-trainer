import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import List, Set, Tuple

import numpy as np
import pycountry
import requests
import spacy
import vertexai
from anthropic import AnthropicVertex
from dotenv import load_dotenv
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from vertexai.preview.vision_models import ImageGenerationModel

from src.config_loader import config

load_dotenv()  # so we can use environment variables for various global settings

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")


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


from typing import Dict, List, Tuple

from pydub import AudioSegment


def create_test_story_dict(
    story_data_dict: Dict[str, Dict], story_parts: int = 2, phrases: int = 2
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

        for j in range(min(phrases, len(part_data["translated_phrase_list"]))):
            test_dict[part_key]["translated_phrase_list"].append(
                part_data["translated_phrase_list"][j]
            )

            # Check if audio data exists and is in the correct format
            audio_data = part_data["translated_phrase_list_audio"][j]
            test_dict[part_key]["translated_phrase_list_audio"].append(audio_data)

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


from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont


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
