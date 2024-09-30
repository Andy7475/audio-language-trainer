import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import List, Set, Tuple

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
import pycountry
from src.config_loader import config

load_dotenv()  # so we can use environment variables for various global settings

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")


def language_to_int(language_name: str, max_value: int = 1000000) -> int:
    """
    Convert a language name to a consistent integer value.

    :param language_name: The name of the language
    :param max_value: The maximum value for the generated integer (default: 1,000,000)
    :return: An integer representation of the language name
    """
    # Normalize the input
    normalized_name = language_name.lower().strip()

    # Use Python's built-in hash function
    hash_value = hash(normalized_name)

    # Ensure the value is positive and within the desired range
    positive_hash = abs(hash_value) % max_value

    return positive_hash


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
            if j < len(part_data["translated_phrase_list_audio"]):
                audio_data = part_data["translated_phrase_list_audio"][j]
                if isinstance(audio_data, AudioSegment):
                    test_dict[part_key]["translated_phrase_list_audio"].append(
                        audio_data
                    )
                elif (
                    isinstance(audio_data, list)
                    and len(audio_data) > 0
                    and isinstance(audio_data[2], AudioSegment)
                ):
                    test_dict[part_key]["translated_phrase_list_audio"].append(
                        audio_data[2]
                    )
                else:
                    print(
                        f"Warning: Unexpected audio data format in part {part_key}, phrase {j}"
                    )

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
    print(used_words)
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
    print(f"Data saved to {file_path}")


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


def create_pdf_booklet(story_data_dict, output_filename="pdf_booklet.pdf"):
    # Register a Unicode font that supports Serbian characters

    font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
    pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))

    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    title_style.alignment = 1  # Center alignment
    title_style.fontName = "DejaVuSans"
    subtitle_style = styles["Heading2"]
    subtitle_style.alignment = 1  # Center alignment
    subtitle_style.fontName = "DejaVuSans"

    # Create a custom style for table cells
    cell_style = ParagraphStyle(
        "CellStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=12,
        alignment=TA_LEFT,
        fontName="DejaVuSans",
    )

    elements.append(Paragraph("Comprehensive Story Translation", title_style))
    elements.append(Spacer(1, 12))

    for story_part, data in story_data_dict.items():
        # Add story part title
        elements.append(Paragraph(story_part.capitalize(), subtitle_style))
        elements.append(Spacer(1, 12))

        # Process translated phrases
        if "translated_phrase_list" in data:
            elements.append(Paragraph("Translated Phrases", styles["Heading3"]))
            phrases_data = [["English", "Target Language"]]  # Table header
            for eng, swe in data["translated_phrase_list"]:
                phrases_data.append(
                    [Paragraph(eng, cell_style), Paragraph(swe, cell_style)]
                )

            phrases_table = Table(phrases_data, colWidths=[250, 250])
            phrases_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
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
            )
            elements.append(phrases_table)
            elements.append(Spacer(1, 12))

        # Process dialogue
        if "dialogue" in data and "translated_dialogue" in data:
            elements.append(Paragraph("Dialogue", styles["Heading3"]))
            dialogue_data = [["English", "Target Language"]]  # Table header
            for eng, swe in zip(data["dialogue"], data["translated_dialogue"]):
                dialogue_data.append(
                    [
                        Paragraph(f"{eng['text']}", cell_style),
                        Paragraph(f"{swe['text']}", cell_style),
                    ]
                )

            dialogue_table = Table(dialogue_data, colWidths=[250, 250])
            dialogue_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
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
            )
            elements.append(dialogue_table)
            elements.append(Spacer(1, 24))  # Add extra space between story parts

    doc.build(elements)
