import csv
import os
import random
from typing import Dict, List, Optional
from pathlib import Path
from PIL import Image
import math

from src.config_loader import config
from src.gcs_storage import (
    get_story_collection_path,
    read_from_gcs,
    get_flashcard_path,
    get_phrase_path,
    get_marketing_images_path,
    upload_to_gcs,
)
from src.convert import get_story_title, clean_filename
from src.utils import get_story_position
from src.anki_tools import create_anki_deck_from_gcs
from src.template_testing import generate_test_html, create_png_of_html


def get_existing_flashcards(
    story_names: List[str],
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Get a dictionary of existing flashcard files from the GCS output directory.

    Args:
        story_names: List of story names to check
        collection: Collection name (default: "LM1000")
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)

    Returns:
        Dict mapping story names to their flashcard file paths
    """
    flashcard_files = {}
    bucket_name = bucket_name or config.GCS_PRIVATE_BUCKET
    gcs_output_dir = os.path.join("../outputs/gcs", bucket_name)

    # Get collection data to determine story positions
    collection_path = get_story_collection_path(collection)
    collection_data = read_from_gcs(bucket_name, collection_path, "json")

    for story_name in story_names:
        try:
            position = get_story_position(story_name, collection_data)
            flashcard_path = get_flashcard_path(
                story_name=story_name,
                collection=collection,
                language=config.TARGET_LANGUAGE_NAME.lower(),
                story_position=position,
            )
            full_path = os.path.join(gcs_output_dir, flashcard_path)
            if os.path.exists(full_path):
                flashcard_files[story_name] = full_path
        except ValueError:
            print(f"Story {story_name} not found in path {full_path}")
            continue  # Skip if story not found in collection

    return flashcard_files


def generate_shopify_template(
    story_name: Optional[str | List[str]] = None,
    deck_name: Optional[str] = None,
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
    output_dir: str = "../outputs/shopify",
) -> None:
    """
    Generate a Shopify import template for flashcards.

    Args:
        story_name: Optional name of the story to filter phrases by. Can be a single story name or a list of story names.
        deck_name: Name of the Anki deck
        collection: Collection name (default: "LM1000")
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)
        output_dir: Directory to save the Shopify template
    """
    # Get collection data to determine story positions
    collection_path = get_story_collection_path(collection)
    collection_data = read_from_gcs(
        bucket_name or config.GCS_PRIVATE_BUCKET, collection_path, "json"
    )

    # Convert single story_name to list for consistent handling
    story_names = [story_name] if isinstance(story_name, str) else story_name

    # If no specific stories requested, get all stories from collection
    if not story_names:
        story_names = list(collection_data.keys())

    # Check for existing flashcard files
    existing_flashcards = get_existing_flashcards(story_names, collection, bucket_name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare CSV data
    csv_data = []

    for story_name in story_names:
        if story_name not in existing_flashcards:
            print(f"Warning: No flashcard file found for {story_name}, skipping...")
            continue

        # Get story position
        position = get_story_position(story_name, collection_data)

        # Get flashcard file path
        flashcard_path = existing_flashcards[story_name]

        # Create product handle (URL-friendly version of the title)
        handle = f"{config.TARGET_LANGUAGE_NAME.lower()}-flashcards-{position:02d}-{story_name.replace('story_', '')}"

        # Create product title
        title = (
            f"{config.TARGET_LANGUAGE_NAME} Flashcards - {get_story_title(story_name)}"
        )

        # Create product description
        description = f"""
        <p>Learn {config.TARGET_LANGUAGE_NAME} with our comprehensive flashcard deck for {get_story_title(story_name)}.</p>
        <p>This deck includes:</p>
        <ul>
            <li>Target language phrases with audio</li>
            <li>English translations</li>
            <li>Wiktionary links for vocabulary</li>
            <li>High-quality images</li>
        </ul>
        <p>Perfect for language learners at all levels.</p>
        """

        # Create product row
        product_row = {
            "Handle": handle,
            "Title": title,
            "Body (HTML)": description,
            "Vendor": "Language Learning Tools",
            "Product Category": "Education > Language Learning",
            "Type": "Flashcards",
            "Tags": f"Language Learning, {config.TARGET_LANGUAGE_NAME}, Flashcards, Digital Download",
            "Published": "TRUE",
            "Option1 Name": "Format",
            "Option1 Value": "Digital Download",
            "Variant SKU": f"FLASH-{config.TARGET_LANGUAGE_NAME.upper()}-{position:02d}",
            "Variant Inventory Tracker": "shopify",
            "Variant Inventory Qty": "999",
            "Variant Inventory Policy": "deny",
            "Variant Fulfillment Service": "manual",
            "Variant Price": "9.99",
            "Variant Requires Shipping": "FALSE",
            "Variant Taxable": "TRUE",
            "Status": "active",
        }

        csv_data.append(product_row)

    # Write to CSV file
    output_file = os.path.join(
        output_dir, f"{config.TARGET_LANGUAGE_NAME.lower()}_flashcards_shopify.csv"
    )

    # Get field names from template
    template_path = os.path.join("src", "templates", "shopify_template.csv")
    with open(template_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    # Write data to CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Shopify template generated: {output_file}")


def create_spread_deck_image(
    png_files: List[str],
    output_path: str,
    angle_offset: float = 6.0,
    overlap: float = 0.3,
    background_color: str = "#303030",
    max_width: int = 3000,
    max_height: int = 2000,
) -> str:
    """
    Creates a single image showing multiple PNG files spread out like a deck of cards.

    Args:
        png_files: List of paths to PNG files to include in the spread
        output_path: Where to save the final image
        angle_offset: Angle in degrees between each card (default: 6.0)
        overlap: How much each card should overlap with the next (0.0 to 1.0, default: 0.3)
        background_color: Color of the background (default: "#303030")
        max_width: Maximum width of the output image (default: 3000)
        max_height: Maximum height of the output image (default: 2000)

    Returns:
        Path to the created image
    """
    if not png_files:
        raise ValueError("No PNG files provided")

    # Load all images
    images = []
    max_card_width = 0
    max_card_height = 0

    for file in png_files:
        img = Image.open(file)
        # Convert to RGBA if not already
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        images.append(img)
        max_card_width = max(max_card_width, img.width)
        max_card_height = max(max_card_height, img.height)

    # Calculate total width needed for spread
    total_angle = angle_offset * (len(images) - 1)
    # Use sine to calculate the maximum width needed
    max_spread_width = max_card_width * (1 + math.sin(math.radians(total_angle)))

    # Calculate dimensions for the final image
    final_width = min(int(max_spread_width * (1 + overlap * 2)), max_width)
    final_height = min(int(max_card_height * 1.5), max_height)

    # Create new image with background
    final_image = Image.new("RGBA", (final_width, final_height), background_color)

    # Calculate starting position (bottom left)
    start_x = int(final_width * 0.1)
    start_y = int(final_height * 0.8)

    # Place each image with rotation
    for i, img in enumerate(images):
        # Calculate angle for this card (clockwise from bottom left)
        angle = -angle_offset * i

        # Rotate the image
        rotated = img.rotate(angle, expand=True, resample=Image.BICUBIC)

        # Calculate position for this card
        # Move each card slightly to the right and adjust for rotation
        x_offset = int(i * max_card_width * overlap)
        y_offset = int(math.sin(math.radians(angle)) * max_card_width * 0.5)

        # Calculate position
        x = start_x + x_offset
        y = start_y + y_offset - rotated.height

        # Paste the rotated image
        final_image.paste(rotated, (x, y), rotated)

    # Save the final image
    final_image.save(output_path, "PNG")
    return output_path


def generate_spread_deck_image(
    story_positions: Optional[List[int]] = None,
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
    num_phrases: int = 5,
    angle_offset: float = 6.0,
    overlap: float = 0.3,
    background_color: str = "#303030",
    max_width: int = 3000,
    max_height: int = 4000,
) -> List[str]:
    """
    Generate spread deck images from phrases in specified stories or collection and upload to GCS.

    Args:
        story_positions: Optional list of story positions to get phrases from (e.g. [1, 2, 3] for first three stories)
                        If None, samples from entire collection
        collection: Collection name (default: "LM1000")
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)
        num_phrases: Number of phrases to include in the spread (default: 5)
        angle_offset: Angle in degrees between each card (default: 6.0)
        overlap: How much each card should overlap with the next (0.0 to 1.0, default: 0.3)
        background_color: Color of the background (default: "#303030")
        max_width: Maximum width of the output image (default: 2000)
        max_height: Maximum height of the output image (default: 1000)

    Returns:
        List of GCS URIs of the uploaded spread deck images
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Create temp directory
    temp_base_dir = "../outputs/temp"
    os.makedirs(temp_base_dir, exist_ok=True)

    # Get collection data
    collection_path = get_story_collection_path(collection)
    collection_data = read_from_gcs(bucket_name, collection_path, "json")

    # Get story names based on positions if provided
    if story_positions:
        # Convert collection_data keys to list to maintain order
        story_names = list(collection_data.keys())
        selected_stories = []
        for pos in story_positions:
            if 1 <= pos <= len(story_names):
                selected_stories.append(story_names[pos - 1])
            else:
                print(
                    f"Warning: Story position {pos} is out of range (1-{len(story_names)})"
                )

        if not selected_stories:
            raise ValueError("No valid story positions provided")

        # Get phrases from selected stories
        story_phrases = []
        for story_name in selected_stories:
            story_phrases.extend([p["phrase"] for p in collection_data[story_name]])
    else:
        # Get all phrases from the collection
        phrases_path = get_phrase_path(collection)
        story_phrases = read_from_gcs(bucket_name, phrases_path, "json")

    # Randomly sample phrases
    if len(story_phrases) < num_phrases:
        print(
            f"Warning: Only {len(story_phrases)} phrases available, using all of them"
        )
        selected_phrases = story_phrases
    else:
        selected_phrases = random.sample(story_phrases, num_phrases)

    # Generate HTML and PNG files for each phrase
    png_files = []
    for phrase in selected_phrases:
        phrase_key = clean_filename(phrase)

        # Generate HTML files in a temporary directory
        temp_dir = os.path.join(temp_base_dir, f"temp_{phrase_key}")
        os.makedirs(temp_dir, exist_ok=True)

        generate_test_html(
            phrase_key=phrase_key,
            output_dir=temp_dir,
            collection=collection,
            bucket_name=bucket_name,
        )

        # Create PNG from the reading card front
        html_path = os.path.join(temp_dir, "reading_back.html")
        png_path = os.path.join(temp_dir, f"{phrase_key}.png")
        create_png_of_html(html_path, png_path, width=375, height=1100)
        png_files.append(png_path)

    # Create the spread deck image
    temp_output = os.path.join(temp_base_dir, "temp_spread_deck.png")
    create_spread_deck_image(
        png_files=png_files,
        output_path=temp_output,
        angle_offset=angle_offset,
        overlap=overlap,
        background_color=background_color,
        max_width=max_width,
        max_height=max_height,
    )

    # Upload to GCS
    marketing_path = get_marketing_images_path(collection)
    language = config.TARGET_LANGUAGE_NAME.lower()

    # Generate filename based on story positions or collection
    if story_positions:
        story_range = f"stories_{min(story_positions):02d}-{max(story_positions):02d}"
    else:
        story_range = "collection"

    filename = f"{language}_{collection}_spread_deck_{story_range}.png"

    # Read the image file
    with open(temp_output, "rb") as f:
        image_data = f.read()

    # Upload to GCS
    gcs_uri = upload_to_gcs(
        obj=image_data,
        bucket_name=bucket_name,
        file_name=filename,
        base_prefix=marketing_path,
        content_type="image/png",
    )

    # Clean up temporary files
    for png_file in png_files:
        os.remove(png_file)
        os.remove(os.path.join(os.path.dirname(png_file), "reading_back.html"))
        os.rmdir(os.path.dirname(png_file))
    os.remove(temp_output)

    return gcs_uri
