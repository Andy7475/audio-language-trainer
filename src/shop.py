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
from src.template_testing import generate_test_html
from src.images import create_png_of_html


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
    x_offset: float = 0,
    background_color: str = "#FFFFFF",
) -> str:
    """
    Creates a single image showing multiple PNG files spread out like a deck of cards.
    Image size is calculated exactly based on the card arrangement.

    Args:
        png_files: List of paths to PNG files to include in the spread
        output_path: Where to save the final image
        angle_offset: Angle in degrees between each card (default: 6.0)
        x_offset: Horizontal offset between cards in pixels - larger values spread cards wider (default: 0)
        background_color: Color of the background (default: "#FFFFFF" - white)

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

        # Add border to the image
        border_size = 2
        bordered_img = Image.new(
            "RGBA",
            (img.width + 2 * border_size, img.height + 2 * border_size),
            (240, 240, 240, 255),
        )  # Pale grey border
        bordered_img.paste(img, (border_size, border_size), img)

        images.append(bordered_img)
        max_card_width = max(max_card_width, bordered_img.width)
        max_card_height = max(max_card_height, bordered_img.height)

    # Calculate exact dimensions needed for the fan arrangement
    # Find the maximum extent of all rotated cards
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")

    # Calculate positions for all cards to find bounds
    card_positions = []

    for draw_order, img in enumerate(images):
        # Calculate angle for this card
        fan_angle = angle_offset * (draw_order - (len(images) - 1) // 2)

        # Rotate the image to get its dimensions
        rotated = img.rotate(fan_angle, expand=True, resample=Image.BICUBIC)

        # Calculate position relative to pivot point (rotation logic)
        angle_rad = math.radians(fan_angle)
        bottom_center_offset_y = img.height // 2 * math.cos(angle_rad)
        bottom_center_offset_x = img.height // 2 * math.sin(angle_rad)

        # Base position from rotation
        base_x = -rotated.width // 2 - bottom_center_offset_x
        base_y = -rotated.height // 2 - bottom_center_offset_y

        # Apply horizontal offset AFTER rotation calculations
        horizontal_offset = x_offset * (draw_order - (len(images) - 1) // 2)
        final_x = base_x + horizontal_offset
        final_y = base_y

        card_positions.append((rotated, final_x, final_y))

        # Update bounds with final positions
        left = final_x
        right = final_x + rotated.width
        top = final_y
        bottom = final_y + rotated.height

        min_x = min(min_x, left)
        max_x = max(max_x, right)
        min_y = min(min_y, top)
        max_y = max(max_y, bottom)

    # Calculate final image dimensions with small padding
    padding = 20
    final_width = int(max_x - min_x) + 2 * padding
    final_height = int(max_y - min_y) + 2 * padding

    # Calculate pivot position in the final image
    pivot_x = -min_x + padding
    pivot_y = -min_y + padding

    # Create new image with exact dimensions
    final_image = Image.new("RGBA", (final_width, final_height), background_color)

    # Draw all cards using pre-calculated positions
    for rotated, rel_x, rel_y in card_positions:
        x = int(pivot_x + rel_x)
        y = int(pivot_y + rel_y)

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
    x_offset: float = 0.0,
    background_color: str = "#FFFFFF",
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
        random.seed(12)
        selected_phrases = random.sample(story_phrases, num_phrases)

    # Generate HTML and PNG files for each phrase
    png_files = []
    for phrase in selected_phrases:
        phrase_key = clean_filename(phrase)

        # Generate HTML files in a temporary directory
        temp_dir = os.path.join(temp_base_dir, f"temp_{phrase_key}")
        html_path = os.path.join(temp_dir, "reading_back.html")
        png_path = os.path.join(temp_dir, f"{phrase_key}.png")

        # Only generate HTML if directory doesn't exist or PNG doesn't exist
        if not os.path.exists(temp_dir) or not os.path.exists(png_path):
            os.makedirs(temp_dir, exist_ok=True)
            generate_test_html(
                phrase_key=phrase_key,
                output_dir=temp_dir,
                collection=collection,
                bucket_name=bucket_name,
            )

            # Create PNG from the reading card front
            create_png_of_html(html_path, png_path, width=375, height=1100)

        png_files.append(png_path)

    # Create the spread deck image
    temp_output = os.path.join(temp_base_dir, "temp_spread_deck.png")
    create_spread_deck_image(
        png_files=png_files,
        output_path=temp_output,
        angle_offset=angle_offset,
        x_offset=x_offset,
        background_color=background_color,
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

    # Clean up only the final spread deck image
    os.remove(temp_output)

    return gcs_uri
