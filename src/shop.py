import csv
import os
from typing import Dict, List, Optional
from pathlib import Path

from src.config_loader import config
from src.gcs_storage import get_story_collection_path, read_from_gcs, get_flashcard_path
from src.convert import get_story_title
from src.utils import get_story_position
from src.anki_tools import create_anki_deck_from_gcs


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


if __name__ == "__main__":
    # Example usage
    generate_shopify_template(
        story_name="story_community_park",
        deck_name="French - Community Park",
        collection="LM1000",
    )
