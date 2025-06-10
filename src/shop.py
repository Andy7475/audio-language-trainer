import csv
import math
import os
import random
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

from src.config_loader import config
from src.convert import clean_filename, get_story_title, get_collection_title
from src.gcs_storage import (
    get_phrase_path,
    get_story_collection_path,
    get_story_index_path,
    read_from_gcs,
    upload_to_gcs,
    get_marketing_image_path,
    get_public_story_path,
)
from src.images import create_png_of_html
from src.template_testing import generate_test_html
from src.utils import get_story_position, load_template


def calculate_vocab_stats(
    story_names: List[str], story_index: Dict, story_collection: Dict
) -> Tuple[int, int, int]:
    """
    Calculate vocabulary statistics for a list of stories.

    Args:
        story_names: List of story names
        story_index: Story index data from GCS
        story_collection: Story collection data from GCS

    Returns:
        Tuple of (verb_count, vocab_count, phrase_count)
    """
    all_verbs = set()
    all_vocab = set()
    total_phrases = 0

    for story in story_names:
        if story in story_index.get("story_vocab", {}):
            story_verbs = story_index["story_vocab"][story].get("verbs", [])
            story_vocab = story_index["story_vocab"][story].get("vocab", [])
            all_verbs.update(story_verbs)
            all_vocab.update(story_vocab)

        if story in story_collection:
            total_phrases += len(story_collection[story])

    # Round down to nearest 10 for marketing purposes
    verb_count = math.floor(len(all_verbs) / 10) * 10
    vocab_count = math.floor(len(all_vocab) / 10) * 10

    return verb_count, vocab_count, total_phrases


def get_sample_phrases(
    story_name: str, story_collection: Dict, count: int = 5
) -> List[str]:
    """Get sample phrases from a story."""
    if story_name not in story_collection:
        return []

    phrases = [item["phrase"] for item in story_collection[story_name][:count]]
    return phrases


def generate_story_list_html(story_names: List[str], collection: str) -> str:
    """Generate HTML list of stories with their positions and titles."""
    story_items = []
    for story in story_names:
        try:
            position = get_story_position(story, collection)
            title = get_story_title(story)
            story_items.append(f'<li>Story {position:02d}: "{title}"</li>')
        except ValueError:
            continue

    return "<ul>" + "\n".join(story_items) + "</ul>"


def create_product_templates(
    collection: str,
    language: str,
    stories: List[Dict],
    product_config: dict,
    total_phrases: int,
    total_audio_files: int,
    verb_count: int,
    collection_title: str,
) -> Dict[str, str]:
    """Create product templates for individual packs, bundle packs, and complete pack."""
    templates = {}

    # Create individual pack templates
    if "individual" in product_config:
        individual_config = product_config["individual"]
        individual_template = load_template(individual_config["template"])
        individual_templates = {}
        for story in stories:
            if story["position"] in individual_config.get("indices", []):
                story_position = story["position"]
                story_title = story["title"]
                story_theme = story["theme"]
                phrase_count = story["phrase_count"]
                sample_phrases = story["sample_phrases"]
                sample_phrases_html = "\n".join(
                    [f"<li>{phrase}</li>" for phrase in sample_phrases]
                )
                story_hyperlink = get_public_story_path(story["name"], collection)
                # Convert to full public URL
                public_bucket_name = config.GCS_PUBLIC_BUCKET
                story_hyperlink = f"https://storage.googleapis.com/{public_bucket_name}/{story_hyperlink}"
                template = individual_template.replace("${story_title}", story_title)
                template = template.replace("${story_position}", f"{story_position:02d}")
                template = template.replace("${collection}", collection)
                template = template.replace("${collection_title}", collection_title)
                template = template.replace("${story_theme}", story_theme)
                template = template.replace("${phrase_count}", str(phrase_count))
                template = template.replace(
                    "${sample_phrases_html}", sample_phrases_html
                )
                template = template.replace("${language}", language)
                template = template.replace("${story_name}", story["name"])
                template = template.replace("${story_hyperlink}", story_hyperlink)
                individual_templates[f"story_{story_position}"] = template
        templates["individual_templates"] = individual_templates

    # Create bundle pack templates
    if "bundle" in product_config:
        bundle_config = product_config["bundle"]
        bundle_template = load_template(bundle_config["template"])
        bundle_templates = {}
        for start, end in bundle_config.get("ranges", []):
            bundle_stories = [s for s in stories if start <= s["position"] <= end]
            if not bundle_stories:
                continue
            range_display = f"{start}-{end}"
            story_list = "\n".join(
                [
                    f'<li><a href="https://storage.googleapis.com/{config.GCS_PUBLIC_BUCKET}/{get_public_story_path(s["name"], collection)}">Story {s["position"]}: {s["title"]}</a></li>'
                    for s in bundle_stories
                ]
            )
            story_count = len(bundle_stories)
            vocab_count = sum(s["phrase_count"] for s in bundle_stories)
            template = bundle_template.replace("${collection}", collection)
            template = template.replace("${collection_title}", collection_title)
            template = template.replace("${range_display}", range_display)
            template = template.replace("${story_list}", story_list)
            template = template.replace("${story_count}", str(story_count))
            template = template.replace("${vocab_count}", str(vocab_count))
            template = template.replace("${total_phrases}", str(total_phrases))
            bundle_templates[f"bundle_{start}_{end}"] = template
        templates["bundle_templates"] = bundle_templates

    # Create complete pack template
    if "complete" in product_config:
        complete_config = product_config["complete"]
        complete_template = load_template(complete_config["template"])
        story_list = "\n".join(
            [
                f'<li><a href="https://storage.googleapis.com/{config.GCS_PUBLIC_BUCKET}/{get_public_story_path(s["name"], collection)}">Story {s["position"]}: {s["title"]}</a></li>'
                for s in stories
            ]
        )
        complete_template = complete_template.replace("${collection}", collection)
        complete_template = complete_template.replace(
            "${collection_title}", collection_title
        )
        complete_template = complete_template.replace(
            "${total_stories}", str(len(stories))
        )
        complete_template = complete_template.replace("${verb_count}", str(verb_count))
        complete_template = complete_template.replace(
            "${total_phrases}", str(total_phrases)
        )
        complete_template = complete_template.replace(
            "${audio_files}", str(total_audio_files)
        )
        complete_template = complete_template.replace("${story_list}", story_list)
        templates["complete_template"] = complete_template

    return templates


def generate_shopify_csv(
    product_config: dict,
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
    output_dir: str = "../outputs/shopify",
    free_individual_count: int = 2,
) -> str:
    """
    Generate comprehensive Shopify CSV file for flashcard products with multiple images per product.

    Args:
        product_config: Dict mapping product types ('individual', 'bundle', 'complete') to their configs.
                       Each config should include 'price', 'template', and optionally 'indices' or 'ranges'.
        collection: Collection name (default: "LM1000")
        bucket_name: GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)
        output_dir: Output directory for CSV file
        free_individual_count: Number of individual packs to make free (default: 2, set to 0 for none)

    Returns:
        str: Path to generated CSV file
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Load data from GCS
    collection_path = get_story_collection_path(collection)
    story_index_path = get_story_index_path(collection)

    collection_title = get_collection_title(collection)

    collection_data = read_from_gcs(bucket_name, collection_path, "json")
    story_index = read_from_gcs(bucket_name, story_index_path, "json")

    all_stories = list(collection_data.keys())
    total_phrases = calculate_vocab_stats(all_stories, story_index, collection_data)[2]
    total_audio_files = len(all_stories) * 6
    verb_count = calculate_vocab_stats(all_stories, story_index, collection_data)[0]

    # Prepare stories for templates
    stories = []
    for story_name in all_stories:
        position = get_story_position(story_name, collection)
        title = get_story_title(story_name)
        theme = story_index.get("story_themes", {}).get(story_name, "General")
        phrase_count = len(collection_data[story_name])
        sample_phrases = get_sample_phrases(story_name, collection_data)
        stories.append(
            {
                "position": position,
                "title": title,
                "theme": theme,
                "phrase_count": phrase_count,
                "sample_phrases": sample_phrases,
                "name": story_name,
            }
        )

    # Create product templates
    templates = create_product_templates(
        collection,
        config.TARGET_LANGUAGE_NAME.lower(),
        stories,
        product_config,
        total_phrases,
        total_audio_files,
        verb_count,
        collection_title,
    )

    # Prepare CSV data
    csv_data = []

    # Language and source info
    target_language = config.TARGET_LANGUAGE_NAME
    source_language = config.SOURCE_LANGUAGE_NAME

    # Shopify CDN base URL
    shopify_cdn_base = "https://cdn.shopify.com/s/files/1/0925/9630/6250/files/"

    def add_product_with_images(base_product: dict, product_type: str, **image_kwargs):
        """Helper function to add a product with multiple images to csv_data."""
        # Get all image paths for this product type
        image_paths = []

        # Get main product images (3 images)
        main_image_path = get_marketing_image_path(
            product_type=product_type,
            collection=collection,
            language=target_language.lower(),
            **image_kwargs,
        )
        # Extract just the filename from the path
        main_image_filename = os.path.basename(main_image_path)
        image_paths.append(main_image_filename)

        # Get templates image
        templates_image_path = get_marketing_image_path(
            product_type="templates",
            collection=collection,
            language=target_language.lower(),
        )
        # Extract just the filename from the path
        templates_image_filename = os.path.basename(templates_image_path)
        image_paths.append(templates_image_filename)

        # Add product with images
        for i, image_filename in enumerate(image_paths):
            if i == 0:
                # First row: include all product details
                product = base_product.copy()
                product["Image Src"] = shopify_cdn_base + image_filename
                product["Image Position"] = i + 1
                csv_data.append(product)
            else:
                # Subsequent rows: only Handle, Image Src, and Image Position
                image_row = {
                    "Handle": base_product["Handle"],
                    "Image Src": shopify_cdn_base + image_filename,
                    "Image Position": i + 1
                }
                csv_data.append(image_row)

    # Add individual products
    if "individual" in product_config:
        individual_config = product_config["individual"]
        individual_price = individual_config["price"]
        individual_indices = individual_config.get("indices", [])
        for story in stories:
            if story["position"] in individual_indices:
                price = (
                    0.0
                    if story["position"] <= free_individual_count
                    else individual_price
                )
                base_product = {
                    "Handle": f"{target_language.lower()}-{collection.lower()}-story-{story['position']:02d}",
                    "Title": f"{target_language} - {collection_title} - Story {story['position']:02d}: {story['title']}",
                    "Body (HTML)": templates["individual_templates"][
                        f"story_{story['position']}"
                    ],
                    "Vendor": "FirePhrase",
                    "Product Category": "Toys & Games > Toys > Educational Toys > Educational Flash Cards",
                    "Type": "Digital Flashcards",
                    "Tags": f"{target_language}, {source_language}, {collection_title}, Digital Download, Language Learning",
                    "Published": "TRUE",
                    "Option1 Name": "Format",
                    "Option1 Value": "Digital Download",
                    "Variant Price": str(price),
                    "Variant Requires Shipping": "FALSE",
                    "Variant Taxable": "TRUE",
                    "source language (product.metafields.custom.source_language)": source_language,
                    "target language (product.metafields.custom.target_language)": target_language,
                    "pack type (product.metafields.custom.pack_type)": "Single",
                }
                add_product_with_images(
                    base_product, "individual", story_name=story["name"]
                )

    # Add bundle products
    if "bundle" in product_config:
        bundle_config = product_config["bundle"]
        bundle_price = bundle_config["price"]
        bundle_ranges = bundle_config.get("ranges", [])
        for start, end in bundle_ranges:
            bundle_stories = [s for s in stories if start <= s["position"] <= end]
            if not bundle_stories:
                continue
            bundle_key = f"bundle_{start}_{end}"
            if bundle_key in templates["bundle_templates"]:
                base_product = {
                    "Handle": f"{target_language.lower()}-{collection.lower()}-bundle-{start:02d}-{end:02d}",
                    "Title": f"{target_language} - {collection_title} - Bundle Pack (Stories {start:02d}-{end:02d})",
                    "Body (HTML)": templates["bundle_templates"][bundle_key],
                    "Vendor": "FirePhrase",
                    "Product Category": "Toys & Games > Toys > Educational Toys > Educational Flash Cards",
                    "Type": "Digital Flashcards",
                    "Tags": f"{target_language}, {source_language}, {collection_title}, Bundle, Digital Download, Language Learning",
                    "Published": "TRUE",
                    "Option1 Name": "Format",
                    "Option1 Value": "Digital Download",
                    "Variant Price": str(bundle_price),
                    "Variant Requires Shipping": "FALSE",
                    "Variant Taxable": "TRUE",
                    "source language (product.metafields.custom.source_language)": source_language,
                    "target language (product.metafields.custom.target_language)": target_language,
                    "pack type (product.metafields.custom.pack_type)": "Bundle",
                }
                add_product_with_images(
                    base_product, "bundle", bundle_range=f"{start:02d}-{end:02d}"
                )

    # Add complete pack product
    if "complete" in product_config:
        complete_config = product_config["complete"]
        complete_price = complete_config["price"]
        base_product = {
            "Handle": f"{target_language.lower()}-{collection.lower()}-complete-pack",
            "Title": f"{target_language} - {collection_title} - Complete Pack (All {len(all_stories)} Stories)",
            "Body (HTML)": templates["complete_template"],
            "Vendor": "FirePhrase",
            "Product Category": "Toys & Games > Toys > Educational Toys > Educational Flash Cards",
            "Type": "Digital Flashcards",
            "Tags": f"{target_language}, {source_language}, {collection_title}, Complete, Bundle, Digital Download, Language Learning",
            "Published": "TRUE",
            "Option1 Name": "Format",
            "Option1 Value": "Digital Download",
            "Variant Price": str(complete_price),
            "Variant Requires Shipping": "FALSE",
            "Variant Taxable": "TRUE",
            "source language (product.metafields.custom.source_language)": source_language,
            "target language (product.metafields.custom.target_language)": target_language,
            "pack type (product.metafields.custom.pack_type)": "Complete",
        }
        add_product_with_images(base_product, "complete")

    # Write CSV file
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir, f"{target_language.lower()}_{collection.lower()}_shopify.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

    return csv_path


def generate_product_images(
    product_config: Dict,
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
    generate_individual: bool = True,
) -> Dict[str, str]:
    """
    Generate product images for individual packs, bundle packs, and complete pack.

    Args:
        product_config: Dict mapping product types ('individual', 'bundle', 'complete') to their configs.
                       Each config should include 'template' and optionally 'indices' or 'ranges'.
        collection: Collection name (default: "LM1000")
        bucket_name: GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)
        generate_individual: Whether to generate individual pack images (default: True)

    Returns:
        Dict mapping product types to their image paths
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Get collection data to find all stories
    collection_path = get_story_collection_path(collection)
    collection_data = read_from_gcs(bucket_name, collection_path, "json")
    all_stories = list(collection_data.keys())

    language = config.TARGET_LANGUAGE_NAME.lower()
    target_language_name = config.TARGET_LANGUAGE_NAME

    # Track created images
    created_images = {}

    print(f"Generating product images for {collection} in {target_language_name}")
    print(f"Found {len(all_stories)} stories total")

    # 0. Generate Template Types Image
    print("\n=== Generating Template Types Image ===")
    # Get a sample phrase from the first story
    first_story = list(collection_data.keys())[0]
    sample_phrase = collection_data[first_story][0]["phrase"]
    phrase_key = clean_filename(sample_phrase)

    # Create temp directory for template types
    temp_base_dir = "../outputs/temp"
    temp_dir = os.path.join(temp_base_dir, f"temp_template_types_{phrase_key}")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate HTML and PNG files for each template type
    png_files = []
    template_types = ["reading_front", "listening_front", "speaking_front"]

    # Generate HTML for all template types
    generate_test_html(
        phrase_key=phrase_key,
        output_dir=temp_dir,
        collection=collection,
        bucket_name=bucket_name,
    )

    # Create PNGs from the generated HTML files
    for template_type in template_types:
        html_path = os.path.join(temp_dir, f"{template_type}.html")
        png_path = os.path.join(temp_dir, f"{template_type}.png")
        create_png_of_html(html_path, png_path, width=375, height=1100)
        png_files.append(png_path)

    # Create the template types spread deck image
    temp_output = os.path.join(temp_base_dir, "temp_template_types_spread.png")
    create_spread_deck_image(
        png_files=png_files,
        output_path=temp_output,
        angle_offset=24.0,
        x_offset=120.0,
        background_color="#FFFFFF",
    )

    # Upload template types image to GCS
    gcs_file_path = get_marketing_image_path("templates", collection, language)
    with open(temp_output, "rb") as f:
        image_data = f.read()

    template_types_uri = upload_to_gcs(
        obj=image_data,
        bucket_name=bucket_name,
        file_name=gcs_file_path,
        content_type="image/png",
    )

    created_images["template_types"] = template_types_uri
    print(f"✅ Template types image: {template_types_uri}")

    # Clean up template types temp files
    os.remove(temp_output)

    # 1. Generate Complete Pack Spread Deck Image
    if "complete" in product_config:
        print("\n=== Generating Complete Pack Spread Deck Image ===")
        spread_deck_uri = generate_spread_deck_image(
            story_positions=None,
            collection=collection,
            bucket_name=bucket_name,
            product_type="complete",
            num_phrases=12,
            angle_offset=8,
            x_offset=40,
        )
        marketing_path = get_marketing_image_path(
            product_type="complete",
            collection=collection,
            language=language,
        )
        created_images["complete_spread_deck"] = marketing_path

    # 2. Generate Bundle Pack Spread Deck Images
    if "bundle" in product_config:
        print("\n=== Generating Bundle Pack Spread Deck Images ===")
        bundle_ranges = product_config["bundle"].get("ranges", [])
        for start_pos, end_pos in bundle_ranges:
            bundle_stories = [
                story
                for story in all_stories
                if start_pos <= get_story_position(story, collection) <= end_pos
            ]
            if not bundle_stories:
                print(f"No stories found for bundle {start_pos}-{end_pos}")
                continue
            story_positions = [
                get_story_position(story, collection) for story in bundle_stories
            ]
            spread_deck_uri = generate_spread_deck_image(
                story_positions=story_positions,
                collection=collection,
                bucket_name=bucket_name,
                product_type="bundle",
                num_phrases=5,
                angle_offset=12,
                x_offset=15,
            )
            bundle_range = f"{start_pos:02d}-{end_pos:02d}"
            marketing_path = get_marketing_image_path(
                product_type="bundle",
                collection=collection,
                language=language,
                bundle_range=bundle_range,
            )
            created_images[f"bundle_{bundle_range}_spread_deck"] = marketing_path

    # 3. Generate Individual Pack Spread Deck Images
    if generate_individual and "individual" in product_config:
        print("\n=== Generating Individual Pack Spread Deck Images ===")
        individual_indices = product_config["individual"].get("indices", [])
        for story in tqdm(all_stories, desc="Generating individual spread deck images"):
            try:
                story_position = get_story_position(story, collection)
                if story_position not in individual_indices:
                    continue
                spread_deck_uri = generate_spread_deck_image(
                    story_positions=[story_position],
                    collection=collection,
                    bucket_name=bucket_name,
                    product_type="individual",
                    num_phrases=3,
                    angle_offset=12,
                    x_offset=30,
                )

                marketing_path = get_marketing_image_path(
                    product_type="individual",
                    collection=collection,
                    language=language,
                    story_name=story,
                )
                created_images[f"individual_{story}_spread_deck"] = marketing_path
            except ValueError as e:
                print(f"Skipping story {story}: {e}")
                continue

    print(f"\n🎉 Generated {len(created_images)} spread deck images successfully!")

    return created_images


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
    x_offset = abs(x_offset) * -1
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
    product_type: Optional[str] = None,
    template_type: str = "reading_back",
) -> str:
    """
    Generate spread deck images from phrases in specified stories or collection and upload to GCS.
    Updated to use consistent marketing image naming convention.

    Args:
        story_positions: Optional list of story positions to get phrases from (e.g. [1, 2, 3] for first three stories)
                        If None, samples from entire collection
        collection: Collection name (default: "LM1000")
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)
        num_phrases: Number of phrases to include in the spread (default: 5)
        angle_offset: Angle in degrees between each card (default: 6.0)
        x_offset: Horizontal offset between cards in pixels (default: 0.0)
        background_color: Color of the background (default: "#FFFFFF")
        product_type: Product type ("complete", "bundle", "individual") - auto-detected if None
        template_type: Type of template to use ("reading_back", "reading_front", "listening_front", "speaking_front")

    Returns:
        GCS URI of the uploaded spread deck image
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Create temp directory
    temp_base_dir = "../outputs/temp"
    os.makedirs(temp_base_dir, exist_ok=True)

    x_offset = abs(x_offset) * -1
    # Get collection data
    collection_path = get_story_collection_path(collection)
    collection_data = read_from_gcs(bucket_name, collection_path, "json")
    language = config.TARGET_LANGUAGE_NAME.lower()
    random.seed(language + collection)  # Default seed for multi-story
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
        temp_dir = os.path.join(temp_base_dir, language, f"temp_{phrase_key}")
        html_path = os.path.join(temp_dir, f"{template_type}.html")
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

    # Auto-detect product type if not provided
    if product_type is None:
        if story_positions is None:
            product_type = "complete"
        elif len(story_positions) == 1:
            product_type = "individual"
        else:
            product_type = "bundle"

    # Generate filename using get_marketing_image_paths logic
    language = config.TARGET_LANGUAGE_NAME.lower()

    if product_type == "complete":
        filename = f"{language}_{collection}_complete_pack.png"
    elif product_type == "bundle":
        range_str = f"{min(story_positions):02d}-{max(story_positions):02d}"
        filename = f"{language}_{collection}_bundle_{range_str}.png"
    elif product_type == "templates":
        filename = f"{language}_{collection}_template_types.png"
    else:  # individual
        story_name = selected_stories[0] if len(story_positions) == 1 else None
        if story_name:
            filename = f"{language}_{collection}_individual_{story_name}.png"
        else:
            filename = f"{language}_{collection}_individual_pack.png"

    # Upload to GCS using marketing images path

    # Get the marketing image path (first one contains the file we're creating)
    bundle_range = None
    story_name = None

    if product_type == "bundle":
        bundle_range = f"{min(story_positions):02d}-{max(story_positions):02d}"
    elif product_type == "individual" and len(story_positions) == 1:
        story_name = selected_stories[0]

    # Get the first marketing image path (this is the one we're creating)
    marketing_image_path = get_marketing_image_path(
        product_type=product_type,
        collection=collection,
        language=language,
        bundle_range=bundle_range,
        story_name=story_name,
    )

    # The first path is the file we're creating
    gcs_file_path = marketing_image_path

    # Read the image file
    with open(temp_output, "rb") as f:
        image_data = f.read()

    # Upload to GCS
    gcs_uri = upload_to_gcs(
        obj=image_data,
        bucket_name=bucket_name,
        file_name=gcs_file_path,
        content_type="image/png",
    )

    # Clean up only the final spread deck image
    os.remove(temp_output)

    return gcs_uri
