import csv
import math
import os
import random
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

from PIL import Image

from src.config_loader import config
from src.convert import clean_filename, get_story_title
from src.gcs_storage import (
    get_marketing_image_paths,
    get_phrase_path,
    get_story_collection_path,
    get_story_index_path,
    read_from_gcs,
    upload_to_gcs,
)
from src.images import create_png_of_html
from src.template_testing import generate_test_html
from src.utils import get_story_position


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


def generate_story_list_html(story_names: List[str], collection_data: Dict) -> str:
    """Generate HTML list of stories with their positions and titles."""
    story_items = []
    for story in story_names:
        try:
            position = get_story_position(story, collection_data)
            title = get_story_title(story)
            story_items.append(f'<li>Story {position:02d}: "{title}"</li>')
        except ValueError:
            continue

    return "<ul>" + "\n".join(story_items) + "</ul>"


def create_product_templates():
    """Create HTML templates for different product types."""

    # Complete Pack Template
    complete_template = Template(
        """
<p><strong>Complete ${collection} Vocabulary System (All ${total_stories} Stories)</strong></p>
<p>Master the most essential ${vocab_count}+ words (${verb_count}+ verbs, ${vocab_count}+ vocabulary) through our comprehensive, story-based learning system. This complete collection provides a structured path to vocabulary acquisition through ${total_stories} engaging stories, carefully sequenced to optimise retention.</p>

<h3>The Complete System Includes:</h3>
<ul>
<li>All ${total_stories} story-based vocabulary packs</li>
<li>${total_phrases}+ carefully crafted flashcards</li>
<li>${audio_files} M4A audio files organized in story albums:
<ul>
<li>Regular-speed story parts</li>
<li>Fast-mode versions for advanced listening skill development</li>
<li>Works with any standard audio player</li>
<li>Features embedded lyrics that display in the free Oto Music Player app, allowing you to read while listening</li>
</ul>
</li>
<li>High-quality AI-generated audio (human-curated for accuracy)</li>
<li>Access to complementary online companion stories
<ul>
<li>Experimental and unsupported Immersive AI chat features for each story (for advanced technical users with OpenAI API key)</li>
</ul>
</li>
<li>Systematic coverage of the most frequent 1000 words</li>
<li>Visual memory aids and reference links for deeper learning</li>
</ul>

<h3>Your Language Learning Journey:</h3>
${story_list}
<p>Experience words 'just sticking' in your memory with the vivid images and audio. Complete a deck of flashcards, then consolidate by listening to the final story. Each story reinforces previously learned words while introducing new vocabulary in memorable contexts, so it's important to do the stories in order.</p>
<p>With 20 - 30 minutes a day expect to do a story every 5 - 7 days. You will average about 2 new words per flashcard, although initially this is higher as all words might be new, later it is lower. Learning therefore gets easier as the stories progress.</p>
<p>With just 15 - 30 minutes of daily practice, complete this system in 3-4 months and dramatically expand your vocabulary foundation.</p>
<p><strong>Save ${savings_percent}% compared to purchasing individual packs!</strong></p>
"""
    )

    # Bundle Template
    bundle_template = Template(
        """
<p><strong>${collection} Vocabulary Bundle (Stories ${range_display})</strong></p>
<p>Continue your journey to language fluency with this carefully sequenced collection of story-based vocabulary packs. This bundle includes stories ${range_display} from our ${collection} series, designed to systematically build your mastery of the most common 1000 words.</p>

<h3>Bundle Contents:</h3>
${story_list}

<h3>Progressive Learning System:</h3>
<p>Each story builds on vocabulary from previous packs, creating a structured learning path. By completing these ${story_count} packs, you'll master approximately ${vocab_count}+ high-frequency words in memorable context.</p>

<h3>What's Included:</h3>
<ul>
<li>${total_phrases}+ carefully designed digital flashcards (35 per story)</li>
<li>${audio_files} M4A audio files organized in story albums:
<ul>
<li>Regular-speed story parts for each story</li>
<li>Fast-mode versions for advanced listening skill development</li>
<li>Works with any standard audio player</li>
<li>Features embedded lyrics that display in the free Oto Music Player app, allowing you to read while listening</li>
</ul>
</li>
<li>High-quality AI-generated audio (human-curated for accuracy)</li>
<li>Access to online companion stories for each pack</li>
<li>Immersive AI chat features for technically adept users (requires OpenAI API key)</li>
<li>Visual memory aids and Wiktionary links</li>
<li>Save ${savings_percent}% compared to purchasing individually</li>
</ul>
"""
    )

    # Individual Template
    individual_template = Template(
        """
<p><strong>"${story_title}" - Story #${story_position:02d} | ${collection} Vocabulary Series</strong></p>
<p>Part of our systematic approach to mastering the most common 1000 words through engaging story contexts. This pack continues your vocabulary journey with carefully crafted phrases designed to reinforce previously learned words while introducing new ones.</p>

<h3>About This Pack:</h3>
<ul>
<li>${phrase_count} natural, memorable phrases in this ${story_theme} story</li>
<li>Sample phrases include:</li>
<ul>
${sample_phrases_html}
</ul>
<li>Focuses on high-frequency vocabulary from the ${collection} word list</li>
<li>Builds on vocabulary introduced in earlier story packs</li>
</ul>

<h3>What's Included:</h3>
<ul>
<li>${phrase_count} carefully designed digital flashcards</li>
<li>6 M4A audio files organized as an album:
<ul>
<li>3 regular-speed story parts</li>
<li>3 fast-mode versions for advanced listening skill development</li>
<li>Works with any standard audio player</li>
<li>Features embedded lyrics that display in the free Oto Music Player app, allowing you to read while listening</li>
</ul>
</li>
<li>High-quality AI-generated audio (human-curated for accuracy)</li>
<li>Visual memory aids for deeper retention</li>
<li>Wiktionary links for additional context</li>
<li>Optimized for Anki's spaced repetition system</li>
</ul>

<h3>Online Companion Resources:</h3>
<ul>
<li>Access to online companion story</li>
<li>Immersive AI chat feature:
<ul>
<li>Step into the story and tackle common language challenges related to the narrative</li>
<li>Practice speaking in realistic scenarios matching the story context</li>
<li>For advanced technical users comfortable with API integration</li>
<li>Requires your own OpenAI account</li>
<li>Provided as-is for those wanting to experiment with AI language interactions</li>
</ul>
</li>
</ul>

<h3>Progressive Learning System:</h3>
<p>Our research-based approach introduces vocabulary in an optimal sequence. While each pack can be used independently, maximum benefit comes from progressing through the series in order, as later stories build upon vocabulary introduced in earlier ones.</p>
<p>Just 10-15 minutes daily practice will dramatically improve your vocabulary retention. Our phrase-based method creates meaningful connections between words, helping you remember them naturally in context.</p>

<h3>Technical Requirements:</h3>
<ul>
<li>Requires the free Anki application</li>
<li>M4A files work with any standard media player</li>
<li>For synchronized lyrics display, download the free Oto Music Player app</li>
<li>AI chat feature requires an OpenAI API key (for technically advanced users)</li>
</ul>
"""
    )

    return {
        "complete": complete_template,
        "bundle": bundle_template,
        "individual": individual_template,
    }


def generate_shopify_csv(
    bundle_config: Dict[str, List[int]] = None,
    prices: Dict[str, float] = None,
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
    output_dir: str = "../outputs/shopify",
) -> str:
    """
    Generate comprehensive Shopify CSV file for flashcard products.

    Args:
        bundle_config: Dict mapping bundle names to story position ranges
                      e.g., {"Bundle 01-08": [1, 8], "Bundle 09-14": [9, 14]}
        prices: Dict with pricing for each product type
        collection: Collection name (default: "LM1000")
        bucket_name: GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)
        output_dir: Output directory for CSV file

    Returns:
        str: Path to generated CSV file
    """
    # Default configurations
    if bundle_config is None:
        bundle_config = {
            "Bundle 01-08": [1, 8],
            "Bundle 09-14": [9, 14],
            "Bundle 15-20": [15, 20],
        }

    if prices is None:
        prices = {"individual": 1.99, "bundle": 7.99, "complete": 19.99}

    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Load data from GCS
    collection_path = get_story_collection_path(collection)
    story_index_path = get_story_index_path(collection)

    collection_data = read_from_gcs(bucket_name, collection_path, "json")
    story_index = read_from_gcs(bucket_name, story_index_path, "json")

    all_stories = list(collection_data.keys())
    templates = create_product_templates()

    # Prepare CSV data
    csv_data = []

    # Language and source info
    target_language = config.TARGET_LANGUAGE_NAME
    source_language = config.SOURCE_LANGUAGE_NAME

    # 1. Generate Complete Pack
    verb_count, vocab_count, total_phrases = calculate_vocab_stats(
        all_stories, story_index, collection_data
    )

    story_list_html = generate_story_list_html(all_stories, collection_data)
    audio_files = len(all_stories) * 6  # 6 files per story
    savings_percent = round(
        (1 - prices["complete"] / (len(all_stories) * prices["individual"])) * 100
    )

    complete_description = templates["complete"].substitute(
        collection=collection,
        total_stories=len(all_stories),
        vocab_count=vocab_count,
        verb_count=verb_count,
        total_phrases=total_phrases,
        audio_files=audio_files,
        story_list=story_list_html,
        savings_percent=savings_percent,
    )

    complete_images = get_marketing_image_paths(
        "complete", collection, target_language.lower()
    )

    complete_product = {
        "Handle": f"{target_language.lower()}-{collection.lower()}-complete-pack",
        "Title": f"{target_language} - {collection} - Complete Pack (All {len(all_stories)} Stories)",
        "Body (HTML)": complete_description,
        "Vendor": "FirePhrase",
        "Product Category": "Education > Language Learning",
        "Type": "Digital Flashcards",
        "Tags": f"{target_language}, {source_language}, {collection}, Complete, Bundle, Digital Download, Language Learning",
        "Published": "TRUE",
        "Option1 Name": "Format",
        "Option1 Value": "Digital Download",
        "Variant Price": prices["complete"],
        "Variant Requires Shipping": "FALSE",
        "Variant Taxable": "TRUE",
        "Image Src": complete_images[0],
        "Image Position": 1,
        "source language (product.metafields.custom.source_language)": source_language,
        "target language (product.metafields.custom.target_language)": target_language,
        "Status": "active",
    }
    csv_data.append(complete_product)

    # Add additional images for complete pack
    for i, img_url in enumerate(complete_images[1:], 2):
        csv_data.append(
            {
                "Handle": complete_product["Handle"],
                "Image Src": img_url,
                "Image Position": i,
                **{
                    k: ""
                    for k in complete_product.keys()
                    if k not in ["Handle", "Image Src", "Image Position"]
                },
            }
        )

    # 2. Generate Bundle Packs
    for bundle_name, (start_pos, end_pos) in bundle_config.items():
        bundle_stories = [
            story
            for i, story in enumerate(all_stories)
            if start_pos <= i + 1 <= end_pos
        ]

        if not bundle_stories:
            continue

        verb_count, vocab_count, total_phrases = calculate_vocab_stats(
            bundle_stories, story_index, collection_data
        )

        story_list_html = generate_story_list_html(bundle_stories, collection_data)
        audio_files = len(bundle_stories) * 6
        range_display = f"{start_pos:02d}-{end_pos:02d}"
        savings_percent = round(
            (1 - prices["bundle"] / (len(bundle_stories) * prices["individual"])) * 100
        )

        bundle_description = templates["bundle"].substitute(
            collection=collection,
            range_display=range_display,
            story_count=len(bundle_stories),
            vocab_count=vocab_count,
            verb_count=verb_count,
            total_phrases=total_phrases,
            audio_files=audio_files,
            story_list=story_list_html,
            savings_percent=savings_percent,
        )

        bundle_images = get_marketing_image_paths(
            "bundle", collection, target_language.lower(), range_display
        )

        bundle_product = {
            "Handle": f"{target_language.lower()}-{collection.lower()}-{bundle_name.lower().replace(' ', '-')}",
            "Title": f"{target_language} - {collection} - {bundle_name}",
            "Body (HTML)": bundle_description,
            "Vendor": "FirePhrase",
            "Product Category": "Education > Language Learning",
            "Type": "Digital Flashcards",
            "Tags": f"{target_language}, {source_language}, {collection}, Bundle, Digital Download, Language Learning",
            "Published": "TRUE",
            "Option1 Name": "Format",
            "Option1 Value": "Digital Download",
            "Variant Price": prices["bundle"],
            "Variant Requires Shipping": "FALSE",
            "Variant Taxable": "TRUE",
            "Image Src": bundle_images[0],
            "Image Position": 1,
            "source language (product.metafields.custom.source_language)": source_language,
            "target language (product.metafields.custom.target_language)": target_language,
            "Status": "active",
        }
        csv_data.append(bundle_product)

        # Add additional images for bundle
        for i, img_url in enumerate(bundle_images[1:], 2):
            csv_data.append(
                {
                    "Handle": bundle_product["Handle"],
                    "Image Src": img_url,
                    "Image Position": i,
                    **{
                        k: ""
                        for k in bundle_product.keys()
                        if k not in ["Handle", "Image Src", "Image Position"]
                    },
                }
            )

    # 3. Generate Individual Packs
    for story in all_stories:
        try:
            position = get_story_position(story, collection_data)
            story_title = get_story_title(story)
            phrase_count = len(collection_data[story])

            sample_phrases = get_sample_phrases(story, collection_data, 5)
            sample_phrases_html = "\n".join(
                [f'<li>"{phrase}"</li>' for phrase in sample_phrases]
            )

            # Determine story theme (simplified)
            story_theme = (
                "engaging"  # Could be enhanced to detect theme from story content
            )

            individual_description = templates["individual"].substitute(
                story_title=story_title,
                story_position=position,
                collection=collection,
                phrase_count=phrase_count,
                story_theme=story_theme,
                sample_phrases_html=sample_phrases_html,
            )

            individual_images = get_marketing_image_paths(
                "individual", collection, target_language.lower(), story_name=story
            )

            individual_product = {
                "Handle": f"{target_language.lower()}-{collection.lower()}-story-{position:02d}-{story.replace('story_', '').replace('_', '-')}",
                "Title": f"{target_language} - {collection} - Story {position:02d}: {story_title}",
                "Body (HTML)": individual_description,
                "Vendor": "FirePhrase",
                "Product Category": "Education > Language Learning",
                "Type": "Digital Flashcards",
                "Tags": f"{target_language}, {source_language}, {collection}, Individual, Digital Download, Language Learning",
                "Published": "TRUE",
                "Option1 Name": "Format",
                "Option1 Value": "Digital Download",
                "Variant Price": prices["individual"],
                "Variant Requires Shipping": "FALSE",
                "Variant Taxable": "TRUE",
                "Image Src": individual_images[0],
                "Image Position": 1,
                "source language (product.metafields.custom.source_language)": source_language,
                "target language (product.metafields.custom.target_language)": target_language,
                "Status": "active",
            }
            csv_data.append(individual_product)

            # Add additional images for individual pack
            for i, img_url in enumerate(individual_images[1:], 2):
                csv_data.append(
                    {
                        "Handle": individual_product["Handle"],
                        "Image Src": img_url,
                        "Image Position": i,
                        **{
                            k: ""
                            for k in individual_product.keys()
                            if k not in ["Handle", "Image Src", "Image Position"]
                        },
                    }
                )

        except ValueError:
            print(f"Skipping story {story} - not found in collection")
            continue

    # Write CSV file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"{target_language.lower()}_{collection.lower()}_shopify_products.csv",
    )

    # Define fieldnames based on the required columns
    fieldnames = [
        "Handle",
        "Title",
        "Body (HTML)",
        "Vendor",
        "Product Category",
        "Type",
        "Tags",
        "Published",
        "Option1 Name",
        "Option1 Value",
        "Variant Price",
        "Variant Requires Shipping",
        "Variant Taxable",
        "Image Src",
        "Image Position",
        "source language (product.metafields.custom.source_language)",
        "target language (product.metafields.custom.target_language)",
        "Status",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in csv_data:
            # Only write fields that exist in fieldnames
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)

    print(f"Generated Shopify CSV with {len(csv_data)} entries: {output_file}")
    return output_file


def generate_product_images(
    collection: str = "LM1000",
    bundle_config: Dict[str, List[int]] = None,
    bucket_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate all required product images for Shopify listings.

    Args:
        collection: Collection name (default: "LM1000")
        bundle_config: Dict mapping bundle names to story position ranges
                      e.g., {"Bundle 01-08": [1, 8], "Bundle 09-14": [9, 14]}
        bucket_name: GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)

    Returns:
        Dict mapping product types to their generated image URIs
    """
    if bundle_config is None:
        bundle_config = {
            "Bundle 01-08": [1, 8],
            "Bundle 09-14": [9, 14],
            "Bundle 15-20": [15, 20],
        }

    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    generated_images = {}

    # 1. Generate Complete Pack Image
    # More cards, smaller angles and offsets for cleaner look
    print("Generating complete pack image...")
    complete_uri = generate_spread_deck_image(
        story_positions=None,  # Uses entire collection
        collection=collection,
        bucket_name=bucket_name,
        num_phrases=12,
        angle_offset=6.0,  # Smaller angle for many cards
        x_offset=20.0,  # Minimal offset for tight spread
        background_color="#FFFFFF",
        product_type="complete",
    )
    generated_images["complete"] = complete_uri
    print(f"✅ Complete pack image: {complete_uri}")

    # 2. Generate Bundle Images
    # Medium number of cards, moderate angles and offsets
    for bundle_name, (start_pos, end_pos) in bundle_config.items():
        print(f"Generating bundle image for {bundle_name}...")
        bundle_positions = list(range(start_pos, end_pos + 1))

        bundle_uri = generate_spread_deck_image(
            story_positions=bundle_positions,
            collection=collection,
            bucket_name=bucket_name,
            num_phrases=5,
            angle_offset=12.0,  # Medium angle for moderate spread
            x_offset=15.0,  # Medium offset
            background_color="#FFFFFF",
            product_type="bundle",
        )
        generated_images[f"bundle_{start_pos:02d}_{end_pos:02d}"] = bundle_uri
        print(f"✅ Bundle {bundle_name} image: {bundle_uri}")

    # 3. Generate Individual Pack Images (One for each story)
    # Fewer cards, larger angles and offsets for dramatic effect
    print("Generating individual pack images...")

    # Get collection data to iterate through stories
    from src.gcs_storage import get_story_collection_path, read_from_gcs
    from src.utils import get_story_position

    collection_path = get_story_collection_path(collection)
    collection_data = read_from_gcs(bucket_name, collection_path, "json")
    all_stories = list(collection_data.keys())

    for story in all_stories:
        try:
            position = get_story_position(story, collection_data)
            print(f"Generating individual image for story {position:02d}: {story}")

            # Create spread using phrases from this specific story (single position)
            individual_uri = generate_spread_deck_image(
                story_positions=[position],  # Single story position
                collection=collection,
                bucket_name=bucket_name,
                num_phrases=3,
                angle_offset=12.0,  # Larger angle for dramatic fan effect
                x_offset=30.0,  # Larger offset for wider spread
                background_color="#FFFFFF",
                product_type="individual",
            )

            generated_images[f"individual_{story}"] = individual_uri
            print(f"✅ Individual pack image for {story}: {individual_uri}")

        except ValueError as e:
            print(f"⚠️ Skipping story {story}: {e}")
            continue

    print(f"\n🎉 Generated {len(generated_images)} product images successfully!")
    return generated_images


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
    product_type: Optional[str] = None,
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

        # For single story, use story name as seed for consistent sampling
        if len(story_positions) == 1:
            import random

            random.seed(hash(selected_stories[0]) % (2**32))
        else:
            import random

            random.seed(12)  # Default seed for multi-story
    else:
        # Get all phrases from the collection
        phrases_path = get_phrase_path(collection)
        story_phrases = read_from_gcs(bucket_name, phrases_path, "json")
        import random

        random.seed(12)  # Default seed

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
    else:  # individual
        story_name = selected_stories[0] if len(story_positions) == 1 else None
        if story_name:
            filename = f"{language}_{collection}_individual_{story_name}.png"
        else:
            filename = f"{language}_{collection}_individual_pack.png"

    # Upload to GCS using marketing images path
    from src.gcs_storage import upload_to_gcs

    # Get the marketing image path (first one contains the file we're creating)
    bundle_range = None
    story_name = None

    if product_type == "bundle":
        bundle_range = f"{min(story_positions):02d}-{max(story_positions):02d}"
    elif product_type == "individual" and len(story_positions) == 1:
        story_name = selected_stories[0]

    # Get the first marketing image path (this is the one we're creating)
    marketing_image_paths = get_marketing_image_paths(
        product_type=product_type,
        collection=collection,
        language=language,
        bundle_range=bundle_range,
        story_name=story_name,
    )

    # The first path is the file we're creating
    gcs_file_path = marketing_image_paths[0]

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
