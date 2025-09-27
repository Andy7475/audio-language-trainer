#!/usr/bin/env python3
"""
Process a collection into a new language.

This script handles the complete pipeline for processing a story collection
into a new target language, including translation, audio generation, and
web page creation.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import glob

# Add the parent directory to the path to import src modules
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.anki_tools import create_anki_deck_from_gcs
from src.audio_generation import (
    generate_and_upload_fast_audio,
    generate_dialogue_audio_and_upload,
    upload_phrases_audio_to_gcs,
)
from src.chat import create_html_challenges, get_html_challenge_inputs
from src.config_loader import config
from src.convert import clean_filename
from contextlib import contextmanager
from src.dialogue_generation import translate_and_upload_dialogue
from src.shop import generate_product_images, generate_shopify_csv
from src.zip import create_m4a_zip_collections
from src.gcs_storage import (
    check_blob_exists,
    get_stories_from_collection,
    get_story_challenges_path,
    get_story_collection_path,
    get_story_dialogue_path,
    get_story_translated_dialogue_path,
    get_translated_phrases_path,
    get_wiktionary_cache_path,
    read_from_gcs,
    upload_to_gcs,
)
from src.story import (
    create_album_files,
    create_and_upload_html_story,
    prepare_dialogue_with_wiktionary,
    prepare_story_data_from_gcs,
    update_all_index_pages_hierarchical,
    upload_styles_to_gcs,
)
from src.translation import (
    review_story_dialogue_translations,
    review_translated_phrases_batch,
    translate_phrases,
)
from src.wiktionary import add_wiktionary_links


def setup_authentication():
    """Setup Google Cloud authentication."""
    try:
        from google.auth import default

        credentials, project = default()
        print(f"✅ Authenticated with Google Cloud project: {project}")
        return credentials, project
    except Exception as e:
        print(f"❌ Failed to authenticate with Google Cloud: {e}")
        sys.exit(1)


def normalize_language_name(language_name):
    """
    Normalize language name to title case for config consistency.

    Args:
        language_name: Language name in any case (e.g., 'french', 'FRENCH', 'French')

    Returns:
        Title case language name (e.g., 'French')
    """
    return language_name.strip().title()


@contextmanager
def language_context(language_name):
    """Context manager to temporarily switch the target language."""
    if language_name is None:
        # No language switch needed
        yield
        return

    # Store original language
    original_language = config.TARGET_LANGUAGE_NAME

    try:
        # Switch to new language (normalize to title case)
        normalized_language = normalize_language_name(language_name)
        config.TARGET_LANGUAGE_NAME = normalized_language
        print(f"🌍 Switched to language: {normalized_language}")
        yield
    finally:
        # Restore original language
        config.TARGET_LANGUAGE_NAME = original_language


def print_config_info():
    """Print current configuration information."""
    print(f"Target Language: {config.TARGET_LANGUAGE_NAME}")

    try:
        english_voice, female_voice, male_voice = config.get_voice_models()
        english_voice_story, female_voice_story, male_voice_story = (
            config.get_voice_models("stories")
        )
        print(
            f"Voices - Female: {female_voice.voice_id}, Female Story: {female_voice_story.voice_id}, Male Story: {male_voice_story.voice_id}"
        )
    except Exception as e:
        print(f"Warning: Could not load voice configuration: {e}")


def translate_phrases_step(collection: str, story_limit: int = None):
    """Step 1: Translate all phrases in the collection."""
    print("\n🔄 Step 1: Translating phrases...")

    all_stories = get_stories_from_collection(collection=collection, limit=story_limit)
    story_collection = read_from_gcs(
        bucket_name=config.GCS_PRIVATE_BUCKET,
        file_path=get_story_collection_path(collection=collection),
    )
    translated_phrases_path = get_translated_phrases_path(collection=collection)
    language_name_lower = config.TARGET_LANGUAGE_NAME.lower()

    # Google translate all phrases
    results = dict()
    print(f"Processing {len(all_stories)} stories...")

    for story in tqdm(all_stories, desc="Translating story phrases"):
        # Extract just the phrases from the story's phrase list
        english_phrases = [item["phrase"] for item in story_collection[story]]
        translated_phrases = translate_phrases(english_phrases)

        for phrase, translation in translated_phrases:
            phrase_key = clean_filename(phrase)
            results[phrase_key] = {"english": phrase, language_name_lower: translation}

    # Upload initial translations
    upload_to_gcs(
        results,
        bucket_name=config.GCS_PRIVATE_BUCKET,
        file_name=translated_phrases_path,
    )
    print(f"✅ Initial translations uploaded to {translated_phrases_path}")

    return results


def refine_phrase_translations(collection: str):
    """Step 2: Refine phrase translations using Claude."""
    print("\n🔄 Step 2: Refining phrase translations...")

    translated_phrases_path = get_translated_phrases_path(collection=collection)
    phrase_translations = read_from_gcs(
        bucket_name=config.GCS_PRIVATE_BUCKET, file_path=translated_phrases_path
    )

    improved_translations = review_translated_phrases_batch(
        phrase_translations, model="claude-3-5-sonnet-latest"
    )

    upload_to_gcs(
        obj=improved_translations,
        bucket_name=config.GCS_PRIVATE_BUCKET,
        file_name=translated_phrases_path,
    )

    print("✅ Refined translations uploaded")
    return improved_translations


def add_wiktionary_links_to_phrases(collection: str):
    """Step 3: Add Wiktionary links to phrases."""
    print("\n🔄 Step 3: Adding Wiktionary links to phrases...")

    translated_phrases_path = get_translated_phrases_path(collection=collection)
    improved_translations = read_from_gcs(
        bucket_name=config.GCS_PRIVATE_BUCKET, file_path=translated_phrases_path
    )

    try:
        word_link_cache = read_from_gcs(
            config.GCS_PRIVATE_BUCKET, file_path=get_wiktionary_cache_path()
        )
    except:
        word_link_cache = {}

    phrase_translations, word_link_cache = add_wiktionary_links(
        improved_translations, word_link_cache, overwrite=False
    )

    upload_to_gcs(
        obj=phrase_translations,
        bucket_name=config.GCS_PRIVATE_BUCKET,
        file_name=translated_phrases_path,
    )
    upload_to_gcs(
        word_link_cache,
        bucket_name=config.GCS_PRIVATE_BUCKET,
        file_name=get_wiktionary_cache_path(),
    )

    print("✅ Wiktionary links added to phrases")
    return phrase_translations


def generate_phrase_audio(collection: str):
    """Step 4: Generate audio for translated phrases."""
    print("\n🔄 Step 4: Generating phrase audio...")

    translated_phrases_path = get_translated_phrases_path(collection=collection)
    phrase_translations = read_from_gcs(
        bucket_name=config.GCS_PRIVATE_BUCKET, file_path=translated_phrases_path
    )

    result = upload_phrases_audio_to_gcs(phrase_translations, overwrite=False)
    print("✅ Phrase audio generated and uploaded")
    return result


def translate_stories(collection: str, story_limit: int = None):
    """Step 5: Translate story dialogues."""
    print("\n🔄 Step 5: Translating story dialogues...")

    all_stories = get_stories_from_collection(collection=collection, limit=story_limit)

    for story_name in tqdm(all_stories, desc="Translating stories"):
        story_file_path = get_story_dialogue_path(story_name, collection=collection)
        translated_file_path = get_story_translated_dialogue_path(
            story_name, collection=collection
        )

        if check_blob_exists(config.GCS_PRIVATE_BUCKET, translated_file_path):
            print(f"  {story_name} already translated")
            continue

        story_dialogue = read_from_gcs(config.GCS_PRIVATE_BUCKET, story_file_path)
        translate_and_upload_dialogue(story_dialogue, story_name, collection=collection)
        print(f"  ✅ Translated {story_name}")

    print("✅ All story dialogues translated")


def refine_story_translations(collection: str, story_limit: int = None):
    """Step 6: Refine story dialogue translations."""
    print("\n🔄 Step 6: Refining story dialogue translations...")

    all_stories = get_stories_from_collection(collection=collection, limit=story_limit)

    for story_name in tqdm(all_stories, desc="Refining story translations"):
        translated_file_path = get_story_translated_dialogue_path(
            story_name, collection=collection
        )

        if not check_blob_exists(config.GCS_PRIVATE_BUCKET, translated_file_path):
            print(f"  ⚠️  {story_name} not yet translated, skipping")
            continue

        translated_dialogue = read_from_gcs(
            config.GCS_PRIVATE_BUCKET, translated_file_path
        )
        reviewed_dialogue = review_story_dialogue_translations(translated_dialogue)
        upload_to_gcs(
            obj=reviewed_dialogue,
            bucket_name=config.GCS_PRIVATE_BUCKET,
            file_name=translated_file_path,
        )
        print(f"  ✅ Refined {story_name}")

    print("✅ All story dialogue translations refined")


def add_wiktionary_links_to_stories(collection: str, story_limit: int = None):
    """Step 7: Add Wiktionary links to story dialogues."""
    print("\n🔄 Step 7: Adding Wiktionary links to story dialogues...")

    all_stories = get_stories_from_collection(collection=collection, limit=story_limit)

    for story_name in tqdm(all_stories, desc="Adding Wiktionary links to stories"):
        translated_file_path = get_story_translated_dialogue_path(
            story_name, collection=collection
        )

        if not check_blob_exists(config.GCS_PRIVATE_BUCKET, translated_file_path):
            print(f"  ⚠️  {story_name} not yet translated, skipping")
            continue

        translated_dialogue = read_from_gcs(
            config.GCS_PRIVATE_BUCKET, translated_file_path
        )
        translated_dialogue_with_links = prepare_dialogue_with_wiktionary(
            translated_dialogue
        )
        upload_to_gcs(
            obj=translated_dialogue_with_links,
            bucket_name=config.GCS_PRIVATE_BUCKET,
            file_name=translated_file_path,
        )
        print(f"  ✅ Added Wiktionary links to {story_name}")

    print("✅ Wiktionary links added to all stories")


def generate_story_audio(
    collection: str, overwrite: bool = False, story_limit: int = None
):
    """Step 8: Generate audio for story dialogues."""
    print(f"\n🔄 Step 8: Generating story dialogue audio (overwrite={overwrite})...")

    all_stories = get_stories_from_collection(collection=collection, limit=story_limit)

    for story_name in tqdm(all_stories, desc="Generating story audio"):
        translated_file_path = get_story_translated_dialogue_path(
            story_name, collection=collection
        )

        if not check_blob_exists(config.GCS_PRIVATE_BUCKET, translated_file_path):
            print(f"  ⚠️  {story_name} not yet translated, skipping")
            continue

        translated_dialogue = read_from_gcs(
            config.GCS_PRIVATE_BUCKET, translated_file_path
        )
        generate_dialogue_audio_and_upload(
            translated_dialogue, story_name, collection=collection, overwrite=overwrite
        )
        print(f"  ✅ Generated audio for {story_name}")

    print("✅ All story dialogue audio generated")


def generate_fast_audio(
    collection: str, overwrite: bool = False, story_limit: int = None
):
    """Step 9: Generate fast audio for stories."""
    print(f"\n🔄 Step 9: Generating fast audio (overwrite={overwrite})...")

    all_stories = get_stories_from_collection(collection=collection, limit=story_limit)

    for story_name in tqdm(all_stories, desc="Generating fast audio"):
        try:
            generate_and_upload_fast_audio(
                story_name, collection=collection, overwrite=overwrite
            )
            print(f"  ✅ Generated fast audio for {story_name}")
        except Exception as e:
            print(f"  ❌ Failed to generate fast audio for {story_name}: {e}")

    print("✅ Fast audio generation completed")


def create_challenges(collection: str, story_limit: int = None):
    """Step 10: Create challenge pages (multi-language aware)."""

    def _create_challenges_for_language(collection: str, language: str = None):
        language_info = f" for {language}" if language else ""
        print(f"\n🔄 Step 10: Creating challenge pages{language_info}...")
        with language_context(language):
            all_stories = get_stories_from_collection(
                collection=collection, limit=story_limit
            )
            for story_name in tqdm(all_stories, desc="Creating challenges"):
                try:
                    challenge_file_path = get_story_challenges_path(
                        story_name, collection=collection
                    )
                    if not check_blob_exists(
                        config.GCS_PRIVATE_BUCKET, challenge_file_path
                    ):
                        print(f"  ⚠️  No challenges found for {story_name}, skipping")
                        continue
                    scenario_dicts = read_from_gcs(
                        bucket_name=config.GCS_PRIVATE_BUCKET,
                        file_path=challenge_file_path,
                    )
                    challenges = get_html_challenge_inputs(scenario_dicts)
                    chat_webpage_file = create_html_challenges(
                        challenges, story_name=story_name, collection=collection
                    )
                    print(f"  ✅ Created challenges for {story_name}")
                except Exception as e:
                    print(f"  ❌ Failed to create challenges for {story_name}: {e}")
            print("✅ Challenge pages created" + language_info)

    return _create_challenges_for_language


def create_story_pages(collection: str, story_limit: int = None):
    """Step 11: Create story HTML pages."""
    print("\n🔄 Step 11: Creating story pages...")

    all_stories = get_stories_from_collection(collection=collection, limit=story_limit)

    for story_name in tqdm(all_stories, desc="Creating story pages"):
        try:
            print(f"  Processing {story_name}...")
            story_data = prepare_story_data_from_gcs(story_name, collection=collection)

            if not story_data:
                print(f"  ⚠️  No story data found for {story_name}, skipping")
                continue

            create_and_upload_html_story(story_data, story_name, collection=collection)
            print(f"  ✅ Created pages for {story_name}")
        except Exception as e:
            print(f"  ❌ Failed to create pages for {story_name}: {e}")

    print("✅ Story pages created")


def create_albums(collection: str, story_limit: int = None):
    """Step 12: Create album files."""
    print("\n🔄 Step 12: Creating album files...")

    all_stories = get_stories_from_collection(collection=collection, limit=story_limit)

    for story_name in tqdm(all_stories, desc="Creating albums"):
        try:
            print(f"  Processing {story_name}...")
            story_data = prepare_story_data_from_gcs(story_name, collection=collection)

            if not story_data:
                print(f"  ⚠️  No story data found for {story_name}, skipping")
                continue

            create_album_files(story_data, story_name, collection=collection)
            print(f"  ✅ Created album for {story_name}")
        except Exception as e:
            print(f"  ❌ Failed to create album for {story_name}: {e}")

    print("✅ Album files created")


def update_index_pages():
    """Step 13: Update index pages."""
    print("\n🔄 Step 13: Updating index pages...")

    languages = [
        "French",
        "Spanish",
        "German",
        "Swedish",
        "Italian",
        "Mandarin Chinese",
        "Russian",
    ]
    collections = ["LM1000", "WarmUp150", "LM2000"]

    upload_styles_to_gcs()
    result = update_all_index_pages_hierarchical(
        languages=languages, collections=collections
    )

    print("✅ Index pages updated")
    return result


def create_anki_decks(collection: str, story_limit: int = None):
    """Step 14: Create Anki decks from GCS data."""
    print("\n🔄 Step 14: Creating Anki decks...")

    try:
        create_anki_deck_from_gcs(
            collection=collection, output_dir="../outputs/gcs", story_limit=story_limit
        )
        print("✅ Anki decks created successfully")
    except Exception as e:
        print(f"❌ Failed to create Anki decks: {e}")


def load_product_config(collection: str):
    """Load product configuration from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), "product_configs.json")

    try:
        with open(config_path, "r") as f:
            all_configs = json.load(f)

        if collection not in all_configs:
            print(f"⚠️  No product config found for {collection}, using default")
            return None

        return all_configs[collection]
    except Exception as e:
        print(f"❌ Failed to load product config: {e}")
        return None


def create_zip_files(collection: str):
    """Step 15: Create M4A zip collections for products."""
    print("\n🔄 Step 15: Creating M4A zip collections...")

    product_config = load_product_config(collection)
    if not product_config:
        print("❌ Cannot create zip files without product config")
        return

    try:
        zips = create_m4a_zip_collections(
            product_config=product_config,
            collection=collection,
            bucket_name=config.GCS_PRIVATE_BUCKET,
            output_base_dir="../outputs/gcs",
            use_local_files=True,
        )
        print(f"✅ Created {len(zips)} zip files")
        for zip_type, zip_path in zips.items():
            print(f"  {zip_type}: {zip_path}")
    except Exception as e:
        print(f"❌ Failed to create zip files: {e}")


def generate_images(collection: str, language: str = None):
    """Step 16: Generate product images."""
    language_info = f" for {language}" if language else ""
    print(f"\n🔄 Step 16: Generating product images{language_info}...")

    product_config = load_product_config(collection)
    if not product_config:
        print("❌ Cannot generate images without product config")
        return

    try:
        with language_context(language):
            generated_images = generate_product_images(
                product_config=product_config,
                collection=collection,
                generate_individual=True,
            )
            print(f"✅ Generated {len(generated_images)} product images")
            for product_type, uri in generated_images.items():
                print(f"  {product_type}: {uri}")
    except Exception as e:
        print(f"❌ Failed to generate product images: {e}")


def generate_csv(collection: str, language: str = None):
    """Step 17: Generate Shopify CSV."""
    language_info = f" for {language}" if language else ""
    print(f"\n🔄 Step 17: Generating Shopify CSV{language_info}...")

    product_config = load_product_config(collection)
    if not product_config:
        print("❌ Cannot generate CSV without product config")
        return

    try:
        with language_context(language):
            csv_path = generate_shopify_csv(
                product_config=product_config,
                collection=collection,
                free_individual_count=1,
            )
            print(f"✅ Shopify CSV generated at: {csv_path}")
    except Exception as e:
        print(f"❌ Failed to generate Shopify CSV: {e}")


def execute_multi_language_step(func, collection: str, languages: list = None):
    """Execute a function for multiple languages or just the current language."""
    if languages:
        # Execute for each specified language
        for language in languages:
            func(collection, language)
    else:
        # Execute for current language only
        func(collection)


def merge_csv_files(languages: list = None):
    """Step 18: Merge all CSV files into a single dated CSV."""
    print("\n🔄 Step 18: Merging CSV files...")

    # Define the output directory where CSV files are generated
    output_dir = "../outputs/shopify"

    # If specific languages are provided, look for their CSV files
    if languages:
        csv_files = []
        for lang in languages:
            # CSV files use lowercase language names, so normalize for pattern matching
            lang_lower = lang.lower()
            lang_pattern = os.path.join(output_dir, f"*{lang_lower}*.csv")
            lang_files = glob.glob(lang_pattern)
            csv_files.extend(lang_files)
            print(f"  Looking for CSV files matching pattern: *{lang_lower}*.csv")

        if not csv_files:
            # Fallback to all CSV files
            csv_pattern = os.path.join(output_dir, "*.csv")
            csv_files = glob.glob(csv_pattern)
            print(
                f"  No language-specific files found, using all CSV files in {output_dir}"
            )
    else:
        csv_pattern = os.path.join(output_dir, "*.csv")
        csv_files = glob.glob(csv_pattern)

    try:
        if not csv_files:
            print(f"⚠️  No CSV files found in {output_dir}")
            return

        print(f"Found {len(csv_files)} CSV files to merge:")
        for csv_file in csv_files:
            print(f"  - {os.path.basename(csv_file)}")

        # Read and merge all CSV files
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dataframes.append(df)
                print(f"  ✅ Loaded {csv_file} ({len(df)} rows)")
            except Exception as e:
                print(f"  ⚠️  Failed to load {csv_file}: {e}")

        if not dataframes:
            print("❌ No CSV files could be loaded")
            return

        # Merge all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)

        # Create output filename with current date
        current_date = datetime.now().strftime("%Y%m%d")
        # Use lowercase for filename consistency
        lang_suffix = (
            f"_{'_'.join([lang.lower() for lang in languages])}" if languages else ""
        )
        output_filename = f"{current_date}{lang_suffix}_shopify.csv"
        output_path = os.path.join(output_dir, output_filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save merged CSV
        merged_df.to_csv(output_path, index=False)

        print(f"✅ Merged {len(dataframes)} CSV files into {output_path}")
        print(f"   Total rows: {len(merged_df)}")
        print(f"   Columns: {', '.join(merged_df.columns.tolist())}")

    except Exception as e:
        print(f"❌ Failed to merge CSV files: {e}")


def main():
    """Main function to process a collection into a new language."""
    parser = argparse.ArgumentParser(
        description="Process a collection into a new language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_collection_to_new_language.py WarmUp150
  python process_collection_to_new_language.py LM1000 --skip-audio
  python process_collection_to_new_language.py WarmUp150 --overwrite-audio --languages French Spanish
  python process_collection_to_new_language.py LM1000 --start-from zip
  python process_collection_to_new_language.py WarmUp150 --skip-phrases --skip-audio --start-from zip
  python process_collection_to_new_language.py LM1000 --only zip
  python process_collection_to_new_language.py WarmUp150 --only zip anki images csv
  python process_collection_to_new_language.py LM1000 --only albums
  python process_collection_to_new_language.py LM1000 --only merge-csv
  python process_collection_to_new_language.py WarmUp150 --only csv images --languages French Spanish German
  python process_collection_to_new_language.py LM1000 --only csv --languages French Spanish Italian
  python process_collection_to_new_language.py WarmUp150 --only csv images --languages french spanish german
  python process_collection_to_new_language.py LM1000 --only merge-csv --languages French Spanish Italian
        """,
    )

    parser.add_argument(
        "collection", help="Collection name to process (e.g., WarmUp150, LM1000)"
    )
    parser.add_argument(
        "--skip-phrases",
        action="store_true",
        help="Skip phrase translation and audio generation",
    )
    parser.add_argument(
        "--skip-audio", action="store_true", help="Skip audio generation steps"
    )
    parser.add_argument(
        "--overwrite-audio", action="store_true", help="Overwrite existing audio files"
    )
    parser.add_argument(
        "--skip-challenges", action="store_true", help="Skip challenge page creation"
    )
    parser.add_argument(
        "--skip-stories", action="store_true", help="Skip story page creation"
    )
    parser.add_argument(
        "--skip-albums", action="store_true", help="Skip album file creation"
    )
    parser.add_argument(
        "--skip-index", action="store_true", help="Skip index page updates"
    )
    parser.add_argument(
        "--skip-anki", action="store_true", help="Skip Anki deck creation"
    )
    parser.add_argument(
        "--skip-zip", action="store_true", help="Skip M4A zip file creation"
    )
    parser.add_argument(
        "--skip-images", action="store_true", help="Skip product image generation"
    )
    parser.add_argument(
        "--skip-csv", action="store_true", help="Skip Shopify CSV generation"
    )
    parser.add_argument(
        "--skip-merge-csv", action="store_true", help="Skip CSV file merging"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Languages for index page generation and CSV/image processing (default: current target language)",
    )
    parser.add_argument(
        "--story-limit",
        type=int,
        default=None,
        help="Limit the number of stories processed (default: all)",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["LM1000", "WarmUp150", "LM2000"],
        help="Collections for index page generation",
    )
    parser.add_argument(
        "--start-from",
        choices=[
            "phrases",
            "refine-phrases",
            "wiktionary-phrases",
            "phrase-audio",
            "story-translate",
            "refine-stories",
            "wiktionary-stories",
            "story-audio",
            "fast-audio",
            "challenges",
            "stories",
            "albums",
            "index",
            "anki",
            "zip",
            "images",
            "csv",
            "merge-csv",
        ],
        help="Start processing from a specific step",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[
            "phrases",
            "refine-phrases",
            "wiktionary-phrases",
            "phrase-audio",
            "story-translate",
            "refine-stories",
            "wiktionary-stories",
            "story-audio",
            "fast-audio",
            "challenges",
            "stories",
            "albums",
            "index",
            "anki",
            "zip",
            "images",
            "csv",
            "merge-csv",
        ],
        help="Run only the specified steps in order (cannot be used with --start-from)",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if args.only and args.start_from:
        print("❌ Cannot use --only and --start-from together")
        sys.exit(1)

    print(
        f"🚀 Processing collection '{args.collection}' into {config.TARGET_LANGUAGE_NAME}"
    )
    print("=" * 60)

    # Setup authentication
    setup_authentication()
    print_config_info()

    # Define the processing steps
    steps = [
        (
            "phrases",
            lambda: translate_phrases_step(
                args.collection, story_limit=args.story_limit
            ),
        ),
        ("refine-phrases", lambda: refine_phrase_translations(args.collection)),
        (
            "wiktionary-phrases",
            lambda: add_wiktionary_links_to_phrases(args.collection),
        ),
        ("phrase-audio", lambda: generate_phrase_audio(args.collection)),
        (
            "story-translate",
            lambda: translate_stories(args.collection, story_limit=args.story_limit),
        ),
        (
            "refine-stories",
            lambda: refine_story_translations(
                args.collection, story_limit=args.story_limit
            ),
        ),
        (
            "wiktionary-stories",
            lambda: add_wiktionary_links_to_stories(
                args.collection, story_limit=args.story_limit
            ),
        ),
        (
            "story-audio",
            lambda: generate_story_audio(
                args.collection, args.overwrite_audio, story_limit=args.story_limit
            ),
        ),
        (
            "fast-audio",
            lambda: generate_fast_audio(
                args.collection, args.overwrite_audio, story_limit=args.story_limit
            ),
        ),
        (
            "challenges",
            lambda: execute_multi_language_step(
                create_challenges(args.collection, story_limit=args.story_limit),
                args.collection,
                args.languages,
            ),
        ),
        (
            "stories",
            lambda: create_story_pages(args.collection, story_limit=args.story_limit),
        ),
        (
            "albums",
            lambda: create_albums(args.collection, story_limit=args.story_limit),
        ),
        ("index", lambda: update_index_pages()),
        ("anki", lambda: create_anki_decks(args.collection)),
        ("zip", lambda: create_zip_files(args.collection)),
        (
            "images",
            lambda: execute_multi_language_step(
                generate_images, args.collection, args.languages
            ),
        ),
        (
            "csv",
            lambda: execute_multi_language_step(
                generate_csv, args.collection, args.languages
            ),
        ),
        ("merge-csv", lambda: merge_csv_files(args.languages)),
    ]

    # Determine which steps to run
    if args.only:
        # Run only the specified steps in the order they appear in the steps list
        steps_to_run = []
        for step_name, step_func in steps:
            if step_name in args.only:
                steps_to_run.append((step_name, step_func))

        if not steps_to_run:
            print("❌ No valid steps specified with --only")
            sys.exit(1)

        print(f"🔄 Running only steps: {', '.join(args.only)}")

    elif args.start_from:
        # Find starting point and run from there
        try:
            start_index = next(
                i for i, (name, _) in enumerate(steps) if name == args.start_from
            )
            steps_to_run = list(enumerate(steps[start_index:], start_index + 1))
            print(f"🔄 Starting from step: {args.start_from}")
        except StopIteration:
            print(f"❌ Invalid start step: {args.start_from}")
            sys.exit(1)
    else:
        # Run all steps
        steps_to_run = list(enumerate(steps, 1))

    # Execute steps
    try:
        for item in steps_to_run:
            if args.only:
                # For --only, steps_to_run contains (step_name, step_func) tuples
                step_name, step_func = item
                step_number = next(
                    i for i, (name, _) in enumerate(steps, 1) if name == step_name
                )
                total_steps = len(steps_to_run)
                current_step = steps_to_run.index(item) + 1
            else:
                # For --start-from or default, steps_to_run contains (i, (step_name, step_func)) tuples
                step_number, (step_name, step_func) = item
                total_steps = len(steps)
                current_step = step_number

            # Skip steps based on arguments
            if (
                (
                    step_name
                    in [
                        "phrases",
                        "refine-phrases",
                        "wiktionary-phrases",
                        "phrase-audio",
                    ]
                    and args.skip_phrases
                )
                or (step_name in ["story-audio", "fast-audio"] and args.skip_audio)
                or (step_name == "challenges" and args.skip_challenges)
                or (step_name == "stories" and args.skip_stories)
                or (step_name == "albums" and args.skip_albums)
                or (step_name == "index" and args.skip_index)
                or (step_name == "anki" and args.skip_anki)
                or (step_name == "zip" and args.skip_zip)
                or (step_name == "images" and args.skip_images)
                or (step_name == "csv" and args.skip_csv)
                or (step_name == "merge-csv" and args.skip_merge_csv)
            ):
                print(f"\n⏭️  Skipping step {current_step}: {step_name}")
                continue

            if args.only:
                print(f"\n📍 Running step {current_step}/{total_steps}: {step_name}")
            else:
                print(f"\n📍 Running step {step_number}/{total_steps}: {step_name}")
            step_func()

        print("\n" + "=" * 60)
        print(
            f"🎉 Successfully processed collection '{args.collection}' into {config.TARGET_LANGUAGE_NAME}!"
        )

    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
