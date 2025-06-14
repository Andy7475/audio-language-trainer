#!/usr/bin/env python3
"""
Process a collection into a new language.

This script handles the complete pipeline for processing a story collection
into a new target language, including translation, audio generation, and
web page creation.
"""

import argparse
import os
import sys
from tqdm import tqdm

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
from src.dialogue_generation import translate_and_upload_dialogue
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


def translate_phrases_step(collection: str):
    """Step 1: Translate all phrases in the collection."""
    print("\n🔄 Step 1: Translating phrases...")

    all_stories = get_stories_from_collection(collection=collection)
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


def translate_stories(collection: str):
    """Step 5: Translate story dialogues."""
    print("\n🔄 Step 5: Translating story dialogues...")

    all_stories = get_stories_from_collection(collection=collection)

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


def refine_story_translations(collection: str):
    """Step 6: Refine story dialogue translations."""
    print("\n🔄 Step 6: Refining story dialogue translations...")

    all_stories = get_stories_from_collection(collection=collection)

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


def add_wiktionary_links_to_stories(collection: str):
    """Step 7: Add Wiktionary links to story dialogues."""
    print("\n🔄 Step 7: Adding Wiktionary links to story dialogues...")

    all_stories = get_stories_from_collection(collection=collection)

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


def generate_story_audio(collection: str, overwrite: bool = False):
    """Step 8: Generate audio for story dialogues."""
    print(f"\n🔄 Step 8: Generating story dialogue audio (overwrite={overwrite})...")

    all_stories = get_stories_from_collection(collection=collection)

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


def generate_fast_audio(collection: str, overwrite: bool = False):
    """Step 9: Generate fast audio for stories."""
    print(f"\n🔄 Step 9: Generating fast audio (overwrite={overwrite})...")

    all_stories = get_stories_from_collection(collection=collection)

    for story_name in tqdm(all_stories, desc="Generating fast audio"):
        try:
            generate_and_upload_fast_audio(
                story_name, collection=collection, overwrite=overwrite
            )
            print(f"  ✅ Generated fast audio for {story_name}")
        except Exception as e:
            print(f"  ❌ Failed to generate fast audio for {story_name}: {e}")

    print("✅ Fast audio generation completed")


def create_challenges(collection: str):
    """Step 10: Create challenge pages."""
    print("\n🔄 Step 10: Creating challenge pages...")

    all_stories = get_stories_from_collection(collection=collection)

    for story_name in tqdm(all_stories, desc="Creating challenges"):
        try:
            challenge_file_path = get_story_challenges_path(
                story_name, collection=collection
            )

            if not check_blob_exists(config.GCS_PRIVATE_BUCKET, challenge_file_path):
                print(f"  ⚠️  No challenges found for {story_name}, skipping")
                continue

            scenario_dicts = read_from_gcs(
                bucket_name=config.GCS_PRIVATE_BUCKET, file_path=challenge_file_path
            )
            challenges = get_html_challenge_inputs(scenario_dicts)
            chat_webpage_file = create_html_challenges(
                challenges, story_name=story_name, collection=collection
            )
            print(f"  ✅ Created challenges for {story_name}")
        except Exception as e:
            print(f"  ❌ Failed to create challenges for {story_name}: {e}")

    print("✅ Challenge pages created")


def create_story_pages(collection: str):
    """Step 11: Create story HTML pages and album files."""
    print("\n🔄 Step 11: Creating story pages and albums...")

    all_stories = get_stories_from_collection(collection=collection)

    for story_name in tqdm(all_stories, desc="Creating story pages"):
        try:
            print(f"  Processing {story_name}...")
            story_data = prepare_story_data_from_gcs(story_name, collection=collection)

            if not story_data:
                print(f"  ⚠️  No story data found for {story_name}, skipping")
                continue

            create_and_upload_html_story(story_data, story_name, collection=collection)
            create_album_files(story_data, story_name, collection=collection)
            print(f"  ✅ Created pages and album for {story_name}")
        except Exception as e:
            print(f"  ❌ Failed to create pages for {story_name}: {e}")

    print("✅ Story pages and albums created")


def update_index_pages():
    """Step 12: Update index pages."""
    print("\n🔄 Step 12: Updating index pages...")

    languages = ["French", "Spanish", "German", "Swedish"]
    collections = ["LM1000", "WarmUp150"]

    upload_styles_to_gcs()
    result = update_all_index_pages_hierarchical(
        languages=languages, collections=collections
    )

    print("✅ Index pages updated")
    return result


def create_anki_decks(collection: str):
    """Step 13: Create Anki decks from GCS data."""
    print("\n🔄 Step 13: Creating Anki decks...")

    try:
        create_anki_deck_from_gcs(
            collection=collection,
            output_dir="../outputs/gcs"
        )
        print("✅ Anki decks created successfully")
    except Exception as e:
        print(f"❌ Failed to create Anki decks: {e}")


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
        "--skip-index", action="store_true", help="Skip index page updates"
    )
    parser.add_argument(
        "--skip-anki", action="store_true", help="Skip Anki deck creation"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Languages for index page generation (default: current target language)",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["LM1000", "WarmUp150"],
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
            "index",
            "anki",
        ],
        help="Start processing from a specific step",
    )

    args = parser.parse_args()

    print(
        f"🚀 Processing collection '{args.collection}' into {config.TARGET_LANGUAGE_NAME}"
    )
    print("=" * 60)

    # Setup authentication
    setup_authentication()
    print_config_info()

    # Define the processing steps
    steps = [
        ("phrases", lambda: translate_phrases_step(args.collection)),
        ("refine-phrases", lambda: refine_phrase_translations(args.collection)),
        (
            "wiktionary-phrases",
            lambda: add_wiktionary_links_to_phrases(args.collection),
        ),
        ("phrase-audio", lambda: generate_phrase_audio(args.collection)),
        ("story-translate", lambda: translate_stories(args.collection)),
        ("refine-stories", lambda: refine_story_translations(args.collection)),
        (
            "wiktionary-stories",
            lambda: add_wiktionary_links_to_stories(args.collection),
        ),
        (
            "story-audio",
            lambda: generate_story_audio(args.collection, args.overwrite_audio),
        ),
        (
            "fast-audio",
            lambda: generate_fast_audio(args.collection, args.overwrite_audio),
        ),
        ("challenges", lambda: create_challenges(args.collection)),
        ("stories", lambda: create_story_pages(args.collection)),
        ("index", lambda: update_index_pages(args.languages, args.collections)),
        ("anki", lambda: create_anki_decks(args.collection)),
    ]

    # Find starting point
    start_index = 0
    if args.start_from:
        try:
            start_index = next(
                i for i, (name, _) in enumerate(steps) if name == args.start_from
            )
            print(f"🔄 Starting from step: {args.start_from}")
        except StopIteration:
            print(f"❌ Invalid start step: {args.start_from}")
            sys.exit(1)

    # Execute steps
    try:
        for i, (step_name, step_func) in enumerate(
            steps[start_index:], start_index + 1
        ):
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
                or (step_name == "index" and args.skip_index)
                or (step_name == "anki" and args.skip_anki)
            ):
                print(f"\n⏭️  Skipping step {i}: {step_name}")
                continue

            print(f"\n📍 Running step {i}/{len(steps)}: {step_name}")
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
