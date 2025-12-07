import json
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

from google.cloud import storage
from pydub import AudioSegment
from tqdm import tqdm

from src.config_loader import (
    config,
)  # Keep for backward compatibility with existing functions
from src.convert import (
    convert_audio_to_base64,
    convert_PIL_image_to_base64,
    get_story_title,
    get_collection_title,
)
from src.gcs_storage import (
    check_blob_exists,
    get_image_path,
    get_fast_audio_path,
    get_public_story_path,
    get_story_translated_dialogue_path,
    get_utterance_audio_path,
    get_wiktionary_cache_path,
    read_from_gcs,
    upload_to_gcs,
    get_story_collection_path,
    sanitize_path_component,
)
from src.llm_tools.story_generation import generate_story
from src.storage import (
    PRIVATE_BUCKET,
    get_story_dialogue_path,
    upload_file_to_gcs,
)
from src.utils import load_template
from src.wiktionary import generate_wiktionary_links


def upload_styles_to_gcs():
    """Upload the styles.css file to the public GCS bucket."""

    # Load the CSS content with correct path handling
    try:
        # Try current directory first (when running from project root)
        styles_content = load_template("styles.css", "src/templates")
    except FileNotFoundError:
        # Fallback to relative path (when running from subdirectory)
        styles_content = load_template("styles.css", "../src/templates")

    # Upload to GCS
    public_url = upload_to_gcs(
        obj=styles_content,
        bucket_name=config.GCS_PUBLIC_BUCKET,
        file_name="styles.css",
        content_type="text/css",
    )

    print("✅ Styles uploaded successfully!")
    print(f"🌐 Public URL: {public_url}")

    return public_url


def create_and_upload_html_story(
    prepared_data: Dict,
    story_name: str,
    bucket_name: str = config.GCS_PUBLIC_BUCKET,
    component_path: str = "StoryViewer.js",
    template_path: str = "story_template.html",
    collection: str = "LM1000",
) -> str:
    """
    Create a standalone HTML file from prepared story data and upload it to GCS.

    Args:
        prepared_data: Dictionary containing prepared story data with base64 encoded assets
        story_name: Name of the story
        bucket_name: GCS bucket name for upload
        component_path: Path to the React component file
        template_path: Path to the HTML template file
        output_dir: Local directory to save HTML file before upload
        collection: Collection name for organizing stories

    Returns:
        str: Public URL of the uploaded HTML file
    """

    language = config.TARGET_LANGUAGE_NAME

    # Clean the story name for display
    story_title = get_story_title(story_name)

    # Read the React component
    react_component = load_template(component_path)

    # Read the HTML template
    template = Template(load_template(template_path))

    # Substitute the template variables
    html_content = template.substitute(
        title=story_title,
        story_data=json.dumps(prepared_data),
        language=language,
        react_component=react_component,
        collection_name=get_collection_title(collection),
        collection_raw=collection,
    )

    try:
        blob_path = get_public_story_path(story_name, collection)

        public_url = upload_to_gcs(
            obj=html_content,
            bucket_name=bucket_name,
            file_name=blob_path,
            content_type="text/html",
        )
        print(f"Uploaded HTML story to: {public_url}")

        return public_url

    except Exception as e:
        print(f"Error uploading to GCS: {str(e)}")
        return str(blob_path)  # Return local path if upload fails


def generate_language_section(langauge: str, stories: List[Dict[str, str]]) -> str:
    """Generate HTML for a single language section."""

    stories_html = "\n".join(
        f'<li><a href="{story["url"]}">{story["name"]}</a></li>'
        for story in sorted(stories, key=lambda x: x["name"])
    )

    return f"""
    <section class="language-section" id="{langauge.lower()}">
        <h2>{langauge}</h2>
        <ul class="story-list">
            {stories_html}
        </ul>
    </section>
    """


def generate_special_pages_section(special_pages: List[Dict[str, str]]) -> str:
    """Generate HTML for special pages section."""
    if not special_pages:
        return ""

    links_html = "\n".join(
        f'<a href="{page["url"]}">{page["name"]}</a>' for page in special_pages
    )

    return f"""
    <div class="special-pages">
        <h3>Additional Resources</h3>
        {links_html}
    </div>
    """


def prepare_dialogue_with_wiktionary(story_data_dict: dict, language_name: str = None):
    """Add wiktionary links to utterances for target language text.

    Args:
        story_data_dict: Story data dictionary containing dialogue
        language_name: Name of target language for Wiktionary links

    Returns:
        List of processed utterances with added wiktionary_links field
    """
    if language_name is None:
        language_name = config.TARGET_LANGUAGE_NAME

    word_link_cache_path = get_wiktionary_cache_path()

    if check_blob_exists(config.GCS_PRIVATE_BUCKET, word_link_cache_path):
        word_link_cache = read_from_gcs(
            config.GCS_PRIVATE_BUCKET, word_link_cache_path, "json"
        )
        print(f"Got word link cache of size {len(word_link_cache)} from GCS")
    else:
        word_link_cache = {}

    story_data_dict_copy = story_data_dict.copy()
    for story_part in tqdm(
        story_data_dict_copy, desc="Getting dialogue links for story_parts"
    ):
        dialogue = story_data_dict_copy[story_part].get("translated_dialogue", [])
        # Process each utterance in the dialogue
        for utterance in dialogue:
            # Generate Wiktionary links for the target language text

            wiktionary_links, word_link_cache = generate_wiktionary_links(
                utterance["text"],
                language_name=language_name,
                word_link_cache=word_link_cache,
                return_cache=True,
            )
            utterance["wiktionary_links"] = wiktionary_links

    # upload the additional word_link cache to GCS
    upload_to_gcs(
        obj=word_link_cache,
        bucket_name=config.GCS_PRIVATE_BUCKET,
        file_name=word_link_cache_path,
    )

    return story_data_dict_copy


def upload_story_image(
    image_file: str,
    story_part: str,
    story_name: str,
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
) -> str:
    """
    Upload an image file for a story part to GCS.

    Args:
        image_file: Path to the image file
        story_part: Part of the story (e.g., 'introduction')
        story_name: Name of the story
        collection: Collection name (e.g., 'LM1000', 'LM2000')
        bucket_name: Optional bucket name

    Returns:
        GCS URI of the uploaded image file
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Ensure story_name is properly formatted
    story_name = (
        f"story_{story_name}" if not story_name.startswith("story_") else story_name
    )

    # Create the GCS path
    image_path = get_image_path(story_name, story_part, collection)
    # Read the image file
    with open(image_file, "rb") as f:
        image_data = f.read()

    # Upload the image file
    gcs_uri = upload_to_gcs(
        obj=image_data,
        bucket_name=bucket_name,
        file_name=image_path,
    )

    print(f"Image uploaded to {gcs_uri}")
    return gcs_uri


def prepare_story_data_from_gcs(
    story_name: str,
    bucket_name: str = config.GCS_PRIVATE_BUCKET,
    collection: str = "LM1000",
) -> Dict:
    """
    Prepare story data for HTML by downloading required files from GCS.

    Args:
        bucket_name: GCS bucket name
        story_name: Name of the story (e.g., "story_a_fishing_trip")
        language: Target language (e.g., "french")
        collection: Collection name (default: "LM1000")

    Returns:
        Dict: Prepared data for HTML rendering
    """

    language_name = config.TARGET_LANGUAGE_NAME.lower()
    # Get story translated dialogue, this will have wiktionary links already
    dialogue_path = get_story_translated_dialogue_path(story_name, collection)
    try:
        story_dialogue = read_from_gcs(bucket_name, dialogue_path, "json")
    except FileNotFoundError:
        print(f"Dialogue not found for {story_name} in {language_name}")
        return {}

    prepared_data = {}

    # Process each section of the story (introduction, development, etc.)
    for story_part, dialogue in tqdm(
        story_dialogue.items(), desc=f"Preparing {story_name} in {language_name}"
    ):
        try:
            # Initialize the section data structure
            fast_audio_segment = read_from_gcs(
                config.GCS_PRIVATE_BUCKET,
                get_fast_audio_path(story_name, story_part, collection),
                "audio",
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Fast audio not found for {story_part}: {str(e)}")
            fast_audio_segment = AudioSegment.silent(duration=1000)  # 1 second silence

        prepared_data[story_part] = {
            "dialogue": dialogue.get("dialogue", []),
            "translated_dialogue": dialogue.get("translated_dialogue", []),
            "audio_data": {
                "dialogue": [],
                "fast_dialogue": convert_audio_to_base64(
                    fast_audio_segment
                ),  # will be base64 encoded audio for whole story part
            },
        }

        # Process audio for each dialogue utterance
        for i, utterance in tqdm(
            enumerate(dialogue.get("translated_dialogue", [])),
            colour="red",
            desc="Downloading utterance audio",
        ):
            try:
                audio_path = get_utterance_audio_path(
                    story_name,
                    story_part,
                    i,
                    utterance["speaker"],
                    language_name,
                    collection,
                )
                audio_segment = read_from_gcs(bucket_name, audio_path, "audio")
                audio_base64 = convert_audio_to_base64(audio_segment)
                prepared_data[story_part]["audio_data"]["dialogue"].append(audio_base64)
            except (FileNotFoundError, ValueError) as e:
                print(
                    f"Warning: Audio not found for utterance {i} in {story_part}: {str(e)}"
                )

        # Add image data
        try:
            image_path = get_image_path(story_name, story_part, collection)
            image_data = read_from_gcs(bucket_name, image_path, "image")
            image_base64 = convert_PIL_image_to_base64(image_data)
            prepared_data[story_part]["image_data"] = image_base64
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Image not found for {story_part}: {str(e)}")

    return prepared_data


def update_all_index_pages_hierarchical(
    languages: List[str] = None,
    collections: List[str] = None,
    bucket_name: str = None,
    force_upload: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Update all index pages using the new hierarchical system (Language > Collection > Story).
    Only generates index pages for stories and collections that actually exist.

    This function generates and uploads:
    - Main language selector index (index.html)
    - Language-level collection indexes
    - Collection-level story indexes (with numbered stories)

    Args:
        languages: List of language names (defaults to [config.TARGET_LANGUAGE_NAME])
        collections: List of collection names (defaults to ["LM1000"])
        bucket_name: Name of the GCS bucket (defaults to config.GCS_PUBLIC_BUCKET)
        force_upload: Whether to upload generated files even if they exist
        verbose: Whether to print detailed progress information

    Returns:
        dict: Dictionary with paths and URLs for all generated index pages
    """
    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET
    if languages is None:
        languages = [config.TARGET_LANGUAGE_NAME]
    if collections is None:
        collections = ["LM1000", "WarmUp150"]

    if verbose:
        print(f"Starting hierarchical index generation for bucket: {bucket_name}")
        print(f"Checking existence of content for languages: {languages}")
        print(f"Checking existence of content for collections: {collections}")

    # Filter to only existing languages and collections
    existing_languages = get_existing_languages(languages, collections, bucket_name)

    if verbose:
        print(f"Found existing languages: {existing_languages}")
        if not existing_languages:
            print(
                "❌ No existing languages found with content. Skipping index generation."
            )
            return {"error": "No existing languages found with content"}

    # Upload latest styles first
    if verbose:
        print("Uploading latest styles...")
    upload_styles_to_gcs()

    results = {}

    try:
        # Generate hierarchical index system with only existing content
        if verbose:
            print("Generating hierarchical index system...")

        hierarchical_results = generate_hierarchical_index_system(
            languages=existing_languages,
            collections=collections,  # Pass all collections, filtering happens inside
            bucket_name=bucket_name,
            upload=force_upload,
        )

        results.update(
            {
                "main_index": {"url": hierarchical_results["main_index"]},
                "language_indexes": hierarchical_results["language_indexes"],
                "collection_indexes": hierarchical_results["collection_indexes"],
            }
        )

        if verbose:
            print(f"✅ Main index: {hierarchical_results['main_index']}")
            for lang, url in hierarchical_results["language_indexes"].items():
                print(f"✅ {lang} language index: {url}")
            for key, url in hierarchical_results["collection_indexes"].items():
                print(f"✅ {key} collection index: {url}")

        return results

    except Exception as e:
        error_msg = f"Error updating hierarchical index pages: {str(e)}"
        print(f"❌ {error_msg}")
        results["error"] = error_msg
        return results


def generate_hierarchical_index_system(
    languages: List[str] = None,
    collections: List[str] = None,
    bucket_name: str = None,
    upload: bool = True,
) -> dict:
    """
    Generate a hierarchical index system: Main Index > Language Index > Collection Index.
    Only generates indexes for content that actually exists.

    This replaces the bucket scraping approach with a structured system based on
    collection data files.

    Args:
        languages: List of language names (defaults to [config.TARGET_LANGUAGE_NAME])
        collections: List of collection names (defaults to ["LM1000"])
        bucket_name: GCS bucket name (defaults to config.GCS_PUBLIC_BUCKET)
        upload: Whether to upload generated files

    Returns:
        dict: Results with URLs for all generated index pages
    """

    if languages is None:
        languages = [config.TARGET_LANGUAGE_NAME]
    if collections is None:
        collections = ["LM1000"]
    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET

    print("🏗️  Starting hierarchical index generation...")
    print(f"   Languages to process: {languages}")
    print(f"   Collections to check: {collections}")
    print(f"   Target bucket: {bucket_name}")

    results = {"main_index": None, "language_indexes": {}, "collection_indexes": {}}

    # 1. Generate main index (language selector) - only for existing languages
    print("\n📄 Generating main language index...")
    results["main_index"] = generate_main_language_index(
        languages=languages,
        bucket_name=bucket_name,
        upload=upload,
        collections=collections,
    )

    # 2. Generate language-level indexes only for languages with existing content
    for language in languages:
        print(f"\n🌐 Processing language: {language}")
        existing_collections = get_existing_collections_for_language(
            language, collections, bucket_name
        )

        if existing_collections:
            print(
                f"   Found existing collections for {language}: {existing_collections}"
            )
            results["language_indexes"][language] = generate_language_collection_index(
                language=language,
                collections=existing_collections,  # Only pass existing collections
                bucket_name=bucket_name,
                upload=upload,
            )

            # 3. Generate collection-level indexes for this language's existing collections
            for collection in existing_collections:
                print(f"   📚 Processing collection: {collection}")
                key = f"{language}_{collection}"
                results["collection_indexes"][key] = generate_collection_story_index(
                    language=language,
                    collection=collection,
                    bucket_name=bucket_name,
                    upload=upload,
                )
        else:
            print(f"   ⚠️  No existing collections found for {language}")

    print("\n✅ Hierarchical index generation complete!")
    return results


def generate_main_language_index(
    languages: List[str],
    bucket_name: str,
    upload: bool = True,
    collections: List[str] = None,
) -> str:
    """Generate the main index page that shows available languages.
    Only includes languages that have existing content."""

    if collections is None:
        collections = ["LM1000"]

    print(f"   🔍 Checking content for languages: {languages}")

    # Get language statistics - only for languages with existing content
    language_cards = ""
    special_pages = get_special_pages_from_bucket(bucket_name)

    for language in languages:
        # Get existing collections for this language
        existing_collections = get_existing_collections_for_language(
            language, collections, bucket_name
        )

        if not existing_collections:
            print(f"   ❌ Skipping language {language} - no existing collections found")
            continue

        print(
            f"   ✅ Including language {language} with collections: {existing_collections}"
        )

        # Get stats for this language across existing collections
        total_stories = 0

        for collection in existing_collections:
            try:
                collection_data = read_from_gcs(
                    config.GCS_PRIVATE_BUCKET,
                    get_story_collection_path(collection),
                    "json",
                )
                if collection_data:
                    total_stories += len(collection_data)
            except:
                continue

        stats_text = f"{len(existing_collections)} collections, {total_stories} stories"
        language_url = f"{language.lower()}/index.html"

        language_cards += f"""
        <div class="language-card">
            <div class="language-name">{language}</div>
            <div class="language-stats">{stats_text}</div>
            <a href="{language_url}" class="language-link">Browse Stories</a>
        </div>
        """

    # Only generate main index if there are languages with content
    if not language_cards.strip():
        print(
            "   ❌ No languages with existing content found. Skipping main index generation."
        )
        return None

    # Generate special pages section
    special_pages_html = ""
    if special_pages:
        special_links = "\n".join(
            f'<a href="{page["url"]}" class="special-link">{page["name"]}</a>'
            for page in special_pages
        )
        special_pages_html = f"""
        <div class="special-section">
            <h2>Additional Resources</h2>
            <div class="special-links">
                {special_links}
            </div>
        </div>
        """

    # Load and fill template
    template = Template(load_template("index_template.html"))
    html_content = template.substitute(
        language_cards=language_cards,
        special_pages=special_pages_html,
    )

    # Upload directly using upload_to_gcs (which handles local saving automatically)
    if upload:
        url = upload_to_gcs(
            obj=html_content,
            bucket_name=bucket_name,
            file_name="index.html",
            content_type="text/html",
        )
        print(f"   ✅ Created main index: {url}")
        return url

    return None


def generate_language_collection_index(
    language: str,
    collections: List[str],
    bucket_name: str,
    upload: bool = True,
) -> str:
    """Generate a language-level index showing available collections.
    Only includes collections that exist for the given language."""

    print(
        f"     🔍 Generating language index for {language} with collections: {collections}"
    )

    collection_cards = ""

    for collection in collections:
        try:
            # Get collection data
            collection_data = read_from_gcs(
                config.GCS_PRIVATE_BUCKET, get_story_collection_path(collection), "json"
            )

            # Double-check that this collection exists for this language
            if not check_collection_exists_for_language(
                collection, language, bucket_name
            ):
                print(
                    f"     ❌ Collection {collection} doesn't exist for language {language}, skipping"
                )
                continue

            story_count = len(collection_data)
            collection_title = get_collection_title(collection)
            print(f"     ✅ Including collection {collection} ({story_count} stories)")

            collection_url = f"{collection.lower()}/index.html"

            collection_cards += f"""
            <div class="collection-card">
                <div class="collection-title">{collection_title}</div>
                <div class="collection-stats">{story_count} stories</div>
                <div class="collection-links">
                    <a href="{collection_url}" class="collection-link primary">View Stories</a>
                </div>
            </div>
            """

        except Exception as e:
            print(f"     ⚠️  Warning: Could not load collection {collection}: {e}")
            continue

    # Only generate language index if there are collections with content
    if not collection_cards.strip():
        print(
            f"     ❌ No collections with existing content found for language {language}. Skipping language index generation."
        )
        return None

    # Load and fill template
    template = Template(load_template("language_index_template.html"))
    html_content = template.substitute(
        language_name=language,
        collection_cards=collection_cards,
        main_index_url="../index.html",
    )

    # Upload directly using upload_to_gcs
    if upload:
        blob_path = f"{language.lower()}/index.html"
        url = upload_to_gcs(
            obj=html_content,
            bucket_name=bucket_name,
            file_name=blob_path,
            content_type="text/html",
        )
        print(f"     ✅ Created language index for {language}: {url}")
        return url

    return None


def generate_collection_story_index(
    language: str,
    collection: str,
    bucket_name: str,
    upload: bool = True,
) -> str:
    """Generate a collection-level index showing stories ordered by position.
    Only includes stories that actually exist for the given language."""

    print(f"       🔍 Generating collection index for {language}/{collection}")

    try:
        # Get collection data
        collection_data = read_from_gcs(
            config.GCS_PRIVATE_BUCKET, get_story_collection_path(collection), "json"
        )

        story_cards = ""
        existing_story_count = 0

        # Generate story cards ordered by position - only for existing stories
        for position, (story_name, story_data) in enumerate(collection_data.items(), 1):
            # Check if this story actually exists for this language
            if not check_story_exists_for_language(
                story_name, collection, language, bucket_name
            ):
                print(
                    f"       ❌ Story {story_name} doesn't exist for language {language}, skipping"
                )
                continue

            existing_story_count += 1
            story_title = get_story_title(story_name)
            phrase_count = len(story_data) if isinstance(story_data, list) else 0

            print(
                f"       ✅ Including story {position:02d}: {story_title} ({phrase_count} phrases)"
            )

            # Generate URLs - relative from collection index to story folder
            story_url = f"{story_name}/{story_name}.html"
            challenges_url = f"{story_name}/challenges.html"

            story_cards += f"""
            <div class="story-card">
                <div class="story-number">{position:02d}</div>
                <div class="story-title">{story_title}</div>
                <div class="story-info">{phrase_count} phrases</div>
                <div class="story-links">
                    <a href="{story_url}" class="story-link primary">Read Story</a>
                    <a href="{challenges_url}" class="story-link secondary">Challenges</a>
                </div>
            </div>
            """

        # Only generate collection index if there are existing stories
        if existing_story_count == 0:
            print(
                f"       ❌ No existing stories found for collection {collection} in language {language}. Skipping collection index generation."
            )
            return None

        collection_title = get_collection_title(collection)

        # Load and fill template
        template = Template(load_template("collection_index_template.html"))
        html_content = template.substitute(
            collection_title=collection_title,
            language_name=language,
            story_count=existing_story_count,
            story_cards=story_cards,
            main_index_url="../../index.html",
            language_index_url="../index.html",
        )

        # Upload directly using upload_to_gcs
        if upload:
            blob_path = f"{language.lower()}/{collection.lower()}/index.html"
            url = upload_to_gcs(
                obj=html_content,
                bucket_name=bucket_name,
                file_name=blob_path,
                content_type="text/html",
            )
            print(
                f"       ✅ Created collection index for {language}/{collection}: {url}"
            )
            return url

        return None

    except Exception as e:
        print(
            f"       ❌ Error generating collection index for {language}/{collection}: {e}"
        )
        return None


def get_special_pages_from_bucket(bucket_name: str) -> List[Dict[str, str]]:
    """Get special pages (non-story pages) from bucket."""
    special_pages = []

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        for blob in bucket.list_blobs():
            if blob.name.endswith(".html"):
                path = Path(blob.name)
                parts = path.parts

                # Only root-level files that aren't index.html
                if len(parts) == 1 and parts[0] != "index.html":
                    special_pages.append(
                        {
                            "name": parts[0]
                            .replace(".html", "")
                            .replace("_", " ")
                            .title(),
                            "url": f"https://storage.googleapis.com/{bucket_name}/{blob.name}",
                        }
                    )
    except Exception as e:
        print(f"Warning: Could not load special pages: {e}")

    return special_pages


def check_collection_exists_for_language(
    collection: str,
    language: str,
    bucket_name: str = None,
) -> bool:
    """
    Check if a collection has any content for a specific language.

    Args:
        collection: Collection name (e.g., "LM1000")
        language: Language name (e.g., "French")
        bucket_name: GCS bucket name (defaults to config.GCS_PUBLIC_BUCKET)

    Returns:
        bool: True if the collection exists and has content for the language
    """
    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET

    try:
        # First check if the collection data file exists
        collection_data = read_from_gcs(
            config.GCS_PRIVATE_BUCKET,
            get_story_collection_path(collection),
            "json",
        )

        if not collection_data:
            return False

        # Check if there's at least one story published for this language/collection
        language_folder = sanitize_path_component(language.lower())
        collection_folder = sanitize_path_component(collection.lower())

        # Check for any story HTML files in the public bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        prefix = f"{language_folder}/{collection_folder}/"

        # Look for any .html files in the collection folder
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if blob.name.endswith(".html"):
                return True

        return False

    except Exception as e:
        print(f"Error checking collection {collection} for language {language}: {e}")
        return False


def check_story_exists_for_language(
    story_name: str,
    collection: str,
    language: str,
    bucket_name: str = None,
) -> bool:
    """
    Check if a specific story exists for a given language and collection.

    Args:
        story_name: Name of the story
        collection: Collection name
        language: Language name
        bucket_name: GCS bucket name (defaults to config.GCS_PUBLIC_BUCKET)

    Returns:
        bool: True if the story exists for the language
    """
    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET

    # Check if the story HTML file exists
    story_path = get_public_story_path(story_name, collection)
    # The path uses config.TARGET_LANGUAGE_NAME, so we need to construct it manually
    language_folder = sanitize_path_component(language.lower())
    collection_folder = sanitize_path_component(collection.lower())
    story_folder = sanitize_path_component(story_name)
    story_path = (
        f"{language_folder}/{collection_folder}/{story_folder}/{story_name}.html"
    )

    return check_blob_exists(bucket_name, story_path)


def get_existing_collections_for_language(
    language: str,
    collections: List[str],
    bucket_name: str = None,
) -> List[str]:
    """
    Get list of collections that actually exist for a specific language.

    Args:
        language: Language name
        collections: List of collection names to check
        bucket_name: GCS bucket name

    Returns:
        List[str]: Collections that exist for the language
    """
    existing_collections = []

    for collection in collections:
        if check_collection_exists_for_language(collection, language, bucket_name):
            existing_collections.append(collection)

    return existing_collections


def get_existing_languages(
    languages: List[str],
    collections: List[str],
    bucket_name: str = None,
) -> List[str]:
    """
    Get list of languages that have at least one collection with content.

    Args:
        languages: List of language names to check
        collections: List of collection names to check
        bucket_name: GCS bucket name

    Returns:
        List[str]: Languages that have existing content
    """
    existing_languages = []

    for language in languages:
        # Check if this language has any existing collections
        existing_collections = get_existing_collections_for_language(
            language, collections, bucket_name
        )
        if existing_collections:
            existing_languages.append(language)

    return existing_languages


# ============================================================================
# NEW STORY GENERATION FUNCTIONS (using llm_tools pattern)
# ============================================================================


def generate_and_upload_story(
    verbs: List[str],
    vocab: List[str],
    collection: str = "LM1000",
    bucket_name: str = PRIVATE_BUCKET,
) -> Tuple[str, Dict, str]:
    """Generate English story and upload to GCS using modern llm_tools pattern.

    Stories are ALWAYS generated in English. Translation happens separately
    using existing translation functions.

    This is the recommended way to generate new stories. It uses:
    - src.llm_tools.story_generation for LLM calls
    - src.storage for path generation and uploads
    - Dynamic story structure based on verb count

    Args:
        verbs: List of verbs to incorporate in the story
        vocab: List of other vocabulary words to use
        collection: Collection name (default: "LM1000")
        bucket_name: GCS bucket for upload (default: PRIVATE_BUCKET)

    Returns:
        Tuple of (story_name, story_dialogue, gcs_uri)
        - story_name: 3-word story title
        - story_dialogue: Dictionary with story parts and dialogue
        - gcs_uri: GCS URI where dialogue JSON was uploaded

    Example:
        >>> story_name, dialogue, uri = generate_and_upload_story(
        ...     verbs=["go", "see", "want"],
        ...     vocab=["coffee", "table", "friend"],
        ...     collection="LM1000"
        ... )
        >>> print(f"Created story: {story_name}")
        >>> print(f"Saved to: {uri}")
    """
    # Generate story using LLM tool (always in English)
    print(f"Generating story from {len(verbs)} verbs and {len(vocab)} vocab words...")
    story_name, story_dialogue = generate_story(verbs, vocab)

    print(f"Generated story: {story_name}")

    # Upload dialogue to GCS
    dialogue_path = get_story_dialogue_path(story_name, collection)
    gcs_uri = upload_file_to_gcs(
        obj=story_dialogue,
        bucket_name=bucket_name,
        file_path=dialogue_path,
    )

    print(f"Uploaded story dialogue to: {gcs_uri}")

    return story_name, story_dialogue, gcs_uri
