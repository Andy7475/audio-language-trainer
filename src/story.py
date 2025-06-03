import base64
import json
import os
import zipfile
from collections import defaultdict
from pathlib import Path
from string import Template
from typing import Dict, List, Optional

from google.cloud import storage
from pydub import AudioSegment
from tqdm import tqdm

from src.audio_generation import create_m4a_with_timed_lyrics
from src.config_loader import config
from src.convert import (
    convert_audio_to_base64,
    convert_base64_list_to_audio_segments,
    convert_base64_to_audio,
    convert_PIL_image_to_base64,
    get_story_title,
    get_collection_title,
)
from src.gcs_storage import (
    check_blob_exists,
    get_fast_audio_path,
    get_image_path,
    get_m4a_file_path,
    get_m4a_filename,
    get_public_story_path,
    get_story_translated_dialogue_path,
    get_utterance_audio_path,
    get_wiktionary_cache_path,
    read_from_gcs,
    upload_to_gcs,
    get_story_collection_path,
)
from src.utils import load_template, get_story_position
from src.wiktionary import generate_wiktionary_links


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

    # Read the shared styles
    styles = load_template("styles.css")

    # Substitute the template variables
    html_content = template.substitute(
        title=story_title,
        story_data=json.dumps(prepared_data),
        language=language,
        react_component=react_component,
        styles=styles,
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


def create_html_story(
    story_data_dict: Dict,
    image_dir: str,
    story_name: str,
    language: str = None,
    component_path: str = "StoryViewer.js",
    template_path: str = "story_template.html",
) -> Path:
    """
    Create a standalone HTML file from the story data dictionary using string.Template.

    Args:
        story_data_dict: Dictionary containing story data, translations, and audio
        image_dir: Path where images are stored and output HTML will be saved
        story_name: Name of the story
        language: Target language name for Wiktionary links (defaults to config.TARGET_LANGUAGE_NAME)
        component_path: Path to the React component file..uses default parent folder in load_template()
        template_path: Path to the HTML template file..uses default parent folder in load_template()

    Returns:
        The html file path
    """
    if language is None:
        language = config.TARGET_LANGUAGE_NAME
    story_title = get_story_title(story_name)

    # Process the story data and convert audio to base64
    prepared_data = prepare_story_data_for_html(
        story_data_dict,
        story_name=story_name,
        m4a_folder=image_dir / language,
        image_folder=image_dir,
    )

    # Read the React component
    react_component = load_template(component_path)

    # Read the HTML template
    template = Template(load_template(template_path))

    # Read the shared styles
    styles = load_template("styles.css")

    # Substitute the template variables
    html_content = template.substitute(
        title=story_title,
        story_data=json.dumps(prepared_data),
        language=language,
        react_component=react_component,
        styles=styles,
    )

    # Create html file path
    html_path = image_dir / language / f"{story_name}.html"

    # Create parent directory if it doesn't exist
    html_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the HTML file
    html_path.write_text(html_content, encoding="utf-8")

    print(f"HTML story created at: {html_path}")
    return html_path


def create_album_files(
    story_data_dict: dict, story_name: str, collection: str = "LM1000"
):
    """Creates and saves M4A files for the story, with album artwork.
    Optionally uploads files to Google Cloud Storage.
    """
    REPEATS_OF_FAST_DIALOGUE = 10
    PAUSE_TEXT = "---------"
    GAP_BETWEEN_PHRASES = AudioSegment.silent(duration=500)

    gcs_bucket_name = config.GCS_PRIVATE_BUCKET
    # Get story position from collection
    story_position = get_story_position(story_name, collection)

    ALBUM_NAME = (
        f"{story_position:02d} - "
        + get_story_title(story_name)
        + f" ({config.TARGET_LANGUAGE_NAME})"
    )
    TOTAL_TRACKS = len(story_data_dict) * 2  # 1 track normal, 1 track fast

    m4a_file_paths = []

    story_data_first_key = list(story_data_dict.keys())[0]
    cover_image_base64 = story_data_dict[story_data_first_key]["image_data"]

    # Create the story tracks for each story section
    for track_number, (story_part, data) in enumerate(
        tqdm(story_data_dict.items(), desc="creating album"), start=1
    ):
        audio_list = []
        captions_list = []

        # Get dialogue texts and audio
        dialogue_list = [utterance["text"] for utterance in data["translated_dialogue"]]
        dialogue_audio_list = convert_base64_list_to_audio_segments(
            data["audio_data"]["dialogue"]
        )

        # Initial dialogue at normal speed
        audio_list.append(GAP_BETWEEN_PHRASES)
        captions_list.append(f"{story_part} - Initial Dialogue")

        audio_list.extend(dialogue_audio_list)
        captions_list.extend(dialogue_list)

        # Create M4A file
        m4a_filename = get_m4a_file_path(
            story_name,
            story_part,
            fast=False,
            story_position=story_position,
            collection=collection,
        )

        m4a_path = create_m4a_with_timed_lyrics(
            audio_segments=audio_list,
            phrases=captions_list,
            m4a_filename=m4a_filename,
            album_name=ALBUM_NAME,
            track_title=story_part,
            track_number=track_number,
            total_tracks=TOTAL_TRACKS,
            cover_image_base64=cover_image_base64,
            bucket_name=gcs_bucket_name,
        )
        m4a_file_paths.append(m4a_path)
        print(f"Saved M4A file track number {track_number}")

    # Now generate fast versions
    for track_number, (story_part, data) in enumerate(
        tqdm(story_data_dict.items(), desc="creating fast tracks for album"),
        start=len(story_data_dict) + 1,
    ):
        audio_list = []
        captions_list = []

        # Fast dialogue section
        captions_list.append(f"{story_part} - Fast Dialogue Practice")
        fast_dialogue_audio = convert_base64_to_audio(
            data["audio_data"]["fast_dialogue"]
        )
        # Add fast dialogue (there are 10 repeats in the audio)
        for i in range(REPEATS_OF_FAST_DIALOGUE):
            audio_list.append(fast_dialogue_audio)
            captions_list.append(f"Fast Dialogue - Repetition {i+1}")
            audio_list.append(GAP_BETWEEN_PHRASES)
            captions_list.append(PAUSE_TEXT)

        # Create M4A file
        m4a_filename_fast = get_m4a_file_path(
            story_name=story_name,
            story_part=story_part,
            fast=True,
            story_position=story_position,
        )
        m4a_path = create_m4a_with_timed_lyrics(
            audio_segments=audio_list,
            phrases=captions_list,
            m4a_filename=m4a_filename_fast,
            album_name=ALBUM_NAME,
            track_title=story_part + " (fast)",
            track_number=track_number,
            total_tracks=TOTAL_TRACKS,
            cover_image_base64=cover_image_base64,
            bucket_name=gcs_bucket_name,
        )
        m4a_file_paths.append(m4a_path)
        print(f"Saved M4A file track number {track_number}")

    return m4a_file_paths


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


def prepare_story_data_for_html(
    story_data_dict: Dict,
    story_name: str,
    m4a_folder: Optional[str] = None,
    image_folder: Optional[str] = None,
    collection: str = "LM1000",
) -> Dict:
    """Process the story data dictionary to include base64 encoded audio and images.
    M4A files are now handled separately through the m4a_index.html page."""
    prepared_data = {}

    for section_name, section_data in tqdm(
        story_data_dict.items(), desc="Preparing HTML data"
    ):
        prepared_data[section_name] = {
            "dialogue": section_data.get("dialogue", []),
            "translated_dialogue": prepare_dialogue_with_wiktionary(
                section_data.get("translated_dialogue", []),
                language_name=config.TARGET_LANGUAGE_NAME,
            ),
            "audio_data": {
                "dialogue": [],
                "fast_dialogue": None,
            },
        }

        # Process normal dialogue audio
        if "translated_dialogue_audio" in section_data:
            for audio_segment in section_data["translated_dialogue_audio"]:
                audio_base64 = convert_audio_to_base64(audio_segment)
                prepared_data[section_name]["audio_data"]["dialogue"].append(
                    audio_base64
                )

        # Process fast dialogue audio (single segment per section)
        if "translated_dialogue_audio_fast" in section_data:
            fast_audio_base64 = convert_audio_to_base64(
                section_data["translated_dialogue_audio_fast"]
            )
            prepared_data[section_name]["audio_data"][
                "fast_dialogue"
            ] = fast_audio_base64

        # Add image data if folder is provided
        if image_folder:
            image_filename = f"{story_name}_{section_name}.png"
            image_path = os.path.join(image_folder, image_filename)

            try:
                if os.path.exists(image_path):
                    with open(image_path, "rb") as img_file:
                        image_data = img_file.read()
                        image_base64 = base64.b64encode(image_data).decode("utf-8")
                        prepared_data[section_name]["image_data"] = image_base64
            except Exception as e:
                print(f"Warning: Failed to process image for {section_name}: {str(e)}")

        # NOTE: We no longer include m4a_data here as it's available through the dedicated downloads page

    return prepared_data


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
        # Initialize the section data structure
        fast_audio_segment = read_from_gcs(
            config.GCS_PRIVATE_BUCKET,
            get_fast_audio_path(story_name, story_part, collection=collection),
            "audio",
        )

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


def generate_hierarchical_index_system(
    languages: List[str] = None,
    collections: List[str] = None,
    bucket_name: str = None,
    upload: bool = True,
) -> dict:
    """
    Generate a hierarchical index system: Main Index > Language Index > Collection Index.

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

    results = {"main_index": None, "language_indexes": {}, "collection_indexes": {}}

    # 1. Generate main index (language selector)
    results["main_index"] = generate_main_language_index(
        languages=languages,
        bucket_name=bucket_name,
        upload=upload,
        collections=collections,
    )

    # 2. Generate language-level indexes
    for language in languages:
        results["language_indexes"][language] = generate_language_collection_index(
            language=language,
            collections=collections,
            bucket_name=bucket_name,
            upload=upload,
        )

        # 3. Generate collection-level indexes for this language
        for collection in collections:
            key = f"{language}_{collection}"
            results["collection_indexes"][key] = generate_collection_story_index(
                language=language,
                collection=collection,
                bucket_name=bucket_name,
                upload=upload,
            )

    return results


def generate_main_language_index(
    languages: List[str],
    bucket_name: str,
    upload: bool = True,
    collections: List[str] = None,
) -> str:
    """Generate the main index page that shows available languages."""

    if collections is None:
        collections = ["LM1000"]

    # Get language statistics
    language_cards = ""
    special_pages = get_special_pages_from_bucket(bucket_name)

    for language in languages:
        # Get stats for this language across all collections
        total_stories = 0
        collections_available = []

        # Check which collections exist for this language
        for collection in collections:
            try:
                collection_data = read_from_gcs(
                    config.GCS_PRIVATE_BUCKET,
                    get_story_collection_path(collection),
                    "json",
                )
                if collection_data:
                    collections_available.append(collection)
                    total_stories += len(collection_data)
            except:
                continue

        stats_text = (
            f"{len(collections_available)} collections, {total_stories} stories"
        )
        language_url = f"{language.lower()}/index.html"

        language_cards += f"""
        <div class="language-card">
            <div class="language-name">{language}</div>
            <div class="language-stats">{stats_text}</div>
            <a href="{language_url}" class="language-link">Browse Stories</a>
        </div>
        """

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
        return url

    return None


def generate_language_collection_index(
    language: str,
    collections: List[str],
    bucket_name: str,
    upload: bool = True,
) -> str:
    """Generate a language-level index showing available collections."""

    collection_cards = ""

    for collection in collections:
        try:
            # Get collection data
            collection_data = read_from_gcs(
                config.GCS_PRIVATE_BUCKET, get_story_collection_path(collection), "json"
            )

            story_count = len(collection_data)
            collection_title = get_collection_title(collection)

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
            print(f"Warning: Could not load collection {collection}: {e}")
            continue

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
        return url

    return None


def generate_collection_story_index(
    language: str,
    collection: str,
    bucket_name: str,
    upload: bool = True,
) -> str:
    """Generate a collection-level index showing stories ordered by position."""

    try:
        # Get collection data
        collection_data = read_from_gcs(
            config.GCS_PRIVATE_BUCKET, get_story_collection_path(collection), "json"
        )

        story_cards = ""

        # Generate story cards ordered by position
        for position, (story_name, story_data) in enumerate(collection_data.items(), 1):
            story_title = get_story_title(story_name)
            phrase_count = len(story_data) if isinstance(story_data, list) else 0

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

        story_count = len(collection_data)
        collection_title = get_collection_title(collection)

        # Load and fill template
        template = Template(load_template("collection_index_template.html"))
        html_content = template.substitute(
            collection_title=collection_title,
            language_name=language,
            story_count=story_count,
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
            return url

        return None

    except Exception as e:
        print(f"Error generating collection index for {language}/{collection}: {e}")
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


def update_all_index_pages_hierarchical(
    languages: List[str] = None,
    collections: List[str] = None,
    bucket_name: str = None,
    force_upload: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Update all index pages using the new hierarchical system (Language > Collection > Story).

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
        collections = ["LM1000"]

    if verbose:
        print(f"Starting hierarchical index generation for bucket: {bucket_name}")

    results = {}

    try:
        # Generate hierarchical index system
        if verbose:
            print("Generating hierarchical index system...")

        hierarchical_results = generate_hierarchical_index_system(
            languages=languages,
            collections=collections,
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
