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
    process_bucket_contents,
    read_from_gcs,
    upload_to_gcs,
)
from src.utils import load_template, get_story_position
from src.wiktionary import generate_wiktionary_links


def generate_and_upload_m4a_index(bucket_name=None, output_dir="../outputs/stories"):
    """
    Generate the M4A index page and upload it to Google Cloud Storage.

    Args:
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PUBLIC_BUCKET)
        output_dir: Directory where the HTML file will be saved locally

    Returns:
        str: Public URL of the uploaded file
    """
    # First generate the index
    local_path = generate_m4a_index_html(bucket_name, output_dir)

    # Initialize storage client
    storage_client = storage.Client()

    # Get bucket
    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET
    bucket = storage_client.bucket(bucket_name)

    # Upload directly to the root of the bucket
    blob = bucket.blob("m4a_downloads.html")
    blob.upload_from_filename(local_path, content_type="text/html")

    print("M4A index uploaded to GCS: m4a_downloads.html")

    # Return the public URL
    return f"https://storage.googleapis.com/{bucket_name}/m4a_downloads.html"


def update_all_index_pages(
    output_dir: str = "../outputs/stories",
    bucket_name: str = None,
    force_upload: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Update all index pages for the language learning platform.

    This function generates and uploads:
    - Main story index (index.html)
    - Audio downloads index (m4a_downloads.html)

    Args:
        output_dir: Directory where HTML files will be saved locally
        bucket_name: Name of the GCS bucket (defaults to config.GCS_PUBLIC_BUCKET)
        force_upload: Whether to upload generated files even if they exist
        verbose: Whether to print detailed progress information

    Returns:
        dict: Dictionary with paths and URLs for all generated index pages
    """
    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET

    if verbose:
        print(f"Starting index page updates for bucket: {bucket_name}")

    results = {}

    try:
        # Generate and update main and M4A indices
        if verbose:
            print("Generating main index and M4A downloads index...")

        main_path, m4a_path, main_url, m4a_url = generate_and_update_index_html(
            output_dir=output_dir, bucket_name=bucket_name, upload=force_upload
        )

        results.update(
            {
                "main_index": {"local_path": main_path, "url": main_url},
                "m4a_index": {"local_path": m4a_path, "url": m4a_url},
            }
        )

        if verbose:
            print(f"✅ Main index updated: {main_url}")
            print(f"✅ M4A downloads index updated: {m4a_url}")

        # Check if either URL is None (indicating upload failure)
        if force_upload and (main_url is None or m4a_url is None):
            print("⚠️ Warning: Upload was requested but one or more URLs are missing.")

        if verbose:
            print("All index pages updated successfully.")

        return results

    except Exception as e:
        error_msg = f"Error updating index pages: {str(e)}"
        print(f"❌ {error_msg}")

        # Try to include as much information as possible despite the error
        if "main_path" in locals():
            results["main_index"] = {"local_path": main_path, "url": None}
        if "m4a_path" in locals():
            results["m4a_index"] = {"local_path": m4a_path, "url": None}

        results["error"] = error_msg
        return results


def generate_and_update_index_html(
    output_dir: str = "../outputs/gcs",
    bucket_name: str = None,
    template_path: str = "index_template.html",
    m4a_template_path: str = "m4a_index_template.html",  # Kept for compatibility, but not used
    upload: bool = True,
) -> tuple:
    """
    Generate index.html file from GCS bucket contents and upload it.

    Args:
        output_dir: Directory where the HTML file will be saved locally (defaults to "../outputs/gcs")
        bucket_name: Name of the GCS bucket containing stories (defaults to config.GCS_PUBLIC_BUCKET)
        template_path: Path to the main index HTML template file
        m4a_template_path: (ignored)
        upload: Whether to upload the generated file to GCS

    Returns:
        tuple: (main_index_path, None, main_index_url, None)
    """
    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate main index.html
    # Process bucket contents
    stories_by_language, special_pages = process_bucket_contents(
        bucket_name,
        exclude_patterns=["challenges.html", "m4a_downloads.html"],
    )

    # Remove M4A downloads link from special pages if present
    special_pages = [
        page for page in special_pages
        if not (page["name"].lower().startswith("audio downloads") or "m4a_downloads.html" in page.get("url", ""))
    ]

    # Generate sections HTML
    language_sections = ""
    for language, stories in sorted(stories_by_language.items()):
        language_sections += generate_language_section(language, stories)

    # Generate special pages HTML
    special_pages_html = generate_special_pages_section(special_pages)

    # Add sticky banner HTML
    sticky_banner = """
    <div class="sticky-banner" style="position: fixed; top: 0; left: 0; right: 0; background-color: #FB9A4B; color: #3a3e41; text-align: center; padding: 10px; z-index: 1000; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
        <a href="https://www.firephrase.co.uk" style="text-decoration: none; font-weight: bold; font-size: 1.1em;">
            Get more flashcard decks at FirePhrase.co.uk
        </a>
    </div>
    <div style="margin-top: 50px;"> <!-- Add margin to prevent content from being hidden under the banner -->
    """

    # Load and fill template
    template = Template(load_template(template_path))
    html_content = template.substitute(
        language_sections=sticky_banner + language_sections,
        special_pages=special_pages_html,
    )

    # Write to file
    main_index_path = os.path.join(output_dir, "index.html")
    with open(main_index_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Upload file to GCS if requested
    main_index_url = None
    if upload:
        main_index_url = upload_to_gcs(
            obj=html_content,
            bucket_name=bucket_name,
            file_name="index.html",
            content_type="text/html",
        )
        print(f"Main index uploaded to: {main_index_url}")

    return (main_index_path, None, main_index_url, None)


def create_and_upload_html_story(
    prepared_data: Dict,
    story_name: str,
    bucket_name: str = config.GCS_PUBLIC_BUCKET,
    component_path: str = "StoryViewer.js",
    template_path: str = "story_template.html",
    output_dir: str = "../outputs/stories/",
) -> str:
    """
    Create a standalone HTML file from prepared story data and upload it to GCS.

    Args:
        prepared_data: Dictionary containing prepared story data with base64 encoded assets
        story_name: Name of the story
        bucket_name: GCS bucket name for upload
        language: Target language name (defaults to config.TARGET_LANGUAGE_NAME)
        component_path: Path to the React component file
        template_path: Path to the HTML template file
        output_dir: Local directory to save HTML file before upload

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
    )

    try:

        blob_path = get_public_story_path(story_name)

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

    # Substitute the template variables
    html_content = template.substitute(
        title=story_title,
        story_data=json.dumps(prepared_data),
        language=language,
        react_component=react_component,
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
    """
    Process dialogue utterances to include Wiktionary links for the target language text.
    We will try and read the existing wiktionary links cache from GCS first. As this
    will probably already have data in it from processing the flashcard phrases.
    Args:
        story_data_dict: Dictionary of story_part :dialogue : [list of utternaces]
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
                section_data.get("translated_dialogue", [])
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


def generate_m4a_index_html(
    bucket_name: str = None,
    output_dir: str = "../outputs/stories",
    template_path: str = "m4a_index_template.html",
) -> str:
    """
    Generate an index.html file for M4A downloads organized by language and story.

    Args:
        bucket_name: GCS bucket containing M4A files
        output_dir: Where to save the generated HTML
        template_path: Path to the HTML template

    Returns:
        str: Path to the generated index file
    """
    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET

    # Initialize storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Get all M4A files in the bucket
    m4a_files = defaultdict(lambda: defaultdict(list))

    for blob in bucket.list_blobs():
        if blob.name.endswith(".m4a"):
            parts = blob.name.split("/")

            # Expected format: language/story_name/story_name_part.m4a
            if len(parts) >= 3:
                language = parts[0].capitalize()
                story_name = get_story_title(parts[1])

                # Get file size in MB
                size_mb = blob.size / (1024 * 1024)

                m4a_files[language][story_name].append(
                    {
                        "name": parts[-1],
                        "url": f"https://storage.googleapis.com/{bucket_name}/{blob.name}",
                        "size": f"{size_mb:.1f} MB",
                        "size_bytes": blob.size,
                    }
                )

    # Generate HTML content
    languages_html = ""
    total_size = 0
    file_count = 0

    for language, stories in sorted(m4a_files.items()):
        # Important: Use a local variable for language_id, not $languageId
        language_id = language.lower().replace(" ", "_")
        stories_html = ""
        language_size = 0
        language_file_count = 0

        for story, files in sorted(stories.items()):
            # Create story_id for HTML IDs
            story_id = f"{language_id}_{story.lower().replace(' ', '_')}"
            files_html = ""
            story_size = 0

            for file_info in files:
                file_id = f"{story_id}_{file_info['name'].replace('.', '_')}"
                files_html += f"""
                <div class="file-item">
                    <label class="flex items-center space-x-2">
                        <input type="checkbox" id="{file_id}" 
                               data-url="{file_info['url']}" 
                               data-size="{file_info['size_bytes']}"
                               data-name="{file_info['name']}"
                               class="file-checkbox">
                        <span>{file_info['name']}</span>
                        <span class="text-gray-500 text-sm">{file_info['size']}</span>
                    </label>
                </div>
                """
                story_size += file_info["size_bytes"]
                language_size += file_info["size_bytes"]
                total_size += file_info["size_bytes"]
                language_file_count += 1
                file_count += 1

            story_size_mb = story_size / (1024 * 1024)

            stories_html += f"""
            <div class="story-section mb-4">
                <div class="story-header bg-gray-100 p-2 rounded flex items-center">
                    <label class="flex items-center space-x-2 flex-grow">
                        <input type="checkbox" id="{story_id}_all" class="story-checkbox">
                        <span class="font-medium">{story}</span>
                        <span class="text-gray-500 text-sm">({len(files)} files, {story_size_mb:.1f} MB)</span>
                    </label>
                    <button class="toggle-btn px-2" data-target="{story_id}_files">▼</button>
                </div>
                <div id="{story_id}_files" class="story-files pl-6 pt-2">
                    {files_html}
                </div>
            </div>
            """

        language_size_mb = language_size / (1024 * 1024)

        languages_html += f"""
        <div class="language-section mb-6">
            <div class="language-header bg-blue-100 p-3 rounded flex items-center">
                <label class="flex items-center space-x-2 flex-grow">
                    <input type="checkbox" id="{language_id}_all" class="language-checkbox">
                    <span class="font-medium text-lg">{language}</span>
                    <span class="text-gray-600">
                        ({len(stories)} stories, {language_file_count} files, {language_size_mb:.1f} MB)
                    </span>
                </label>
                <button class="toggle-btn px-2" data-target="{language_id}_stories">▼</button>
            </div>
            <div id="{language_id}_stories" class="language-stories pl-6 pt-3">
                {stories_html}
            </div>
        </div>
        """

    total_size_mb = total_size / (1024 * 1024)

    # Load template content (make sure this loads the exact template you pasted)
    template_content = load_template(template_path)
    template = Template(template_content)

    # Only pass the variables that exist in the template
    template_vars = {
        "language_sections": languages_html,
        "file_count": file_count,
        "total_size": f"{total_size_mb:.1f}",
    }

    try:
        # Try to substitute with only the variables we know are in the template
        html_content = template.safe_substitute(**template_vars)
    except KeyError as e:
        # If an error occurs, print helpful debugging information
        print(f"KeyError: {e} not found in template variables")
        print(f"Template variables provided: {list(template_vars.keys())}")
        print(f"Check if {e} is used in your template but missing from the variables")
        raise

    # Write to file
    output_path = os.path.join(output_dir, "m4a_downloads.html")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


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
