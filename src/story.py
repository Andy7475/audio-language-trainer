import base64
import io
import json
import os
from string import Template
from typing import Dict, Optional

from PIL import Image
from pydub import AudioSegment
from tqdm import tqdm

from src.anki_tools import generate_wiktionary_links, load_template
from src.audio_generation import create_m4a_with_timed_lyrics
from src.config_loader import config

from collections import defaultdict
from pathlib import Path
from string import Template
from typing import Dict, List, Tuple

from google.cloud import storage


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


def process_bucket_contents(bucket_name: str, exclude_patterns: list = None) -> tuple:
    """
    Process bucket contents, excluding specified patterns.

    Args:
        bucket_name: Name of the GCS bucket
        exclude_patterns: List of patterns to exclude (e.g., ['challenges.html'])

    Returns:
        tuple: (stories_by_language, special_pages)
    """
    if exclude_patterns is None:
        exclude_patterns = []

    stories_by_language = defaultdict(list)
    special_pages = []

    # default to public bucket
    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    for blob in blobs:
        if any(pattern in blob.name for pattern in exclude_patterns):
            continue
        path = Path(blob.name)
        parts = path.parts

        # Skip non-HTML files
        if not blob.name.endswith(".html"):
            continue

        # Handle special pages at root level
        if len(parts) == 1:
            if parts[0] != "index.html":
                special_pages.append(
                    {
                        "name": parts[0].replace(".html", "").replace("_", " ").title(),
                        "url": f"https://storage.googleapis.com/{bucket_name}/{blob.name}",
                    }
                )
            continue

        # Process story pages
        if len(parts) >= 3 and parts[1].startswith("story_"):
            language = parts[0].title()
            story_name = parts[1].replace("story_", "").replace("_", " ").title()
            url = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"

            stories_by_language[language].append({"name": story_name, "url": url})

    return dict(stories_by_language), special_pages


def generate_and_update_index_html(
    output_dir: str = "../outputs/stories",
    bucket_name: str = None,
    template_path: str = "index_template.html",
    m4a_template_path: str = "m4a_index_template.html",
    upload: bool = True,
) -> tuple:
    """
    Generate index.html and m4a_downloads.html files from GCS bucket contents and upload them.

    Args:
        output_dir: Directory where the HTML files will be saved locally
        bucket_name: Name of the GCS bucket containing stories (defaults to config.GCS_PUBLIC_BUCKET)
        template_path: Path to the main index HTML template file
        m4a_template_path: Path to the M4A index HTML template file
        upload: Whether to upload the generated files to GCS

    Returns:
        tuple: (main_index_path, m4a_index_path, main_index_url, m4a_index_url)
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

    # Add M4A downloads link to special pages
    special_pages.append(
        {
            "name": "Audio Downloads",
            "url": f"https://storage.googleapis.com/{bucket_name}/m4a_downloads.html",
        }
    )

    # Generate sections HTML
    language_sections = ""
    for language, stories in sorted(stories_by_language.items()):
        language_sections += generate_language_section(language, stories)

    # Generate special pages HTML
    special_pages_html = generate_special_pages_section(special_pages)

    # Load and fill template
    template = Template(load_template(template_path))
    html_content = template.substitute(
        language_sections=language_sections, special_pages=special_pages_html
    )

    # Write to file
    main_index_path = os.path.join(output_dir, "index.html")
    with open(main_index_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # 2. Generate M4A index
    m4a_index_path = generate_m4a_index_html(
        bucket_name=bucket_name, output_dir=output_dir, template_path=m4a_template_path
    )

    # 3. Upload files to GCS if requested
    main_index_url = None
    m4a_index_url = None

    if upload:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Upload main index
        main_blob = bucket.blob("index.html")
        main_blob.upload_from_filename(main_index_path, content_type="text/html")
        main_index_url = f"https://storage.googleapis.com/{bucket_name}/index.html"
        print(f"Main index uploaded to: {main_index_url}")

        # Upload M4A index
        m4a_blob = bucket.blob("m4a_downloads.html")
        m4a_blob.upload_from_filename(m4a_index_path, content_type="text/html")
        m4a_index_url = (
            f"https://storage.googleapis.com/{bucket_name}/m4a_downloads.html"
        )
        print(f"M4A downloads index uploaded to: {m4a_index_url}")

    return (main_index_path, m4a_index_path, main_index_url, m4a_index_url)


def add_index_navigation_to_story(story_html_path: str, language: str) -> None:
    """
    Add navigation link back to index.html for a story page.

    Args:
        story_html_path: Path to the story HTML file
        language: Language section to link back to
    """
    with open(story_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Create navigation link HTML
    nav_html = f"""
    <div style="position: fixed; top: 20px; left: 20px; z-index: 1000;">
        <a href="/index.html#{language.lower()}" 
           style="background: white; padding: 10px 20px; border-radius: 5px; 
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-decoration: none; 
                  color: #2980b9; display: inline-flex; align-items: center; gap: 5px;">
            ← Back to {language} Stories
        </a>
    </div>
    """

    # Insert navigation before closing body tag
    if "</body>" in content:
        content = content.replace("</body>", f"{nav_html}</body>")

        with open(story_html_path, "w", encoding="utf-8") as f:
            f.write(content)


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
    story_title = clean_story_name(story_name)

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


def convert_audio_to_base64(audio_segment: AudioSegment) -> str:
    """Convert an AudioSegment to a base64 encoded string."""
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3")
    buffer.seek(0)
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


def convert_m4a_file_to_base64(m4a_file_path: str) -> str:
    """
    Convert an M4A file to a base64 encoded string.

    Args:
        m4a_file_path: Path to the M4A file

    Returns:
        str: Base64 encoded string representation of the M4A file

    Raises:
        FileNotFoundError: If the M4A file doesn't exist
        IOError: If there's an error reading the file
    """
    try:
        with open(m4a_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            return base64.b64encode(audio_bytes).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"M4A file not found at: {m4a_file_path}")
    except IOError as e:
        raise IOError(f"Error reading M4A file: {str(e)}")


def create_album_files(
    story_data_dict: dict,
    cover_image: Image.Image,
    output_dir: str,
    story_name: str,
    upload_to_gcs: bool = True,
):
    """Creates and saves M4A files for the story, with album artwork.
    Optionally uploads files to Google Cloud Storage.
    """
    REPEATS_OF_FAST_DIALOGUE = 10
    PAUSE_TEXT = "---------"
    GAP_BETWEEN_PHRASES = AudioSegment.silent(duration=500)

    ALBUM_NAME = clean_story_name(story_name)
    TOTAL_TRACKS = len(story_data_dict) * 2  # 1 track normal, 1 track fast

    m4a_file_paths = []

    # Create the story tracks for each story section
    for track_number, (story_part, data) in enumerate(
        tqdm(story_data_dict.items(), desc="creating album"), start=1
    ):
        audio_list = []
        captions_list = []

        # Get dialogue texts and audio
        dialogue_list = [utterance["text"] for utterance in data["translated_dialogue"]]
        dialogue_audio_list = data["translated_dialogue_audio"]

        # Initial dialogue at normal speed
        audio_list.append(GAP_BETWEEN_PHRASES)
        captions_list.append(f"{story_part} - Initial Dialogue")

        audio_list.extend(dialogue_audio_list)
        captions_list.extend(dialogue_list)

        # Create M4A file
        m4a_filename = f"{config.TARGET_LANGUAGE_NAME}_{story_name}_{story_part}.m4a"
        m4a_path = os.path.join(output_dir, m4a_filename)

        create_m4a_with_timed_lyrics(
            audio_segments=audio_list,
            phrases=captions_list,
            output_file=m4a_path,
            album_name=ALBUM_NAME,
            track_title=story_part,
            track_number=track_number,
            total_tracks=TOTAL_TRACKS,
            cover_image=cover_image,
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

        # Add fast dialogue (there are 10 repeats in the audio)
        for i in range(REPEATS_OF_FAST_DIALOGUE):
            audio_list.append(data["translated_dialogue_audio_fast"])
            captions_list.append(f"Fast Dialogue - Repetition {i+1}")
            audio_list.append(GAP_BETWEEN_PHRASES)
            captions_list.append(PAUSE_TEXT)

        # Create M4A file
        m4a_filename = (
            f"{config.TARGET_LANGUAGE_NAME}_{story_name}_{story_part}_FAST.m4a"
        )
        m4a_path = os.path.join(output_dir, m4a_filename)

        create_m4a_with_timed_lyrics(
            audio_segments=audio_list,
            phrases=captions_list,
            output_file=m4a_path,
            album_name=ALBUM_NAME,
            track_title=story_part + " (fast)",
            track_number=track_number,
            total_tracks=TOTAL_TRACKS,
            cover_image=cover_image,
        )
        m4a_file_paths.append(m4a_path)
        print(f"Saved M4A file track number {track_number}")

    # Upload files to GCS if required
    if upload_to_gcs:
        client = storage.Client()
        bucket = client.bucket(config.GCS_PUBLIC_BUCKET)

        for m4a_path in tqdm(m4a_file_paths, desc="Uploading M4A files to GCS"):
            # Extract the filename from the path
            filename = os.path.basename(m4a_path)

            # Create a GCS path for the file
            # Format: language/story_name/filename.m4a
            gcs_path = f"{config.TARGET_LANGUAGE_NAME.lower()}/{story_name}/{filename}"

            # Upload the file to GCS
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(m4a_path)

            print(f"Uploaded {filename} to gs://{config.GCS_PUBLIC_BUCKET}/{gcs_path}")

    return m4a_file_paths


def clean_story_name(story_name: str) -> str:
    """
    Clean a story name by removing 'story' and underscores, returning in title case.

    Args:
        story_name: Input story name (e.g. "story_community_park")

    Returns:
        str: Cleaned story name in title case (e.g. "Community Park")

    Example:
        >>> clean_story_name("story_community_park")
        'Community Park'
    """
    # Remove 'story' and split on underscores
    name = story_name.replace("story_", "")
    words = name.split("_")

    # Convert to title case and join with spaces
    return " ".join(word.title() for word in words)


def prepare_dialogue_with_wiktionary(dialogue, language_name: str = None):
    """
    Process dialogue utterances to include Wiktionary links for the target language text.

    Args:
        dialogue: List of dialogue utterances with text and speaker
        language_name: Name of target language for Wiktionary links

    Returns:
        List of processed utterances with added wiktionary_links field
    """
    if language_name is None:
        language_name = config.TARGET_LANGUAGE_NAME
    processed_dialogue = []
    for utterance in dialogue:
        # Create a copy of the utterance to avoid modifying the original
        processed_utterance = utterance.copy()
        # Generate Wiktionary links for the target language text
        wiktionary_links = generate_wiktionary_links(
            utterance["text"], language_name=language_name
        )
        processed_utterance["wiktionary_links"] = wiktionary_links
        processed_dialogue.append(processed_utterance)
    return processed_dialogue


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
                story_name = clean_story_name(parts[1])

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

    print(f"M4A index uploaded to GCS: m4a_downloads.html")

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
