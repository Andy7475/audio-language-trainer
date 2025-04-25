import io
import json
import mimetypes
import os
import re
from collections import defaultdict
from pathlib import Path
from string import Template
from typing import Any, Dict, Optional, List

from google.cloud import storage
from PIL import Image
from pydub import AudioSegment

from src.config_loader import config
from src.story import (
    generate_language_section,
    generate_special_pages_section,
    generate_m4a_index_html,
)
from src.utils import clean_story_name, load_template


def sanitize_path_component(s: str) -> str:
    """
    Sanitize a string for use in GCS paths.
    Replaces spaces with underscores, removes special characters, and converts to lowercase.
    """
    # Replace one or more spaces with a single underscore
    s = re.sub(r"\s+", "_", s)
    # Remove any characters that aren't alphanumeric, underscore, or hyphen
    s = "".join(c for c in s if c.isalnum() or c in "_-")
    # Convert to lowercase for consistency
    return s.lower()


def upload_to_gcs(
    obj: Any,
    bucket_name: str,
    file_name: str,
    base_prefix: str = "",
    content_type: Optional[str] = None,
) -> str:
    """
    Upload various file types directly to Google Cloud Storage without writing to local disk.

    Args:
        obj: The object to upload (bytes, dict, PIL Image, AudioSegment, etc.)
        bucket_name: Name of the GCS bucket
        file_name: Name of the file to upload
        base_prefix: Prefix/folder path in the bucket. Defaults to ''.
        content_type: MIME content type. If None, will be inferred.

    Returns:
        GCS URI of the uploaded file

    Raises:
        ValueError: If unsupported object type is provided
    """
    # Create a client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Construct full blob path
    full_path = f"{base_prefix.rstrip('/')}/{file_name}".lstrip("/")
    blob = bucket.blob(full_path)

    # Determine content type if not provided
    if content_type is None:
        # Infer content type from file extension
        content_type, _ = mimetypes.guess_type(file_name)

    # Handle different object types
    if isinstance(obj, bytes):
        # Direct bytes upload
        blob.upload_from_string(obj, content_type=content_type)

    elif isinstance(obj, dict):
        # JSON object upload
        json_str = json.dumps(obj)
        blob.upload_from_string(json_str, content_type="application/json")

    elif str(type(obj)).endswith("AudioSegment'>"):  # For pydub AudioSegment
        buffer = io.BytesIO()
        format_name = file_name.split(".")[-1].lower()

        # Default to mp3 if format can't be determined
        if not format_name or format_name not in ["mp3", "m4a", "wav", "ogg"]:
            format_name = "mp3"

        if format_name == "m4a":
            format_name = "ipod"  # pydub uses 'ipod' for m4a format

        obj.export(buffer, format=format_name)
        buffer.seek(0)

        # Set content type based on format
        format_content_types = {
            "mp3": "audio/mpeg",
            "ipod": "audio/mp4",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
        }
        audio_content_type = format_content_types.get(format_name, "audio/mpeg")

        blob.upload_from_file(buffer, content_type=content_type or audio_content_type)

    elif hasattr(obj, "save") and hasattr(obj, "mode"):  # For PIL Image
        # Get format from filename or default to PNG
        try:
            format_name = file_name.split(".")[-1].upper()
            if format_name not in ["PNG", "JPEG", "JPG", "GIF", "WEBP", "BMP"]:
                format_name = "PNG"

            # Normalize JPEG format name
            if format_name == "JPG":
                format_name = "JPEG"

            buffer = io.BytesIO()
            obj.save(buffer, format=format_name)
            buffer.seek(0)

            # Set appropriate content type
            image_content_types = {
                "PNG": "image/png",
                "JPEG": "image/jpeg",
                "GIF": "image/gif",
                "WEBP": "image/webp",
                "BMP": "image/bmp",
            }
            img_content_type = image_content_types.get(format_name, "image/png")

            blob.upload_from_file(buffer, content_type=content_type or img_content_type)
        except Exception as e:
            raise ValueError(f"Failed to save image: {e}")

    elif hasattr(obj, "read"):  # For file-like objects
        blob.upload_from_file(obj, content_type=content_type)

    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")

    # Return the GCS URI
    return f"gs://{bucket_name}/{full_path}"


def check_blob_exists(bucket_name: str, blob_path: str) -> bool:
    """
    Check if a blob exists in a GCS bucket.

    Args:
        bucket_name: Name of the GCS bucket
        blob_path: Path to the blob within the bucket

    Returns:
        True if the blob exists, False otherwise
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.exists()


def read_from_gcs(
    bucket_name: str, file_path: str, expected_type: Optional[str] = None
) -> Any:
    """
    Download a file from Google Cloud Storage and return it as the appropriate object type.

    Args:
        bucket_name: Name of the GCS bucket
        file_path: Path to the file within the bucket
        expected_type: Optional type hint to force a specific return type
                      ('audio', 'image', 'json', 'bytes', 'text')

    Returns:
        The file content as an appropriate Python object:
        - AudioSegment for audio files (.mp3, .m4a, .wav, .ogg)
        - PIL.Image for image files (.png, .jpg, .jpeg, .gif, .webp)
        - dict for JSON files (.json)
        - bytes for binary files (if type cannot be determined)
        - str for text files (.txt, .html, .css, .csv)

    Raises:
        FileNotFoundError: If the file doesn't exist in the bucket
        ValueError: If the file type is unsupported or there's an error processing the file
    """
    # Initialize storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Check if file exists
    if not blob.exists():
        raise FileNotFoundError(f"File not found in GCS: {bucket_name}/{file_path}")

    # Download file into memory
    content = blob.download_as_bytes()

    # Determine file type from extension if not explicitly provided
    if expected_type is None:
        file_extension = file_path.lower().split(".")[-1]

        # Map extension to type
        if file_extension in ["mp3", "wav", "ogg"]:
            expected_type = "audio"
        elif file_extension == "m4a":
            expected_type = "audio"
        elif file_extension in ["png", "jpg", "jpeg", "gif", "webp", "bmp"]:
            expected_type = "image"
        elif file_extension == "json":
            expected_type = "json"
        elif file_extension in ["txt", "html", "css", "js", "csv"]:
            expected_type = "text"
        else:
            expected_type = "bytes"

    # Convert to the appropriate type
    try:
        if expected_type == "audio":
            buffer = io.BytesIO(content)
            return AudioSegment.from_file(buffer)

        elif expected_type == "image":
            buffer = io.BytesIO(content)
            return Image.open(buffer)

        elif expected_type == "json":
            return json.loads(content)

        elif expected_type == "text":
            return content.decode("utf-8")

        else:  # Default to bytes
            return content

    except Exception as e:
        raise ValueError(
            f"Error processing file {file_path} as {expected_type}: {str(e)}"
        )


def upload_story_to_gcs(html_file_path: str, bucket_name: Optional[str] = None) -> str:
    """
    Upload a story HTML file to GCS using the standard language/story folder structure.

    Args:
        html_file_path: Path to the story HTML file
        bucket_name: Optional bucket name (defaults to config.GCS_PUBLIC_BUCKET)

    Returns:
        str: Public URL of the uploaded file

    Raises:
        FileNotFoundError: If the HTML file doesn't exist
    """
    if not os.path.exists(html_file_path):
        raise FileNotFoundError(f"HTML file not found: {html_file_path}")

    if bucket_name is None:
        bucket_name = config.GCS_PUBLIC_BUCKET

    language_name = config.TARGET_LANGUAGE_NAME

    # Get story name for bucket prefix
    story_name = Path(html_file_path).stem
    file_name = Path(html_file_path).name
    language_folder = sanitize_path_component(language_name.lower())
    base_prefix = f"{language_folder}/{story_name}"

    # Read the HTML file content
    with open(html_file_path, "rb") as f:
        html_content = f.read()

    # Upload the content using the new upload_to_gcs function
    gcs_uri = upload_to_gcs(
        obj=html_content,
        bucket_name=bucket_name,
        file_name=file_name,
        base_prefix=base_prefix,
        content_type="text/html",
    )

    # Convert GCS URI to public URL
    # Format: gs://bucket-name/path/file.html -> https://storage.googleapis.com/bucket-name/path/file.html
    public_url = gcs_uri.replace("gs://", "https://storage.googleapis.com/")

    return public_url


def create_and_upload_html_story(
    prepared_data: Dict,
    story_name: str,
    bucket_name: str = config.GCS_PUBLIC_BUCKET,
    component_path: str = "StoryViewer.js",
    template_path: str = "story_template.html",
    output_dir: str = "outputs/stories/",
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
    story_title = clean_story_name(story_name)

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

    # Create local file path for temporary storage
    local_dir = Path(output_dir) / story_name / language
    local_dir.mkdir(parents=True, exist_ok=True)
    local_html_path = local_dir / f"{story_name}.html"

    # Write the HTML file locally
    local_html_path.write_text(html_content, encoding="utf-8")
    print(f"HTML story created locally at: {local_html_path}")

    # Upload to GCS
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Create the GCS blob path
        language_folder = sanitize_path_component(language.lower())
        story_folder = sanitize_path_component(story_name)
        blob_path = f"{language_folder}/{story_folder}/{story_name}.html"

        # Create the blob and upload
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(local_html_path), content_type="text/html")

        # Generate the public URL
        public_url = f"https://storage.googleapis.com/{bucket_name}/{blob_path}"
        print(f"Uploaded HTML story to: {public_url}")

        return public_url

    except Exception as e:
        print(f"Error uploading to GCS: {str(e)}")
        return str(local_html_path)  # Return local path if upload fails


def get_story_collection_path(collection: str = "LM1000") -> str:
    """Get the GCS path for a story collection file."""
    return f"collections/{collection}/{collection}.json"


def get_story_dialogue_path(
    story_name: str, language: str, collection: str = "LM1000"
) -> str:
    """Get the GCS path for a story's translated dialogue file."""
    return f"collections/{collection}/stories/{story_name}/dialogue/{language}/translated_dialogue.json"


def get_utterance_audio_path(
    story_name: str,
    story_part: str,
    index: int,
    speaker: str,
    language: str,
    collection: str = "LM1000",
) -> str:
    """Get the GCS path for an utterance audio file."""
    filename = f"part_{index}_{speaker.lower()}.mp3"
    return f"collections/{collection}/stories/{story_name}/audio/{language}/{story_part}/{filename}"


def get_fast_audio_path(
    story_name: str, story_part: str, language: str, collection: str = "LM1000"
) -> str:
    """Get the GCS path for a fast audio file."""
    return f"collections/{collection}/stories/{story_name}/audio/{language}/{story_part}/fast.mp3"


def get_image_path(story_name: str, story_part: str, collection: str = "LM1000") -> str:
    """Get the GCS path for a story part image."""
    return f"collections/{collection}/stories/{story_name}/images/{story_part}.png"


def process_bucket_contents(bucket_name: str, exclude_patterns: list = None) -> tuple:
    """
    Process bucket contents - extract html files to populate the story index, excluding specified patterns.

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


def get_stories_from_collection(
    bucket_name: str = config.GCS_PRIVATE_BUCKET, collection: str = "LM1000"
) -> List[str]:
    """
    Get list of story names from a collection file.

    Args:
        bucket_name: GCS bucket name
        collection: Collection name

    Returns:
        List[str]: List of story names
    """
    collection_path = get_story_collection_path(collection)
    try:
        collection_data = read_from_gcs(bucket_name, collection_path, "json")
        # Assuming the collection file has story names as keys
        return list(collection_data.keys())
    except Exception as e:
        print(f"Error loading collection {collection}: {str(e)}")
        return []


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
