import io
import json
import mimetypes
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from google.cloud import storage
from PIL import Image
from pydub import AudioSegment

from src.config_loader import config


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
    save_local: bool = True,
    local_base_dir: str = "../outputs/gcs",
) -> str:
    """
    Upload various file types directly to Google Cloud Storage without writing to local disk.

    Args:
        obj: The object to upload (bytes, dict, str, PIL Image, AudioSegment, etc.)
        bucket_name: Name of the GCS bucket
        file_name: Name of the file to upload
        base_prefix: Prefix/folder path in the bucket. Defaults to ''.
        content_type: MIME content type. If None, will be inferred.
        save_local: Whether to save a local copy of the file (default: True)
        local_base_dir: Base directory for local GCS mirror (default: "../outputs/gcs")

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
        if save_local:
            local_path = os.path.join(local_base_dir, bucket_name, full_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(obj)

    elif isinstance(obj, str):
        # Handle string content (including HTML)
        if file_name.lower().endswith((".html", ".htm")):
            # For HTML files
            if content_type is None:
                content_type = "text/html"
        elif file_name.lower().endswith(".css"):
            # For CSS files
            if content_type is None:
                content_type = "text/css"
        elif file_name.lower().endswith(".js"):
            # For JavaScript files
            if content_type is None:
                content_type = "application/javascript"
        elif file_name.lower().endswith(".txt"):
            # For plain text files
            if content_type is None:
                content_type = "text/plain"

        # Upload the string directly
        blob.upload_from_string(obj, content_type=content_type)
        if save_local:
            local_path = os.path.join(local_base_dir, bucket_name, full_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(obj)

    elif isinstance(obj, dict):
        # JSON object upload
        json_str = json.dumps(obj)
        blob.upload_from_string(json_str, content_type="application/json")
        if save_local:
            local_path = os.path.join(local_base_dir, bucket_name, full_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(json_str)

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
        if save_local:
            local_path = os.path.join(local_base_dir, bucket_name, full_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(buffer.getvalue())

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
            if save_local:
                local_path = os.path.join(local_base_dir, bucket_name, full_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(buffer.getvalue())
        except Exception as e:
            raise ValueError(f"Failed to save image: {e}")

    elif hasattr(obj, "read"):  # For file-like objects
        blob.upload_from_file(obj, content_type=content_type)
        if save_local:
            local_path = os.path.join(local_base_dir, bucket_name, full_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(obj.read())

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
    bucket_name: str,
    file_path: str,
    expected_type: Optional[str] = None,
    use_local: bool = True,
    local_base_dir: str = "../outputs/gcs",
) -> Any:
    """
    Download a file from Google Cloud Storage or read from local cache if available.

    Args:
        bucket_name: Name of the GCS bucket
        file_path: Path to the file within the bucket
        expected_type: Optional type hint ('audio', 'image', 'json', 'bytes', 'text')
        use_local: Whether to check for a local copy first
        local_base_dir: Base directory for local GCS mirror

    Returns:
        The file content as an appropriate Python object
    """
    # Check if we should try local file first
    if use_local:
        local_path = os.path.join(local_base_dir, bucket_name, file_path)
        if os.path.exists(local_path):
            try:
                # Handle different file types
                if expected_type == "audio":
                    return AudioSegment.from_file(local_path)
                elif expected_type == "image":
                    return Image.open(local_path)
                elif expected_type == "json":
                    with open(local_path, "r") as f:
                        return json.load(f)
                elif expected_type == "text":
                    with open(local_path, "r") as f:
                        return f.read()
                else:  # Default to bytes
                    with open(local_path, "rb") as f:
                        return f.read()
            except Exception as e:
                print(f"Error reading local file {local_path}: {str(e)}")
                # Fall back to GCS if local read fails

    # Original GCS implementation
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Check if file exists
    if not blob.exists():
        raise FileNotFoundError(f"File not found in GCS: {bucket_name}/{file_path}")

    # Download file into memory
    content = blob.download_as_bytes()

    # If use_local is enabled, save a local copy for future use
    if use_local:
        local_path = os.path.join(local_base_dir, bucket_name, file_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)

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
        save_local=True,
    )

    # Convert GCS URI to public URL
    # Format: gs://bucket-name/path/file.html -> https://storage.googleapis.com/bucket-name/path/file.html
    public_url = gcs_uri.replace("gs://", "https://storage.googleapis.com/")

    return public_url


def get_story_collection_path(collection: str = "LM1000") -> str:
    """Get the GCS path for a story collection file."""
    return f"collections/{collection}/{collection}.json"


def get_story_challenges_path(story_name: str, collection: str = "LM1000") -> str:
    """Get the GCS path for a story's challenges JSON file (dictionary of scenarios).
    audio-language-trainer-private-content/
    collections/
    LM1000/
    stories/
    story_a_fishing_trip/
    challenges.json"""

    return f"collections/{collection}/stories/{story_name}/challenges.json"


def get_story_translated_challenges_path(story_name: str) -> str:
    """Get the GCS path for a story's challenges webpage
    e.g: audio-language-trainer-stories/swedish/story_birthday_party_planning_mishap/challenges.html
    """
    return f"{config.TARGET_LANGUAGE_NAME.lower()}/{story_name}/challenges.html"


def get_m4a_file_path(story_name: str, story_part: str, fast: bool = False) -> str:
    """Get the GCS path for a story part's m4a file. Each story part has its own m4a file."""
    language = config.TARGET_LANGUAGE_NAME.lower()
    story_part = sanitize_path_component(story_part)
    if fast:
        filename = get_m4a_filename(story_name, story_part, fast=True)
        return f"{language}/{story_name}/{filename}"
    else:
        filename = get_m4a_filename(story_name, story_part, fast=False)
        return f"{language}/{story_name}/{filename}"


def get_m4a_blob_prefix(story_name: str) -> str:
    """Get the GCS bucket prefix for a story part's m4a file.
    Each story part has its own m4a file.

    the pattern is bucket_name/blob_prefix/filename.m4a.
    Such that get_m4a_file_path = get_m4a_blob_prefix + get_m4a_filename."""

    language = config.TARGET_LANGUAGE_NAME.lower()

    return f"{language}/{story_name}/"


def get_m4a_filename(story_name: str, story_part: str, fast: bool = False) -> str:
    """Get the GCS path for a story part's m4a file. Each story part has its own m4a file."""
    language = config.TARGET_LANGUAGE_NAME.lower()
    story_part = sanitize_path_component(story_part)
    if fast:
        return f"{language}_{story_name}_{story_part}_FAST.m4a"
    else:
        return f"{language}_{story_name}_{story_part}.m4a"


def get_story_dialogue_path(story_name: str, collection: str = "LM1000") -> str:
    """Get the GCS path for a story's dialogue file (initial English only)."""
    return f"collections/{collection}/stories/{story_name}/dialogue.json"


def get_public_story_path(story_name: str) -> str:
    """Get the GCS blob path for a story's public HTML file.
    Meant to go to the GCS public bucket which is for holding stories."""

    language = config.TARGET_LANGUAGE_NAME
    language_folder = sanitize_path_component(language.lower())
    story_folder = sanitize_path_component(story_name)
    blob_path = f"{language_folder}/{story_folder}/{story_name}.html"
    return blob_path


def get_story_translated_dialogue_path(
    story_name: str, collection: str = "LM1000"
) -> str:

    language = config.TARGET_LANGUAGE_NAME.lower()
    """Get the GCS path for a story's translated dialogue file."""
    return f"collections/{collection}/stories/{story_name}/dialogue/{language}/translated_dialogue.json"


def get_translated_phrases_path(collection: str = "LM1000") -> str:
    """Get the GCS path for a story's translated phrases file. These are dictionaries keyed of the phrase key
    and contain translations and wiktionary links."""
    language = config.TARGET_LANGUAGE_NAME.lower()
    return f"collections/{collection}/translations/{language}.json"


def get_wiktionary_cache_path() -> str:
    """Get the GCS path for the Wiktionary link cache. The cache is a JSON dictionary of words and their links.
    The key is the lowercase word, and the value is a link (str)."""

    WORD_LINK_CACHE = f"resources/{config.TARGET_LANGUAGE_NAME.lower()}/{config.TARGET_LANGUAGE_NAME.lower()}_wiktionary_cache.json"
    return WORD_LINK_CACHE


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
    story_name: str, story_part: str, collection: str = "LM1000"
) -> str:
    language = config.TARGET_LANGUAGE_NAME.lower()
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


def get_phrase_audio_path(
    phrase_key: str, speed: Literal["normal", "slow"] = "normal"
) -> str:
    """
    Get the GCS path for a phrase's audio file.

    Args:
        phrase_key: Key identifying the phrase
        speed: "normal" or "slow" speed version

    Returns:
        str: Path to the audio file in GCS
    """
    language = config.TARGET_LANGUAGE_NAME.lower()
    return f"multimedia/audio/phrases/{language}/{speed}/{phrase_key}.mp3"


def get_phrase_image_path(phrase_key: str) -> str:
    """
    Get the GCS path for a phrase's image file.

    Args:
        phrase_key: Key identifying the phrase

    Returns:
        str: Path to the image file in GCS
    """
    return f"multimedia/images/core/{phrase_key}.png"


def get_anki_deck_path(
    story_name: str, collection: str = "LM1000", language: Optional[str] = None
) -> str:
    """
    Get the GCS path for an Anki deck file.

    Args:
        story_name: Name of the story
        collection: Collection name (default: "LM1000")
        language: Optional language code (defaults to config.TARGET_LANGUAGE_NAME)

    Returns:
        str: Path to the Anki deck file in GCS
        Format: collections/{collection}/anki/{language}/{story_name}.apkg
    """
    if language is None:
        language = config.TARGET_LANGUAGE_NAME.lower()

    # Sanitize the story name for use in paths
    sanitized_story = sanitize_path_component(story_name)

    return f"collections/{collection}/anki/{language}/{sanitized_story}.apkg"
