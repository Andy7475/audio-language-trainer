"""GCS storage management with bucket constants and unified file operations.

This module provides:
- Bucket name constants (no config_loader dependency)
- Language-aware path generators using BCP47Language
- Unified upload/download functions with local caching support
"""

import io
import json
import mimetypes
import os
from typing import Any, Literal, Optional, Union

import langcodes
from PIL import Image
from pydub import AudioSegment

from src.models import BCP47Language
from src.connections.gcloud_auth import get_storage_client
from src.logger import logger

# ============================================================================
# BUCKET CONSTANTS
# ============================================================================

PRIVATE_BUCKET = "audio-language-trainer-private-content"
PUBLIC_BUCKET = "audio-language-trainer-stories"
DEFAULT_ENGLISH_LANGUAGE = langcodes.get("en-GB")


# ============================================================================
# URL/PATH CONVERSION HELPERS
# ============================================================================


def gcs_uri_from_file_path(file_path: str, bucket_name: str) -> str:
    """Convert file path to GCS URI.

    Args:
        file_path: Relative path (e.g., 'phrases/fr-FR/audio/flashcard/slow/hello_a3f8d2.mp3')
        bucket_name: GCS bucket name

    Returns:
        str: Full GCS URI (e.g., 'gs://bucket-name/phrases/fr-FR/audio/flashcard/slow/hello_a3f8d2.mp3')
    """
    file_path = file_path.lstrip("/")
    return f"gs://{bucket_name}/{file_path}"


def file_path_from_gcs_uri(gcs_uri: str) -> str:
    """Extract file path from GCS URI.

    Removes the gs://bucket-name/ prefix and returns just the file path.

    Args:
        gcs_uri: Full GCS URI (e.g., 'gs://bucket-name/phrases/fr-FR/audio/flashcard/slow/hello_a3f8d2.mp3')

    Returns:
        str: File path (e.g., 'phrases/fr-FR/audio/flashcard/slow/hello_a3f8d2.mp3')

    Raises:
        ValueError: If URI format is invalid
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    # Parse: gs://bucket-name/file/path
    parts = gcs_uri[5:].split("/", 1)  # Remove 'gs://' and split
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI format: {gcs_uri}")

    return parts[1]


# ============================================================================
# PATH GENERATORS (Language-aware using BCP47Language)
# ============================================================================


def get_phrase_audio_path(
    phrase_hash: str,
    language: Union[str, BCP47Language],
    context: Literal["flashcard", "story"],
    speed: Literal["slow", "normal", "fast"],
) -> str:
    """Get GCS path for phrase audio file.

    Args:
        phrase_hash: Unique phrase identifier (e.g., 'hello_world_a3f8d2')
        language: BCP47Language or language tag string (e.g., 'fr-FR')
        context: Audio context ('flashcard' or 'story')
        speed: Speaking speed ('slow', 'normal', or 'fast')

    Returns:
        str: GCS path (e.g., 'phrases/fr-FR/audio/flashcard/slow/hello_world_a3f8d2.mp3')
    """
    # Handle both BCP47Language objects and strings
    if isinstance(language, str):
        lang_tag = language
    else:
        lang_tag = language.to_tag()

    return f"phrases/{lang_tag}/audio/{context}/{speed}/{phrase_hash}.mp3"


def get_phrase_image_path(
    phrase_hash: str,
    language: Union[str, BCP47Language] = DEFAULT_ENGLISH_LANGUAGE,
) -> str:
    """Get GCS path for phrase image file.

    Defaults to English (en-GB) to ensure a default image always exists.
    All phrases have an English translation with an associated image.

    Args:
        phrase_hash: Unique phrase identifier (e.g., 'hello_world_a3f8d2')
        language: BCP47Language or language tag string (default: en-GB)

    Returns:
        str: GCS path (e.g., 'phrases/en-GB/images/hello_world_a3f8d2.png')
    """
    # Handle both BCP47Language objects and strings
    if isinstance(language, str):
        lang_tag = language
    else:
        lang_tag = language.to_tag()

    return f"phrases/{lang_tag}/images/{phrase_hash}.png"


# ============================================================================
# STORY AND CHALLENGE PATH GENERATORS
# ============================================================================


def get_story_dialogue_path(story_name: str, collection: str = "LM1000") -> str:
    """Get GCS path for story dialogue JSON (English, untranslated).

    Args:
        story_name: Name of the story (e.g., "story_a_fishing_trip")
        collection: Collection name (e.g., "LM1000")

    Returns:
        str: GCS path (e.g., 'stories/LM1000/story_a_fishing_trip/dialogue.json')
    """
    # Ensure story_name has 'story_' prefix
    if not story_name.startswith("story_"):
        story_name = f"story_{story_name}"

    return f"stories/{collection}/{story_name}/dialogue.json"


def get_story_translated_dialogue_path(
    story_name: str, language: Union[str, BCP47Language], collection: str = "LM1000"
) -> str:
    """Get GCS path for translated story dialogue JSON.

    Args:
        story_name: Name of the story (e.g., "story_a_fishing_trip")
        language: BCP47Language or language tag string (e.g., 'fr-FR')
        collection: Collection name (e.g., "LM1000")

    Returns:
        str: GCS path (e.g., 'stories/LM1000/story_a_fishing_trip/fr-FR/dialogue.json')
    """
    # Handle both BCP47Language objects and strings
    if isinstance(language, str):
        lang_tag = language
    else:
        lang_tag = language.to_tag()

    # Ensure story_name has 'story_' prefix
    if not story_name.startswith("story_"):
        story_name = f"story_{story_name}"

    return f"stories/{collection}/{story_name}/{lang_tag}/dialogue.json"


def get_story_collection_path(collection: str = "LM1000") -> str:
    """Get GCS path for story collection metadata JSON.

    Args:
        collection: Collection name (e.g., "LM1000")

    Returns:
        str: GCS path (e.g., 'stories/LM1000/collection.json')
    """
    return f"stories/{collection}/collection.json"


def get_story_challenges_path(
    story_name: str, language: Union[str, BCP47Language], collection: str = "LM1000"
) -> str:
    """Get GCS path for story challenges JSON (scenario data).

    Args:
        story_name: Name of the story (e.g., "story_a_fishing_trip")
        language: BCP47Language or language tag string (e.g., 'fr-FR')
        collection: Collection name (e.g., "LM1000")

    Returns:
        str: GCS path (e.g., 'stories/LM1000/story_a_fishing_trip/fr-FR/challenges.json')
    """
    # Handle both BCP47Language objects and strings
    if isinstance(language, str):
        lang_tag = language
    else:
        lang_tag = language.to_tag()

    # Ensure story_name has 'story_' prefix
    if not story_name.startswith("story_"):
        story_name = f"story_{story_name}"

    return f"stories/{collection}/{story_name}/{lang_tag}/challenges.json"


def get_story_translated_challenges_path(
    story_name: str, language: Union[str, BCP47Language], collection: str = "LM1000"
) -> str:
    """Get GCS path for story challenges HTML page (PUBLIC_BUCKET).

    Uses language tag (e.g., fr-FR) not language name (french) for consistency.

    Args:
        story_name: Name of the story (e.g., "story_a_fishing_trip")
        language: BCP47Language or language tag string (e.g., 'fr-FR')
        collection: Collection name (e.g., "LM1000")

    Returns:
        str: GCS path (e.g., 'fr-fr/lm1000/a_fishing_trip/challenges.html')
    """
    # Handle both BCP47Language objects and strings
    if isinstance(language, str):
        lang_tag = language
    else:
        lang_tag = language.to_tag()

    # Lowercase for URL consistency
    lang_tag_lower = lang_tag.lower()
    collection_lower = collection.lower()

    # Remove 'story_' prefix for folder name
    story_folder = (
        story_name.replace("story_", "")
        if story_name.startswith("story_")
        else story_name
    )

    return f"{lang_tag_lower}/{collection_lower}/{story_folder}/challenges.html"


def get_story_public_path(
    story_name: str, language: Union[str, BCP47Language], collection: str = "LM1000"
) -> str:
    """Get GCS path for public story HTML.

    Uses language tag (e.g., fr-FR) not language name (french).

    Args:
        story_name: Name of the story
        language: BCP47Language or language tag string
        collection: Collection name

    Returns:
        str: GCS path (e.g., 'fr-fr/lm1000/a_fishing_trip/story_a_fishing_trip.html')
    """
    # Handle language parameter
    if isinstance(language, str):
        lang_tag = language
    else:
        lang_tag = language.to_tag()

    # Lowercase for URL consistency
    lang_tag_lower = lang_tag.lower()
    collection_lower = collection.lower()

    # Remove 'story_' prefix for folder name
    story_folder = (
        story_name.replace("story_", "")
        if story_name.startswith("story_")
        else story_name
    )

    return f"{lang_tag_lower}/{collection_lower}/{story_folder}/{story_name}.html"


def get_story_image_path(
    story_name: str, story_part: str, collection: str = "LM1000"
) -> str:
    """Get GCS path for story part image.

    Args:
        story_name: Name of the story
        story_part: Part name (e.g., "introduction", "setup")
        collection: Collection name

    Returns:
        str: GCS path (e.g., 'stories/LM1000/story_a_fishing_trip/images/introduction.png')
    """
    if not story_name.startswith("story_"):
        story_name = f"story_{story_name}"

    return f"stories/{collection}/{story_name}/images/{story_part}.png"


# ============================================================================
# UNIFIED UPLOAD/DOWNLOAD FUNCTIONS
# ============================================================================


def upload_file_to_gcs(
    obj: Any,
    bucket_name: str,
    file_path: str,
    content_type: Optional[str] = None,
    save_local: bool = True,
    local_base_dir: str = "../outputs/gcs",
) -> str:
    """Upload various file types directly to Google Cloud Storage.

    Unified naming: Uses 'file_path' parameter consistently.
    Supports local caching of uploaded files.

    Args:
        obj: Object to upload (bytes, dict, str, PIL Image, AudioSegment, etc.)
        bucket_name: Name of the GCS bucket
        file_path: Path to the file within the bucket (e.g., 'phrases/fr-FR/audio/flashcard/slow/hello_a3f8d2.mp3')
        content_type: MIME content type. If None, will be inferred.
        save_local: Whether to save a local copy of the file (default: True)
        local_base_dir: Base directory for local GCS mirror (default: "../outputs/gcs")

    Returns:
        str: GCS URI of the uploaded file (e.g., 'gs://bucket-name/path/file.mp3')

    Raises:
        ValueError: If unsupported object type is provided
    """
    # Get storage client (singleton from connections module)
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)

    # Ensure file_path doesn't have leading slashes
    file_path = file_path.lstrip("/")
    blob = bucket.blob(file_path)

    # Determine content type if not provided
    if content_type is None:
        content_type, _ = mimetypes.guess_type(file_path)

    # Handle different object types
    if isinstance(obj, bytes):
        # Direct bytes upload
        blob.upload_from_string(obj, content_type=content_type)
        if save_local:
            local_path = os.path.join(local_base_dir, bucket_name, file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(obj)

    elif isinstance(obj, (str, dict, list)):
        # Handle string, dict, or list content
        if isinstance(obj, list) and all(isinstance(item, str) for item in obj):
            # For List[str], ensure we save as JSON
            content_type = "application/json"
            content = json.dumps(obj)
        elif isinstance(obj, dict):
            # For dict, save as JSON
            content_type = "application/json"
            content = json.dumps(obj)
        else:
            # For string content
            if file_path.lower().endswith((".html", ".htm")):
                content_type = "text/html"
            elif file_path.lower().endswith(".css"):
                content_type = "text/css"
            elif file_path.lower().endswith(".js"):
                content_type = "application/javascript"
            elif file_path.lower().endswith(".txt"):
                content_type = "text/plain"
            content = obj

        # Upload the content
        blob.upload_from_string(content, content_type=content_type)
        if save_local:
            local_path = os.path.join(local_base_dir, bucket_name, file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)

    elif str(type(obj)).endswith("AudioSegment'>"):  # For pydub AudioSegment
        buffer = io.BytesIO()
        format_name = file_path.split(".")[-1].lower()

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
            local_path = os.path.join(local_base_dir, bucket_name, file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(buffer.getvalue())

    elif hasattr(obj, "save") and hasattr(obj, "mode"):  # For PIL Image
        # Get format from filename or default to PNG
        try:
            format_name = file_path.split(".")[-1].upper()
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
                local_path = os.path.join(local_base_dir, bucket_name, file_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(buffer.getvalue())
        except Exception as e:
            raise ValueError(f"Failed to save image: {e}")

    elif hasattr(obj, "read"):  # For file-like objects
        # For zip files, ensure we're using the correct content type
        if file_path.lower().endswith(".zip"):
            content_type = "application/zip"

        # Upload to GCS using upload_from_file
        blob.upload_from_file(obj, content_type=content_type)

        # Save locally if requested
        if save_local:
            # Reset file pointer to beginning for local save
            obj.seek(0)
            file_content = obj.read()

            local_path = os.path.join(local_base_dir, bucket_name, file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)

    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")

    # Return the GCS URI
    return f"gs://{bucket_name}/{file_path}"


def download_from_gcs(
    bucket_name: str,
    file_path: str,
    expected_type: Optional[str] = None,
    use_local: bool = True,
    local_base_dir: str = "../outputs/gcs",
) -> Any:
    """Download a file from GCS or read from local cache if available.

    Unified naming: Uses 'file_path' parameter consistently (was 'file_path' in read_from_gcs).
    Supports local caching for efficient access.

    Args:
        bucket_name: Name of the GCS bucket
        file_path: Path to the file within the bucket (e.g., 'phrases/fr-FR/audio/flashcard/slow/hello_a3f8d2.mp3')
        expected_type: Optional type hint ('audio', 'image', 'json', 'bytes', 'text', 'list')
                      If None, will be inferred from file extension
        use_local: Whether to check for a local copy first (default: True)
        local_base_dir: Base directory for local GCS mirror (default: "../outputs/gcs")

    Returns:
        The file content as an appropriate Python object

    Raises:
        FileNotFoundError: If file not found in GCS
        ValueError: If processing fails
    """
    # Infer expected_type from filename if not provided
    if expected_type is None:
        expected_type = _infer_expected_type_from_filename(file_path)

    # Ensure file_path doesn't have leading slashes
    file_path = file_path.lstrip("/")

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
                elif expected_type in ["json", "list"]:
                    with open(local_path, "r") as f:
                        return json.load(f)
                elif expected_type == "text":
                    with open(local_path, "r") as f:
                        return f.read()
                else:  # Default to bytes
                    with open(local_path, "rb") as f:
                        return f.read()
            except Exception as e:
                logger.exception(f"Error reading local file {local_path}")
                # Fall back to GCS if local read fails

    # Download from GCS
    storage_client = get_storage_client()
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
        elif expected_type in ["json", "list"]:
            return json.loads(content)
        elif expected_type == "text":
            return content.decode("utf-8")
        else:  # Default to bytes
            return content
    except Exception as e:
        raise ValueError(
            f"Error processing file {file_path} as {expected_type}: {str(e)}"
        )


def _infer_expected_type_from_filename(file_path: str) -> str:
    """Infer the expected type from a filename based on its extension.

    Args:
        file_path: The file path to analyze

    Returns:
        str: The inferred type ('json', 'image', 'audio', 'text', or 'bytes')
    """
    ext = file_path.lower().split(".")[-1]

    # Map extensions to expected types
    extension_to_type = {
        "json": "json",
        "png": "image",
        "jpg": "image",
        "jpeg": "image",
        "gif": "image",
        "webp": "image",
        "bmp": "image",
        "mp3": "audio",
        "wav": "audio",
        "m4a": "audio",
        "ogg": "audio",
        "html": "text",
        "htm": "text",
        "txt": "text",
        "css": "text",
        "js": "text",
    }

    return extension_to_type.get(ext, "bytes")


def check_blob_exists(bucket_name: str, file_path: str) -> bool:
    """Check if a blob exists in a GCS bucket.

    Args:
        bucket_name: Name of the GCS bucket
        file_path: Path to the blob within the bucket

    Returns:
        bool: True if the blob exists, False otherwise
    """
    file_path = file_path.lstrip("/")
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    return blob.exists()
