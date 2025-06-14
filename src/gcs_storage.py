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
        obj: The object to upload (bytes, dict, str, PIL Image, AudioSegment, List[str], etc.)
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
            if file_name.lower().endswith((".html", ".htm")):
                content_type = "text/html"
            elif file_name.lower().endswith(".css"):
                content_type = "text/css"
            elif file_name.lower().endswith(".js"):
                content_type = "application/javascript"
            elif file_name.lower().endswith(".txt"):
                content_type = "text/plain"
            content = obj

        # Upload the content
        blob.upload_from_string(content, content_type=content_type)
        if save_local:
            local_path = os.path.join(local_base_dir, bucket_name, full_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)

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
        # For zip files, ensure we're using the correct content type
        if file_name.lower().endswith(".zip"):
            content_type = "application/zip"

        # Upload to GCS using upload_from_file
        blob.upload_from_file(obj, content_type=content_type)

        # Save locally if requested
        if save_local:
            # Reset file pointer to beginning for local save
            obj.seek(0)
            file_content = obj.read()

            local_path = os.path.join(local_base_dir, bucket_name, full_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)

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


def infer_expected_type_from_filename(filename: str) -> str:
    """
    Infer the expected type from a filename based on its extension.

    Args:
        filename: The filename to analyze

    Returns:
        str: The inferred type ('json', 'image', 'audio', 'text', or 'bytes')
    """
    ext = filename.lower().split(".")[-1]

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
        expected_type: Optional type hint ('audio', 'image', 'json', 'bytes', 'text', 'list')
                      If None, will be inferred from file extension
        use_local: Whether to check for a local copy first
        local_base_dir: Base directory for local GCS mirror

    Returns:
        The file content as an appropriate Python object
    """
    # Infer expected_type from filename if not provided
    if expected_type is None:
        expected_type = infer_expected_type_from_filename(file_path)

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
    """Get the GCS path for a story collection file.

    The file contains a dictionary mapping story names to lists of phrase information.
    Stories are ordered by priority/sequence. For each story, phrases are ordered by
    their vocabulary coverage score.

    Format:
    {
        "story_name": [
            {
                "phrase": str,           # The English phrase
                "score": float,          # Vocabulary coverage score
                "new_story_verbs": int,  # New verbs for this story
                "new_story_vocab": int,  # New vocab for this story
                "new_global_verbs": int, # New verbs across all stories
                "new_global_vocab": int, # New vocab across all stories
                "total_new_words": int   # Total new words introduced
            },
            ...
        ],
        ...
    }

    Args:
        collection: Collection name (default: "LM1000")

    Returns:
        str: Path to the collection file in GCS
        Format: collections/{collection}/{collection}.json
    """
    return f"collections/{collection}/{collection}.json"


def get_story_challenges_path(story_name: str, collection: str = "LM1000") -> str:
    """Get the GCS path for a story's challenges JSON file (dictionary of scenarios).
    Challenges are language-agnostic (English) so they go in the common folder."""

    return f"collections/{collection}/common/stories/{story_name}/challenges.json"


def get_story_translated_challenges_path(
    story_name: str, collection: str = "LM1000"
) -> str:
    """Get the GCS path for a story's challenges webpage
    e.g: audio-language-trainer-stories/swedish/lm1000/story_birthday_party_planning_mishap/challenges.html
    """
    language = config.TARGET_LANGUAGE_NAME.lower()
    collection_folder = sanitize_path_component(collection.lower())
    return f"{language}/{collection_folder}/{story_name}/challenges.html"


def get_m4a_file_path(
    story_name: str,
    story_part: str,
    story_position: int,
    fast: bool = False,
    collection: str = "LM1000",
) -> str:
    """Get the GCS path for a story part's m4a file. All M4A files for a collection/language are stored in the same folder.

    Args:
        story_name: Name of the story
        story_part: Part of the story (e.g., 'introduction')
        fast: Whether this is a fast version of the audio
        collection: Collection name (default: "LM1000")

    Returns:
        str: Path to the m4a file in GCS
        Format: collections/{collection}/{language}/audio/{filename}
        where filename includes story name and part information
    """
    language = config.TARGET_LANGUAGE_NAME.lower()
    story_part = sanitize_path_component(story_part)

    if fast:
        filename = get_m4a_filename(
            story_name, story_part, fast=True, story_position=story_position
        )
        return f"collections/{collection}/{language}/audio/{filename}"
    else:
        filename = get_m4a_filename(
            story_name, story_part, fast=False, story_position=story_position
        )
        return f"collections/{collection}/{language}/audio/{filename}"


def get_m4a_blob_prefix(story_name: str) -> str:
    """Get the GCS bucket prefix for a story part's m4a file.
    Each story part has its own m4a file.

    the pattern is bucket_name/blob_prefix/filename.m4a.
    Such that get_m4a_file_path = get_m4a_blob_prefix + get_m4a_filename."""

    language = config.TARGET_LANGUAGE_NAME.lower()

    return f"{language}/{story_name}/"


def get_m4a_filename(
    story_name: str,
    story_part: str,
    fast: bool = False,
    story_position: Optional[int] = None,
) -> str:
    """Get the filename for a story part's m4a file.

    Args:
        story_name: Name of the story
        story_part: Part of the story (e.g., 'introduction')
        fast: Whether this is a fast version of the audio
        story_position: Optional position number of the story in the collection

    Returns:
        str: Filename in format: {language}_{position:02d}_{story_name}_{story_part}[_FAST].m4a
    """
    language = config.TARGET_LANGUAGE_NAME.lower()
    story_part = sanitize_path_component(story_part)

    # Format position if provided, otherwise omit it
    position_str = f"{story_position:02d}_" if story_position is not None else ""

    if fast:
        return f"{language}_{position_str}{story_name}_{story_part}_FAST.m4a"
    else:
        return f"{language}_{position_str}{story_name}_{story_part}.m4a"


def get_story_dialogue_path(story_name: str, collection: str = "LM1000") -> str:
    """Get the GCS path for a story's dialogue file (initial English only)."""
    return f"collections/{collection}/common/stories/{story_name}/dialogue.json"


def get_public_story_path(
    story_name: Optional[str] = None, collection: str = "LM1000"
) -> str:
    """Get the GCS blob path for a story's public HTML file.
    Meant to go to the GCS public bucket which is for holding stories."""

    language = config.TARGET_LANGUAGE_NAME
    language_folder = sanitize_path_component(language.lower())
    collection_folder = sanitize_path_component(collection.lower())
    story_folder = sanitize_path_component(story_name)
    index_path = f"{language_folder}/{collection_folder}/index.html"
    story_path = (
        f"{language_folder}/{collection_folder}/{story_folder}/{story_name}.html"
    )
    if story_name is None:
        return index_path
    else:
        return story_path


def get_story_translated_dialogue_path(
    story_name: str, collection: str = "LM1000"
) -> str:
    language = config.TARGET_LANGUAGE_NAME.lower()
    """Get the GCS path for a story's translated dialogue file."""
    return f"collections/{collection}/{language}/stories/{story_name}/translated_dialogue.json"


def get_translated_phrases_path(collection: str = "LM1000") -> str:
    """Get the GCS path for a collections's translated phrases file. These are dictionaries keyed of the phrase key
    and contain translations and wiktionary links."""
    language = config.TARGET_LANGUAGE_NAME.lower()
    return f"collections/{collection}/{language}/translations.json"


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
    return f"collections/{collection}/{language}/stories/{story_name}/audio/{story_part}/{filename}"


def get_fast_audio_path(
    story_name: str, story_part: str, collection: str = "LM1000"
) -> str:
    language = config.TARGET_LANGUAGE_NAME.lower()
    """Get the GCS path for a fast audio file."""
    return f"collections/{collection}/{language}/stories/{story_name}/audio/{story_part}/fast.mp3"


def get_image_path(story_name: str, story_part: str, collection: str = "LM1000") -> str:
    """Get the GCS path for a story part image."""
    return (
        f"collections/{collection}/common/stories/{story_name}/images/{story_part}.png"
    )


def get_story_names(
    collection: str = "LM1000", bucket_name: Optional[str] = None
) -> List[str]:
    """
    List all story names in the given collection by listing blobs in the 'collections/{collection}/common/stories/' folder.
    Returns a list of story names in the format 'story_this_is_the_name'.

    Args:
        collection: Collection name (default: "LM1000")
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)

    Returns:
        List[str]: List of story names (e.g. ['story_sunset_adventure_documentary', ...])
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET
    from google.cloud import storage

    # Use get_story_dialogue_path to get the stories folder prefix
    dummy_story = "story_dummy"
    dialogue_path = get_story_dialogue_path(dummy_story, collection)
    # Remove '/story_dummy/dialogue.json' to get the prefix
    prefix = "/".join(dialogue_path.split("/")[:-2]) + "/"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    story_names = set()
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) >= 5:
            story_name = parts[4]
            if story_name.startswith("story_"):
                story_names.add(story_name)
    return sorted(story_names)


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
    return f"phrases/{language}/audio/{speed}/{phrase_key}.mp3"


def get_phrase_image_path(phrase_key: str, use_language: bool = False) -> str:
    """
    Get the GCS path for a phrase's image file.

    Args:
        phrase_key: Key identifying the phrase

    Returns:
        str: Path to the image file in GCS
    """
    if use_language:
        language = config.TARGET_LANGUAGE_NAME.lower()
        return f"phrases/{language}/images/{phrase_key}.png"
    else:
        return f"phrases/common/images/{phrase_key}.png"


def get_phrase_path(collection: str = "LM1000") -> str:
    """Get the GCS path for storing raw English phrases for a collection.

    Args:
        collection: Collection name (default: "LM1000")

    Returns:
        str: Path to the phrases file in GCS
        Format: collections/{collection}/phrases.json
    """
    return f"collections/{collection}/phrases.json"


def get_phrase_index_path(collection: str = "LM1000") -> str:
    """Get the GCS path for storing the lemma word index file for a collection.

    Args:
        collection: Collection name (default: "LM1000")

    Returns:
        str: Path to the index file in GCS
        Format: collections/{collection}/index.json
    """
    return f"collections/{collection}/index.json"


def get_phrase_to_story_index_path(collection: str = "LM1000") -> str:
    """Get the GCS path for storing the phrase-to-story index file.
    This file maps each phrase key to a list of story names where that phrase appears.

    Args:
        collection: Collection name (default: "LM1000")

    Returns:
        str: Path to the index file in GCS
        Format: collections/{collection}/phrase_to_story_index.json
    """
    return f"collections/{collection}/phrase_to_story_index.json"


def get_story_index_path(collection: str = "LM1000") -> str:
    """Get the GCS path for storing the story index file.
    This file contains indexes mapping words to the stories containing them,
    including verb and vocabulary indexes and word counts.

    Args:
        collection: Collection name (default: "LM1000")

    Returns:
        str: Path to the index file in GCS
        Format: collections/{collection}/story_index.json
    """
    return f"collections/{collection}/story_index.json"


def get_flashcard_path(
    story_name: Optional[str] = None,
    collection: str = "LM1000",
    language: Optional[str] = None,
    story_position: Optional[int] = None,
) -> str:
    """
    Get the GCS path for a story's flashcard file.

    Args:
        story_name: Name of the story. If None, returns just the flashcard folder path.
        collection: Collection name (default: "LM1000")
        language: Optional language name (defaults to config.TARGET_LANGUAGE_NAME)
        story_position: Optional position number for the story

    Returns:
        str: Path to the flashcard file in GCS
        Format: collections/{collection}/{language}/flashcards/{position}_{story_name}.apkg
        If story_name is None, returns: collections/{collection}/{language}/flashcards/
    """
    if language is None:
        language = config.TARGET_LANGUAGE_NAME.lower()

    # If story_name is None, return just the folder path
    if story_name is None:
        return f"collections/{collection}/{language}/flashcards/"

    # Sanitize the story name for use in paths
    sanitized_story = sanitize_path_component(story_name)

    # Format the position if provided
    from src.story import get_story_position

    story_position = get_story_position(story_name, collection=collection)
    if story_position is not None:
        position_str = f"{story_position:02d}"

    return f"collections/{collection}/{language}/flashcards/{language}_{position_str}_{sanitized_story}.apkg"


def get_marketing_image_path(
    product_type: str,
    collection: str = "LM1000",
    language: Optional[str] = None,
    bundle_range: Optional[str] = None,
    story_name: Optional[str] = None,
) -> str:
    """
    Get the GCS path for a marketing image.

    Args:
        product_type: One of "complete", "bundle", "individual", "templates", or "anatomy"
        collection: Collection name (e.g., "LM1000")
        language: Target language (defaults to config.TARGET_LANGUAGE_NAME)
        bundle_range: For bundles, range like "01-08"
        story_name: For individual products, the story name to include in filename

    Returns:
        str: GCS path (after bucket name) for the marketing image
        Format: collections/{collection}/{language}/marketing/{filename}
    """
    if language is None:
        language = config.TARGET_LANGUAGE_NAME.lower()

    base_path = f"collections/{collection}/{language}/marketing/"

    if product_type == "complete":
        filename = f"{language}_{collection}_complete_pack.png"
    elif product_type == "bundle":
        filename = f"{language}_{collection}_bundle_{bundle_range}.png"
    elif product_type == "individual":
        if story_name:
            filename = f"{language}_{collection}_individual_{story_name}.png"
        else:
            filename = f"{language}_{collection}_individual_pack.png"
    elif product_type == "templates":
        filename = f"{language}_{collection}_template_types.png"
    elif product_type == "anatomy":
        filename = f"{language}_{collection}_flashcard_anatomy.png"
    else:
        raise ValueError(f"Invalid product_type: {product_type}")

    return base_path + filename


def get_shopify_image_path(filename: str, bucket_name: Optional[str] = None) -> str:
    """
    Get the GCS path for a Shopify image.

    Args:
        filename: Name of the image file

    Returns:
        str: GCS path for the Shopify image
        Format: audio-language-trainer-private-content/resources/shopify/images/{filename}
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET
    return f"{bucket_name}/resources/shopify/images/{filename}"


def get_stories_from_collection(
    bucket_name: str = config.GCS_PRIVATE_BUCKET, collection: str = "LM1000"
) -> List[str]:
    """
    Get list of story names from a collection file, preserving their order.

    Args:
        bucket_name: GCS bucket name
        collection: Collection name

    Returns:
        List[str]: List of story names in order
    """
    collection_path = get_story_collection_path(collection)
    try:
        collection_data = read_from_gcs(bucket_name, collection_path, "json")
        # Assuming the collection file has story names as keys
        return list(collection_data.keys())
    except Exception as e:
        print(f"Error loading collection {collection}: {str(e)}")
        return []
