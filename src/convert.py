import re
import hashlib
import base64
import io
from pydub import AudioSegment
from typing import List, Optional


def clean_filename(phrase: str) -> str:
    """Convert a phrase to a clean filename-safe string."""
    # Convert to lowercase
    clean = phrase.lower()
    # Replace any non-alphanumeric characters (except spaces) with empty string
    clean = re.sub(r"[^a-z0-9\s]", "", clean)
    # Replace spaces with underscores
    clean = clean.replace(" ", "_")
    # Remove any double underscores
    clean = re.sub(r"_+", "_", clean)
    # Trim any leading/trailing underscores
    clean = clean.strip("_")
    return clean


def get_collection_title(collection: str) -> str:
    """
    Convert a collection name to a title case string.

    Args:
        collection: Input collection name (e.g. "LM1000")

    Returns:
        str: Collection title to accommodate changes in name (e.g. "First1000")

    Example:
        >>> get_collection_title("lm1000")
        'LM1000'
    """

    MAPPING = {"LM1000": "First1000", "LM2000": "Second1000", "WarmUp150": "WarmUp1000"}

    return MAPPING.get(collection, collection.title())


def get_story_title(story_name: str) -> str:
    """
    Clean a story name by removing 'story' and underscores, returning in title case.

    Args:
        story_name: Input story name (e.g. "story_community_park")

    Returns:
        str: Cleaned story name in title case (e.g. "Community Park")

    Example:
        >>> get_story_title("story_community_park")
        'Community Park'
    """
    # Remove 'story' and split on underscores
    name = story_name.replace("story_", "")
    words = name.split("_")

    # Convert to title case and join with spaces
    return " ".join(word.title() for word in words)


def convert_audio_to_base64(audio_segment: Optional[AudioSegment]) -> Optional[str]:
    """Convert an AudioSegment to a base64 encoded string, or return None if no audio segment is provided."""
    if audio_segment is None:
        return None
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3")
    buffer.seek(0)
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


def convert_base64_to_audio(base64_string: str, format: str = "mp3") -> AudioSegment:
    """Convert a base64 encoded string back to an AudioSegment.

    Args:
        base64_string: A base64 encoded string of audio data
        format: Audio format of the encoded data (default: "mp3")

    Returns:
        AudioSegment object containing the audio data

    Raises:
        ValueError: If the base64 string is invalid or the audio format is unsupported
    """
    try:
        # Remove potential data URI prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        # Decode base64 string to bytes
        audio_bytes = base64.b64decode(base64_string)

        # Create a BytesIO object from the bytes
        buffer = io.BytesIO(audio_bytes)

        # Load the audio data into an AudioSegment
        audio_segment = AudioSegment.from_file(buffer, format=format)

        return audio_segment

    except Exception as e:
        raise ValueError(f"Failed to convert base64 to AudioSegment: {str(e)}")


def convert_base64_list_to_audio_segments(
    base64_strings: List[str], format: str = "mp3"
) -> List[AudioSegment]:
    """Convert a list of base64 encoded strings to a list of AudioSegment objects.

    Args:
        base64_strings: List of base64 encoded strings of audio data
        format: Audio format of the encoded data (default: "mp3")

    Returns:
        List of AudioSegment objects containing the audio data

    Raises:
        ValueError: If any base64 string is invalid or the audio format is unsupported
    """
    audio_segments = []

    for i, base64_string in enumerate(base64_strings):
        try:
            audio_segment = convert_base64_to_audio(base64_string, format)
            audio_segments.append(audio_segment)
        except ValueError as e:
            raise ValueError(f"Failed to convert base64 string at index {i}: {str(e)}")

    return audio_segments


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


def string_to_large_int(s: str) -> int:
    """Notes in Anki have a unique ID, and so to create the note ID and ensure
    it correlates with the content we can pass in the translated phrase as a string
    and get back a large interger (a bit like a hash function).

    So this can be used to create a numerical ID from a given phrase.

    Args:
        s (str): The string to convert (usually the translated phrase)

    Returns:
        int: A large interger (equivalent to a hash)
    """
    # Encode the string to bytes
    encoded = s.encode("utf-8")
    # Create a SHA-256 hash
    hash_object = hashlib.sha256(encoded)
    # Get the hexadecimal representation
    hex_dig = hash_object.hexdigest()
    # Take the first 16 characters (64 bits) of the hex string
    truncated_hex = hex_dig[:16]
    # Convert hex to integer
    large_int = int(truncated_hex, 16)
    # Ensure the value is positive and within SQLite's signed 64-bit integer range
    return large_int & 0x7FFFFFFFFFFFFFFF


def get_deck_name(
    story_name: str, collection: str, story_position: Optional[int], language: str
) -> str:
    """
    Format a deck name in the pattern: Language::Collection::Position Story Title

    Args:
        story_name: Name of the story (e.g. "story_community_park")
        collection: Collection name (e.g. "LM1000")
        story_position: Optional position number (e.g. 1 becomes "01")
        language: Language name (e.g. "French")

    Returns:
        str: Formatted deck name (e.g. "French::LM1000::01 Community Park")
    """
    # Get the story title without "story_" prefix and with proper capitalization
    story_title = get_story_title(story_name)
    collection_title = get_collection_title(collection)
    # Format the position if provided
    position_str = ""
    if story_position is not None:
        position_str = f"{story_position:02d} "

    # Capitalize the language name
    language_cap = language.title()

    return (
        f"FirePhrase - {language_cap}::{collection_title}::{position_str} {story_title}"
    )


def convert_bytes_to_base64(data: bytes) -> str:
    """Convert bytes to a base64 encoded string."""
    return base64.b64encode(data).decode("utf-8")


def convert_PIL_image_to_base64(pil_image, format="PNG") -> str:
    """
    Convert a PIL Image to a base64 encoded string.

    Args:
        pil_image: PIL Image object
        format: Image format to use (default: "PNG")

    Returns:
        str: Base64 encoded string representation of the image
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    buffer.seek(0)
    image_bytes = buffer.read()
    return convert_bytes_to_base64(image_bytes)


def get_language_code(language_name: str) -> str:
    """
    Get the ISO 639-1 language code for a given language name using langcodes.
    Returns an empty string if not found.

    Example:
        get_language_code("Swedish") -> "sv"
        get_language_code("French") -> "fr"
    """
    try:
        from langcodes import Language

        lang = Language.find(language_name)
        return lang.language
    except Exception:
        return None
