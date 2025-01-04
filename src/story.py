import base64
import io
import json
import os
from string import Template
from typing import Dict, Optional

from PIL import Image
from pydub import AudioSegment
from tqdm import tqdm

from src.anki_tools import generate_wiktionary_links
from src.audio_generation import create_m4a_with_timed_lyrics
from src.config_loader import config


def create_html_story(
    story_data_dict: Dict,
    image_dir: str,
    story_name: str,
    language: str = None,
    component_path: str = "../src/StoryViewer.js",
    template_path: str = "../src/story_template.html",
) -> None:
    """
    Create a standalone HTML file from the story data dictionary using string.Template.

    Args:
        story_data_dict: Dictionary containing story data, translations, and audio
        image_dir: Path where images are stored and output HTML will be saved
        story_name: Name of the story
        language: Target language name for Wiktionary links (defaults to config.TARGET_LANGUAGE_NAME)
        component_path: Path to the React component file
        template_path: Path to the HTML template file
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
    with open(component_path, "r", encoding="utf-8") as f:
        react_component = f.read()

    # Read the HTML template
    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())

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
):
    """Creates and saves M4A files for the story, with album artwork.
    Each M4A contains normal dialogue, fast dialogue (repeated), and final dialogue.

    story_name is expected to be of the form story_<story-title-with-underscores>"""
    REPEATS_OF_FAST_DIALOGUE = 10
    PAUSE_TEXT = "---------"
    GAP_BETWEEN_PHRASES = AudioSegment.silent(duration=500)

    ALBUM_NAME = clean_story_name(story_name)
    TOTAL_TRACKS = len(story_data_dict)

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

        # Fast dialogue section
        audio_list.append(GAP_BETWEEN_PHRASES)
        captions_list.append(f"{story_part} - Fast Dialogue Practice")

        # Add fast dialogue (there are 10 repeats in the audio)
        for i in range(REPEATS_OF_FAST_DIALOGUE):
            audio_list.append(data["translated_dialogue_audio_fast"])
            captions_list.append(f"Fast Dialogue - Repetition {i+1}")
            audio_list.append(GAP_BETWEEN_PHRASES)
            captions_list.append(PAUSE_TEXT)

        # Final dialogue at normal speed again
        audio_list.append(GAP_BETWEEN_PHRASES)
        captions_list.append(f"{story_part} - Final Dialogue")

        audio_list.extend(dialogue_audio_list)
        captions_list.extend(dialogue_list)

        # Create M4A file
        create_m4a_with_timed_lyrics(
            audio_segments=audio_list,
            phrases=captions_list,
            output_file=f"{output_dir}/{story_name}_{story_part}.m4a",
            album_name=ALBUM_NAME,
            track_title=story_part,
            track_number=track_number,
            total_tracks=TOTAL_TRACKS,
            cover_image=cover_image,
        )
        print(f"Saved M4A file track number {track_number}")


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


def prepare_dialogue_with_wiktionary(
    dialogue, language_name: str = config.TARGET_LANGUAGE_NAME
):
    """
    Process dialogue utterances to include Wiktionary links for the target language text.

    Args:
        dialogue: List of dialogue utterances with text and speaker
        language_name: Name of target language for Wiktionary links

    Returns:
        List of processed utterances with added wiktionary_links field
    """
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
    """Process the story data dictionary to include base64 encoded audio, images and M4A files.

    Args:
        story_data_dict: Dictionary containing story dialogue and audio data
        story_name: Name of the story (used to find corresponding M4A/image files)
        m4a_folder: Optional path to folder containing M4A files
        image_folder: Optional path to folder containing image files

    Returns:
        Dict: Processed dictionary with base64 encoded media content
    """
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

        # Add M4A data if folder is provided
        if m4a_folder:
            m4a_filename = f"{story_name}_{section_name}.m4a"
            m4a_path = os.path.join(m4a_folder, m4a_filename)

            try:
                if os.path.exists(m4a_path):
                    m4a_base64 = convert_m4a_file_to_base64(m4a_path)
                    prepared_data[section_name]["m4a_data"] = m4a_base64
            except Exception as e:
                print(
                    f"Warning: Failed to process M4A file for {section_name}: {str(e)}"
                )

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

    return prepared_data
