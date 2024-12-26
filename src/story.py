from pydub import AudioSegment
from tqdm import tqdm
from src.audio_generation import create_m4a_with_timed_lyrics
from src.config_loader import config
from typing import Optional, Dict
import json
import io
import base64


def create_html_story(
    story_data_dict: Dict,
    output_path: str,
    component_path: str,
    title: Optional[str] = None,
    language: str = config.TARGET_LANGUAGE_NAME,
) -> None:
    """
    Create a standalone HTML file from the story data dictionary.

    Args:
        story_data_dict: Dictionary containing story data, translations, and audio
        output_path: Path where the HTML file should be saved
        component_path: Path to the React component file
        title: Optional title for the story
        language: Target language name for Wiktionary links
    """

    # Process the story data and convert audio to base64
    prepared_data = prepare_story_data_for_html(story_data_dict)

    # Read the React component
    with open(component_path, "r", encoding="utf-8") as f:
        react_component = f.read()

    # Convert the React component from JSX to pure JavaScript
    # Note: In practice, you'd want to use a proper JSX transformer like Babel
    # This is a simplified version that assumes the component is already in JS

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lucide/0.263.1/lucide.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
        <style>
            .audio-player {{
                display: none;
            }}
        </style>
    </head>
    <body>
        <div id="root"></div>
        <script>
            // Embed the story data
            const storyData = {story_data};
            const targetLanguage = "{language}";
            
            {react_component}
            
            // Render the app
            const root = ReactDOM.createRoot(document.getElementById('root'));
            root.render(React.createElement(StoryViewer, {{ 
                storyData: storyData,
                targetLanguage: targetLanguage,
                title: "{title}"
            }}));
        </script>
    </body>
    </html>
    """

    # Format the HTML template
    html_content = html_template.format(
        title=title or "Language Learning Story",
        story_data=json.dumps(prepared_data),
        language=language,
        react_component=react_component,
    )

    # Write the HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML story created at: {output_path}")


def convert_audio_to_base64(audio_segment: AudioSegment) -> str:
    """Convert an AudioSegment to a base64 encoded string."""
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3")
    buffer.seek(0)
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode("utf-8")


def create_album_files(story_data_dict, image_data, output_dir, story_name_clean):
    """Creates and saves M4A files for the story, with album artwork.
    Each M4A contains normal dialogue, fast dialogue (repeated), and final dialogue."""

    PAUSE_TEXT = "---------"
    GAP_BETWEEN_PHRASES = AudioSegment.silent(duration=500)

    ALBUM_NAME = story_name_clean.replace("_", " ")
    TOTAL_TRACKS = len(story_data_dict) + 1  # +1 for the full dialogue track

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

        # Add fast dialogue 10 times
        for i in range(10):
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
            output_file=f"{output_dir}/{story_name_clean}_{story_part}.m4a",
            album_name=ALBUM_NAME,
            track_title=story_part,
            track_number=track_number,
            total_tracks=TOTAL_TRACKS,
            image_data=image_data,
        )
        print(f"Saved M4A file track number {track_number}")

    # Create final track with all dialogue in sequence
    print("Creating full dialogue track...")
    all_dialogue_audio = []
    all_dialogue_captions = []

    # First pass - normal speed
    for story_part in story_data_dict:
        dialogue_list = [
            utterance["text"]
            for utterance in story_data_dict[story_part]["translated_dialogue"]
        ]
        dialogue_audio_list = story_data_dict[story_part]["translated_dialogue_audio"]

        all_dialogue_audio.append(GAP_BETWEEN_PHRASES)
        all_dialogue_captions.append(f"Normal Speed - {story_part}")

        all_dialogue_audio.extend(dialogue_audio_list)
        all_dialogue_captions.extend(dialogue_list)

        all_dialogue_audio.append(GAP_BETWEEN_PHRASES)
        all_dialogue_captions.append(PAUSE_TEXT)

    # Fast version of all dialogue
    all_dialogue_audio.append(GAP_BETWEEN_PHRASES)
    all_dialogue_captions.append("Fast Dialogue Practice - All Parts")

    for i in range(10):  # Repeat fast version 10 times
        for story_part in story_data_dict:
            all_dialogue_audio.append(
                story_data_dict[story_part]["translated_dialogue_audio_fast"]
            )
            all_dialogue_captions.append(f"{story_part} - Fast Repetition {i+1}")
            all_dialogue_audio.append(GAP_BETWEEN_PHRASES)
            all_dialogue_captions.append(PAUSE_TEXT)

    # Final normal speed pass
    all_dialogue_audio.append(GAP_BETWEEN_PHRASES)
    all_dialogue_captions.append("Final Normal Speed - All Parts")

    for story_part in story_data_dict:
        dialogue_list = [
            utterance["text"]
            for utterance in story_data_dict[story_part]["translated_dialogue"]
        ]
        dialogue_audio_list = story_data_dict[story_part]["translated_dialogue_audio"]

        all_dialogue_audio.append(GAP_BETWEEN_PHRASES)
        all_dialogue_captions.append(f"Normal Speed - {story_part}")

        all_dialogue_audio.extend(dialogue_audio_list)
        all_dialogue_captions.extend(dialogue_list)

        all_dialogue_audio.append(GAP_BETWEEN_PHRASES)
        all_dialogue_captions.append(PAUSE_TEXT)

    # Create the full dialogue M4A
    create_m4a_with_timed_lyrics(
        audio_segments=all_dialogue_audio,
        phrases=all_dialogue_captions,
        output_file=f"{output_dir}/{story_name_clean}_full_dialogue.m4a",
        album_name=ALBUM_NAME,
        track_title="Full Dialogue - All Episodes",
        track_number=TOTAL_TRACKS,  # Last track
        total_tracks=TOTAL_TRACKS,
        image_data=image_data,
    )
    print(f"Saved full dialogue M4A as track number {TOTAL_TRACKS}")


def prepare_story_data_for_html(story_data_dict: Dict) -> Dict:
    """Process the story data dictionary to include base64 encoded audio for both
    normal dialogue utterances and fast dialogue versions."""
    prepared_data = {}

    for section_name, section_data in story_data_dict.items():
        prepared_data[section_name] = {
            "dialogue": section_data.get("dialogue", []),
            "translated_dialogue": section_data.get("translated_dialogue", []),
            "audio_data": {
                "dialogue": [],
                "fast_dialogue": None,  # Will hold the single fast version
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

        # Include image data if present
        if "image_data" in section_data:
            prepared_data[section_name]["image_data"] = section_data["image_data"]

        # Include M4A data if present
        if "m4a_data" in section_data:
            prepared_data[section_name]["m4a_data"] = section_data["m4a_data"]

    return prepared_data
