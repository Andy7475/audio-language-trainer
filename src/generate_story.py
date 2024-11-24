#!/usr/bin/env python3
import asyncio
import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.auth import default

# Add the parent directory of 'src' to the Python path
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# Load environment variables from .env file
load_dotenv()

from src.config_loader import config
from src.generate import (
    add_practice_phrases,
    create_story_plan_and_dialogue,
    add_translations,
    add_audio,
    create_album_files,
)
from src.utils import save_defaultdict, generate_story_image, create_html_story
from src.anki_tools import export_to_anki


def generate_story(story_name: str):
    """
    Generate a complete language learning story with audio, images, and interactive elements.

    Args:
        story_name: Name of the story to generate
    """
    # Initialize credentials
    credentials, project = default()

    # Setup paths and directories
    story_name_clean = story_name.replace(" ", "_")
    output_dir = Path(f"../outputs/{story_name_clean}")
    story_data_path = output_dir / f"story_data_{story_name_clean}.json"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config._load_config()
    print(f"Your target language is {config.TARGET_LANGUAGE_NAME}")

    try:
        # Create story plan and dialogue
        print("\nCreating story plan and dialogue...")
        story_data_dict = create_story_plan_and_dialogue(
            story_name, output_dir=str(output_dir)
        )
        save_defaultdict(story_data_dict, story_data_path)

        # Add practice phrases
        print("\nAdding practice phrases...")
        story_data_dict = add_practice_phrases(story_data_dict)
        save_defaultdict(story_data_dict, story_data_path)

        # Add translations
        print("\nAdding translations...")
        story_data_dict = add_translations(story_data_dict)
        save_defaultdict(story_data_dict, story_data_path)

        # Add audio (async operation)
        print("\nGenerating audio...")
        story_data_dict = add_audio(story_data_dict)

        # Save complete data as pickle
        pickle_path = output_dir / f"story_data_{story_name_clean}.pkl"
        print(f"\nSaving complete data to {pickle_path}...")
        with open(pickle_path, "wb") as file:
            pickle.dump(dict(story_data_dict), file)

        # Generate cover image
        print("\nGenerating cover artwork...")
        image_data = generate_story_image(story_name_clean)
        cover_path = output_dir / "cover_artwork.jpg"
        with open(cover_path, "wb") as f:
            f.write(image_data)

        # Create album files
        print("\nCreating album files...")
        create_album_files(
            story_data_dict, image_data, str(output_dir), story_name_clean
        )

        # Create HTML story viewer
        print("\nCreating HTML story viewer...")
        html_path = output_dir / f"{story_name_clean}.html"
        create_html_story(
            story_data_dict,
            str(html_path),
            component_path="../src/StoryViewer.js",
            title=story_name,
        )

        print("\nStory generation complete!")
        return story_data_dict

    except Exception as e:
        print(f"\nError during story generation: {str(e)}")
        raise


def main():
    """Main entry point for the script."""
    if len(sys.argv) != 2:
        print("Usage: python generate_story.py 'story name'")
        sys.exit(1)

    story_name = sys.argv[1]
    print(f"Generating story: {story_name}")

    try:
        generate_story(story_name)
    except Exception as e:
        print(f"Failed to generate story: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
