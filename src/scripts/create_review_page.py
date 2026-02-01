"""Generate HTML review pages for phrase verification.

This module creates comprehensive review pages that display all flashcard content
for verification: images, translations, wiktionary links, and audio files.
"""

import base64
import io
import os
from datetime import datetime
from pathlib import Path
from typing import List, Union

from langcodes import Language
from PIL import Image

# Import from your project structure
from src.models import get_language
from src.storage import PRIVATE_BUCKET, get_phrase_audio_path
from src.images.manipulation import resize_image
from src.utils import render_html_content
from src.convert import convert_PIL_image_to_base64


def create_review_page(
    phrases: List,  # List[Phrase] - using Any to avoid import issues
    source_language: Union[str, Language],
    target_language: Union[str, Language],
    filepath: str = "review.html",
) -> str:
    """Generate an HTML review page for phrase verification.

    Creates a comprehensive review page with:
    - Navigation links to jump between phrases
    - Embedded images (resized and base64 encoded)
    - Source and target language text
    - Wiktionary links
    - Audio players for normal and slow speeds

    The template (phrase_review_template.html.jinja2) should be placed in one of:
    - templates/
    - src/templates/
    - ../src/templates/

    Args:
        phrases: List of Phrase objects to review
        source_language: Source language (e.g., "en-GB" or Language object)
        target_language: Target language (e.g., "fr-FR" or Language object)
        filepath: Output HTML file path (default: "review.html")

    Returns:
        str: Path to the generated HTML file

    Example:
        >>> phrases = [phrase1, phrase2, phrase3]
        >>> create_review_page(phrases, "en-GB", "fr-FR", "french_review.html")
        'french_review.html'
    """
    # Normalize languages
    source_lang = get_language(source_language)
    target_lang = get_language(target_language)

    # Process each phrase
    processed_phrases = []
    for phrase in phrases:
        phrase_data = {
            "phrase_hash": phrase.key,
            "source_text": phrase.english,  # Always English in your model
            "target_text": "",
            "image_base64": None,
            "wiktionary_html": "",
            "audio_normal_path": None,
            "audio_slow_path": None,
        }

        # Get target translation
        target_lang_tag = target_lang.to_tag()
        if target_lang_tag in phrase.translations:
            translation = phrase.translations[target_lang_tag]
            phrase_data["target_text"] = translation.text

            # Get wiktionary links if available
            try:
                phrase_data["wiktionary_html"] = translation.get_wiktionary_links(
                    separator=" "
                )
            except Exception as e:
                print(f"Warning: Could not get wiktionary links for {phrase.key}: {e}")

            # Get image (try to load from translation object)
            try:
                if translation.image is not None:
                    # Image already loaded in memory
                    resized_img = resize_image(translation.image, height=400, width=400)
                    phrase_data["image_base64"] = convert_PIL_image_to_base64(
                        resized_img
                    )
                elif translation.image_file_path:
                    # Try to load from local GCS cache
                    local_img_path = os.path.join(
                        "outputs/gcs", PRIVATE_BUCKET, translation.image_file_path
                    )
                    if os.path.exists(local_img_path):
                        img = Image.open(local_img_path)
                        resized_img = resize_image(img, height=400, width=400)
                        phrase_data["image_base64"] = convert_PIL_image_to_base64(
                            resized_img
                        )
            except Exception as e:
                print(f"Warning: Could not load image for {phrase.key}: {e}")

            # Check for audio files (flashcard context)
            # Normal speed
            normal_file_path = get_phrase_audio_path(
                phrase_hash=phrase.key,
                language=target_lang,
                context="flashcard",
                speed="normal",
            )
            normal_path = os.path.join("outputs/gcs", PRIVATE_BUCKET, normal_file_path)
            if os.path.exists(normal_path):
                phrase_data["audio_normal_path"] = normal_path

            # Slow speed
            slow_file_path = get_phrase_audio_path(
                phrase_hash=phrase.key,
                language=target_lang,
                context="flashcard",
                speed="slow",
            )
            slow_path = os.path.join("outputs/gcs", PRIVATE_BUCKET, slow_file_path)
            if os.path.exists(slow_path):
                phrase_data["audio_slow_path"] = slow_path

        processed_phrases.append(phrase_data)

    # Prepare template data
    template_data = {
        "phrases": processed_phrases,
        "source_language": source_lang.display_name(),
        "target_language": target_lang.display_name(),
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Render template using your existing render_html_content function
    html_content = render_html_content(
        data=template_data, template_name="phrase_review_template.html.jinja2"
    )

    # Write to file
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✓ Review page generated: {filepath}")
    print(f"  - {len(phrases)} phrases")
    print(f"  - {source_lang.display_name()} → {target_lang.display_name()}")

    return str(output_path)
