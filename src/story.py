import json
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

from google.cloud import storage
from pydub import AudioSegment
from tqdm import tqdm

)  # Keep for backward compatibility with existing functions
from src.convert import (
    convert_audio_to_base64,
    convert_PIL_image_to_base64,
    get_story_title,
    get_collection_title,
)
from src.gcs_storage import (
    check_blob_exists,

    read_from_gcs,
    upload_to_gcs,
    sanitize_path_component,
)
from src.llm_tools.story_generation import generate_story
from src.storage import (
    PRIVATE_BUCKET,
    PUBLIC_BUCKET,
    get_story_dialogue_path,
    upload_file_to_gcs,
)
from src.utils import load_template


def upload_styles_to_gcs():
    """Upload the styles.css file to the public GCS bucket."""

    # Load the CSS content with correct path handling
    try:
        # Try current directory first (when running from project root)
        styles_content = load_template("styles.css", "src/templates")
    except FileNotFoundError:
        # Fallback to relative path (when running from subdirectory)
        styles_content = load_template("styles.css", "../src/templates")

    # Upload to GCS
    public_url = upload_to_gcs(
        obj=styles_content,
        bucket_name=PUBLIC_BUCKET,
        file_name="styles.css",
        content_type="text/css",
    )

    print("✅ Styles uploaded successfully!")
    print(f"🌐 Public URL: {public_url}")

    return public_url


def create_and_upload_html_story(
    prepared_data: Dict,
    story_name: str,
    bucket_name: str = PUBLIC_BUCKET,
    component_path: str = "StoryViewer.js",
    template_path: str = "story_template.html",
    collection: str = "LM1000",
) -> str:
    """
    Create a standalone HTML file from prepared story data and upload it to GCS.

    Args:
        prepared_data: Dictionary containing prepared story data with base64 encoded assets
        story_name: Name of the story
        bucket_name: GCS bucket name for upload
        component_path: Path to the React component file
        template_path: Path to the HTML template file
        output_dir: Local directory to save HTML file before upload
        collection: Collection name for organizing stories

    Returns:
        str: Public URL of the uploaded HTML file
    """

    # Clean the story name for display
    story_title = get_story_title(story_name)

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
        collection_name=get_collection_title(collection),
        collection_raw=collection,
    )

    try:
        blob_path = get_public_story_path(story_name, collection)

        public_url = upload_to_gcs(
            obj=html_content,
            bucket_name=bucket_name,
            file_name=blob_path,
            content_type="text/html",
        )
        print(f"Uploaded HTML story to: {public_url}")

        return public_url

    except Exception as e:
        print(f"Error uploading to GCS: {str(e)}")
        return str(blob_path)  # Return local path if upload fails


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


# ============================================================================
# NEW STORY GENERATION FUNCTIONS (using llm_tools pattern)
# ============================================================================


def generate_and_upload_story(
    verbs: List[str],
    vocab: List[str],
    collection: str = "LM1000",
    bucket_name: str = PRIVATE_BUCKET,
) -> Tuple[str, Dict, str]:
    """Generate English story and upload to GCS using modern llm_tools pattern.

    Stories are ALWAYS generated in English. Translation happens separately
    using existing translation functions.

    This is the recommended way to generate new stories. It uses:
    - src.llm_tools.story_generation for LLM calls
    - src.storage for path generation and uploads
    - Dynamic story structure based on verb count

    Args:
        verbs: List of verbs to incorporate in the story
        vocab: List of other vocabulary words to use
        collection: Collection name (default: "LM1000")
        bucket_name: GCS bucket for upload (default: PRIVATE_BUCKET)

    Returns:
        Tuple of (story_name, story_dialogue, gcs_uri)
        - story_name: 3-word story title
        - story_dialogue: Dictionary with story parts and dialogue
        - gcs_uri: GCS URI where dialogue JSON was uploaded

    Example:
        >>> story_name, dialogue, uri = generate_and_upload_story(
        ...     verbs=["go", "see", "want"],
        ...     vocab=["coffee", "table", "friend"],
        ...     collection="LM1000"
        ... )
        >>> print(f"Created story: {story_name}")
        >>> print(f"Saved to: {uri}")
    """
    # Generate story using LLM tool (always in English)
    print(f"Generating story from {len(verbs)} verbs and {len(vocab)} vocab words...")
    story_name, story_dialogue = generate_story(verbs, vocab)

    print(f"Generated story: {story_name}")

    # Upload dialogue to GCS
    dialogue_path = get_story_dialogue_path(story_name, collection)
    gcs_uri = upload_file_to_gcs(
        obj=story_dialogue,
        bucket_name=bucket_name,
        file_path=dialogue_path,
    )

    print(f"Uploaded story dialogue to: {gcs_uri}")

    return story_name, story_dialogue, gcs_uri
