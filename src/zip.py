import os
import zipfile
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from src.config_loader import config
from src.gcs_storage import (
    get_m4a_file_path,
    get_story_collection_path,
    read_from_gcs,
    upload_to_gcs,
)
from src.utils import get_story_position


def create_m4a_zip_collections(
    bundle_config: Dict[str, List[int]],
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
    output_base_dir: str = "../outputs/gcs",
    use_local_files: bool = True,
) -> Dict[str, str]:
    """
    Create zip files for M4A audio collections based on bundle configuration.

    Creates:
    - Complete collection zip (all M4A files)
    - Bundle zips (based on bundle_config)
    - Individual story zips (one per story)

    Args:
        bundle_config: Dict mapping bundle names to story position ranges
                      e.g., {"Bundle 01-08": [1, 8], "Bundle 09-14": [9, 14]}
        collection: Collection name (default: "LM1000")
        bucket_name: GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)
        output_base_dir: Base directory for outputs (default: "../outputs/gcs")
        use_local_files: Whether to check for local files first before downloading

    Returns:
        Dict mapping zip types to their file paths
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Get collection data to find all stories
    collection_path = get_story_collection_path(collection)
    collection_data = read_from_gcs(bucket_name, collection_path, "json")
    all_stories = list(collection_data.keys())

    language = config.TARGET_LANGUAGE_NAME.lower()
    target_language_name = config.TARGET_LANGUAGE_NAME

    # Get the base path for M4A files to use for zip files
    # Use the first story to get the path structure
    if all_stories:
        sample_m4a_path = get_m4a_file_path(
            all_stories[0], "introduction", 1, collection=collection
        )
        base_path = Path(sample_m4a_path).parent
    else:
        base_path = Path(f"collections/{collection}/audio/{language}")

    # Create output directory structure using the same path as M4A files
    zip_output_dir = Path(output_base_dir) / bucket_name / base_path
    zip_output_dir.mkdir(parents=True, exist_ok=True)

    # Track created zip files
    created_zips = {}

    print(f"Creating M4A zip collections for {collection} in {target_language_name}")
    print(f"Found {len(all_stories)} stories total")

    # 1. Create Complete Collection Zip
    print("\n=== Creating Complete Collection Zip ===")
    complete_zip_name = f"{language}_{collection.lower()}_complete_audio_collection.zip"
    complete_zip_path = zip_output_dir / complete_zip_name

    complete_m4a_files = []
    for story in all_stories:
        story_files = _get_story_m4a_files(
            story, collection, bucket_name, use_local_files, output_base_dir
        )
        complete_m4a_files.extend(story_files)

    _create_zip_file(
        complete_m4a_files,
        complete_zip_path,
        f"Complete {target_language_name} {collection} Audio Collection",
        bucket_name,
    )
    created_zips["complete"] = str(complete_zip_path)

    # 2. Create Bundle Zips
    print("\n=== Creating Bundle Zips ===")
    for bundle_name, (start_pos, end_pos) in bundle_config.items():
        bundle_stories = [
            story
            for story in all_stories
            if start_pos <= get_story_position(story, collection) <= end_pos
        ]

        if not bundle_stories:
            print(f"No stories found for {bundle_name}")
            continue

        bundle_zip_name = f"{language}_{collection.lower()}_bundle_{start_pos:02d}_{end_pos:02d}_audio.zip"
        bundle_zip_path = zip_output_dir / bundle_zip_name

        bundle_m4a_files = []
        for story in bundle_stories:
            story_files = _get_story_m4a_files(
                story, collection, bucket_name, use_local_files, output_base_dir
            )
            bundle_m4a_files.extend(story_files)

        _create_zip_file(
            bundle_m4a_files,
            bundle_zip_path,
            f"{target_language_name} {collection} {bundle_name} Audio",
            bucket_name,
        )
        created_zips[f"bundle_{start_pos:02d}_{end_pos:02d}"] = str(bundle_zip_path)

    # 3. Create Individual Story Zips
    print("\n=== Creating Individual Story Zips ===")
    for story in tqdm(all_stories, desc="Creating individual zips"):
        try:
            story_position = get_story_position(story, collection)
            story_files = _get_story_m4a_files(
                story, collection, bucket_name, use_local_files, output_base_dir
            )

            if not story_files:
                print(f"No M4A files found for {story}")
                continue

            individual_zip_name = f"{language}_{collection.lower()}_story_{story_position:02d}_{story.replace('story_', '').replace('_', '-')}_audio.zip"
            individual_zip_path = zip_output_dir / individual_zip_name

            from src.convert import get_story_title

            story_title = get_story_title(story)

            _create_zip_file(
                story_files,
                individual_zip_path,
                f"{target_language_name} {collection} Story {story_position:02d}: {story_title} Audio",
                bucket_name,
            )
            created_zips[f"individual_{story}"] = str(individual_zip_path)

        except ValueError as e:
            print(f"Skipping story {story}: {e}")
            continue

    print(f"\n🎉 Created {len(created_zips)} zip files successfully!")
    print(f"Zip files saved to: {zip_output_dir}")

    return created_zips


def _get_story_m4a_files(
    story_name: str,
    collection: str,
    bucket_name: str,
    use_local_files: bool,
    output_base_dir: str,
) -> List[Tuple[str, str]]:
    """
    Get M4A file paths for a story (both normal and fast versions).

    Returns:
        List of tuples (local_file_path, archive_name)
    """
    story_position = get_story_position(story_name, collection)

    # Get the story parts from the dialogue data in GCS
    from src.gcs_storage import get_story_dialogue_path, read_from_gcs

    dialogue_path = get_story_dialogue_path(story_name, collection)
    try:
        dialogue_data = read_from_gcs(bucket_name, dialogue_path, expected_type="json")
        # Story parts are the top-level keys of the dialogue JSON
        if isinstance(dialogue_data, dict):
            story_parts = list(dialogue_data.keys())
        else:
            # Fallback to standard parts if structure is unexpected
            story_parts = ["introduction", "development", "conclusion"]
    except Exception as e:
        print(f"Warning: Could not load story parts for {story_name}: {e}")
        story_parts = ["introduction", "development", "conclusion"]

    m4a_files = []

    for story_part in story_parts:
        # Normal speed file
        normal_gcs_path = get_m4a_file_path(
            story_name, story_part, story_position, fast=False, collection=collection
        )
        normal_local_path = _ensure_local_m4a_file(
            normal_gcs_path, bucket_name, use_local_files, output_base_dir
        )

        if normal_local_path and os.path.exists(normal_local_path):
            # Use just the filename for the archive
            archive_name = os.path.basename(normal_gcs_path)
            m4a_files.append((normal_local_path, archive_name))

        # Fast speed file
        fast_gcs_path = get_m4a_file_path(
            story_name, story_part, story_position, fast=True, collection=collection
        )
        fast_local_path = _ensure_local_m4a_file(
            fast_gcs_path, bucket_name, use_local_files, output_base_dir
        )

        if fast_local_path and os.path.exists(fast_local_path):
            # Use just the filename for the archive
            archive_name = os.path.basename(fast_gcs_path)
            m4a_files.append((fast_local_path, archive_name))

    return m4a_files


def _ensure_local_m4a_file(
    gcs_file_path: str, bucket_name: str, use_local_files: bool, output_base_dir: str
) -> Optional[str]:
    """
    Ensure M4A file exists locally, downloading from GCS if needed.

    Returns:
        Local file path if successful, None if file not found
    """
    # Construct local path
    local_path = Path(output_base_dir) / bucket_name / gcs_file_path

    # Check if local file exists
    if use_local_files and local_path.exists():
        return str(local_path)

    # Try to download from GCS
    try:
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download file as audio
        audio_segment = read_from_gcs(bucket_name, gcs_file_path, "audio")

        # Export to M4A format
        audio_segment.export(
            str(local_path), format="ipod"
        )  # ipod is the format name for m4a in pydub

        return str(local_path)

    except Exception as e:
        print(f"Warning: Could not get M4A file {gcs_file_path}: {str(e)}")
        return None


def _create_zip_file(
    m4a_files: List[Tuple[str, str]],
    zip_path: Path,
    description: str,
    bucket_name: str,
) -> None:
    """
    Create a zip file from M4A files and upload to GCS.

    Args:
        m4a_files: List of tuples (local_file_path, archive_name)
        zip_path: Path where zip file will be created
        description: Description for progress bar
        bucket_name: GCS bucket name to upload to
    """
    if not m4a_files:
        print(f"No files to zip for {description}")
        return

    # print(f"\nCreating {description} ({len(m4a_files)} files)")
    # print(f"Files to be included:")
    # for local_path, archive_name in m4a_files:
    #    print(f"  - {archive_name} (from {local_path})")

    # Create zip file in memory
    print("\nCreating zip file in memory...")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for local_path, archive_name in tqdm(
            m4a_files, desc="Adding files", leave=False
        ):
            if os.path.exists(local_path):
                # print(f"Adding {archive_name} ({os.path.getsize(local_path) / 1024:.1f} KB)")
                zipf.write(local_path, archive_name)
            else:
                print(f"Warning: File not found: {local_path}")

    # Get zip file size
    zip_size_mb = len(zip_buffer.getvalue()) / (1024 * 1024)
    print(f"\nZip file created in memory: {zip_size_mb:.1f} MB")

    # First write to local file
    print(f"\nWriting zip file to local path: {zip_path}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with open(zip_path, "wb") as f:
        f.write(zip_buffer.getvalue())
    print(
        f"Local file written successfully: {zip_path.stat().st_size / (1024 * 1024):.1f} MB"
    )
