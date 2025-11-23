"""High-level image generation orchestration."""

import copy
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from PIL import Image
from tqdm import tqdm

from src.images.providers import ImagenProvider, StabilityProvider, DeepAIProvider
from src.images.styles import add_image_style, load_image_styles
from src.images.manipulation import resize_image
from src.llm_tools.image_generation import generate_phrase_image_prompt
from src.llm_tools.story_image_generation import generate_story_image_prompt
from src.storage import (
    PRIVATE_BUCKET,
    check_blob_exists,
    get_phrase_image_path,
    upload_file_to_gcs,
)
from src.convert import clean_filename


def generate_image(
    prompt: str,
    style: str = None,
    project_id: Optional[str] = None,
    vertex_region: Optional[str] = None,
    model_order: List[Literal["imagen", "stability", "deepai"]] = None,
) -> Optional[Image.Image]:
    """Generate an image using multiple providers in specified order.

    Args:
        prompt: The image generation prompt
        style: Art style to apply (defaults to "ghibli")
        project_id: GCP project ID for Imagen provider
        vertex_region: GCP region for Imagen provider
        model_order: List of models to try in order (default: ["imagen", "stability", "deepai"])

    Returns:
        Optional[Image.Image]: Generated image or None if all attempts fail
    """
    if model_order is None:
        model_order = ["imagen", "stability", "deepai"]

    if style is None:
        style = "ghibli"

    # Apply style to prompt
    try:
        styles_dict = load_image_styles()
        prompt = add_image_style(prompt, style, styles_dict)
    except Exception as e:
        print(f"Warning: Could not apply style '{style}': {e}")
        # Continue with unstyled prompt

    print(f"🎨 Starting image generation process")
    print(f"   Prompt: {prompt}")
    print(f"   Will try providers in order: {model_order}")

    for model in model_order:
        try:
            print(f"🔄 Attempting image generation with {model}...")

            if model == "imagen":
                provider = ImagenProvider(project_id=project_id)
                image = provider.generate(prompt)
                if image:
                    print(f"✅ Successfully generated image with {model}")
                    return image
                else:
                    print(f"❌ {model} failed: API returned None")

            elif model == "stability":
                provider = StabilityProvider()
                image = provider.generate(prompt)
                if image:
                    print(f"✅ Successfully generated image with {model}")
                    return image
                else:
                    print(f"❌ {model} failed: API returned None")

            elif model == "deepai":
                provider = DeepAIProvider()
                image = provider.generate(prompt)
                if image:
                    print(f"✅ Successfully generated image with {model}")
                    return image
                else:
                    print(f"❌ {model} failed: API returned None")

        except Exception as e:
            print(f"❌ {model} failed with exception: {e}")
            continue

    print(f"🚫 All image generation attempts failed")
    return None


def generate_images_from_phrases(
    phrases: List[str],
    style: str = None,
    bucket_name: Optional[str] = None,
    use_language_folder: bool = False,
    overwrite: bool = False,
    project_id: Optional[str] = None,
) -> Dict:
    """Generate images for a list of phrases and upload them to GCS.

    Args:
        phrases: List of English phrases
        style: Image style to apply (defaults to "ghibli")
        bucket_name: GCS bucket name (defaults to PRIVATE_BUCKET)
        use_language_folder: Whether to store images in language-specific folder
        overwrite: Whether to overwrite existing images in GCS
        project_id: GCP project ID for Imagen provider

    Returns:
        Dictionary mapping phrase keys to image data:
        {
            "phrase_key": {
                "phrase": str,
                "prompt": str,
                "gcs_uri": str,
            },
            ...
        }
    """
    if bucket_name is None:
        bucket_name = PRIVATE_BUCKET

    if style is None:
        style = "ghibli"

    results = {}

    for phrase in tqdm(phrases, desc="Generating phrase images"):
        clean_name = clean_filename(phrase)
        image_path = get_phrase_image_path(clean_name, use_language=use_language_folder)

        # Check if image already exists
        if not overwrite and check_blob_exists(bucket_name, image_path):
            print(f"Image already exists for '{phrase}', skipping generation")
            results[clean_name] = {
                "phrase": phrase,
                "prompt": None,
                "gcs_uri": f"gs://{bucket_name}/{image_path}",
            }
            continue

        # Generate prompt
        try:
            prompt = generate_phrase_image_prompt(phrase)
        except Exception as e:
            print(f"Error generating prompt for phrase '{phrase}': {e}")
            continue

        # Generate and upload the image
        try:
            image = generate_image(prompt, style=style, project_id=project_id)
            if image is None:
                print(f"Failed to generate image for phrase '{phrase}' with all providers")
                continue

            # Resize image to standard size
            image = resize_image(image, height=500, width=500)

            # Upload image to GCS
            gcs_uri = upload_file_to_gcs(
                obj=image,
                bucket_name=bucket_name,
                file_name=image_path,
                content_type="image/png",
            )

            results[clean_name] = {
                "phrase": phrase,
                "prompt": prompt,
                "gcs_uri": gcs_uri,
            }

            print(f"✅ Generated and uploaded image for '{phrase}' to {gcs_uri}")

        except Exception as e:
            print(f"Error processing phrase '{phrase}': {e}")
            continue

    print(f"\n🎯 Successfully generated {len(results)} images out of {len(phrases)} phrases")
    return results


def generate_and_save_story_images(
    story_dict: Dict[str, Dict],
    story_name: str,
    style: str = "ghibli",
    collection: str = "LM1000",
    project_id: Optional[str] = None,
    model_order: List[Literal["imagen", "stability", "deepai"]] = None,
) -> Dict[str, str]:
    """Generate and upload images for each part of a story to GCS.

    Args:
        story_dict: Dictionary containing story data
        story_name: Name of the story (used for filenames)
        style: Art style to apply
        collection: Collection name (default: "LM1000")
        project_id: GCP project ID for Imagen provider
        model_order: Order of image generation models to try

    Returns:
        Dict[str, str]: Mapping of story parts to GCS URIs
    """
    image_paths = {}

    for story_part, content in tqdm(story_dict.items(), desc="Generating story images"):
        # Construct GCS file path for this image: stories/{collection}/{story_name}/{story_part}.png
        gcs_image_path = f"stories/{collection}/{story_name}/{story_part}.png"

        # Check if image already exists in GCS
        if check_blob_exists(PRIVATE_BUCKET, gcs_image_path):
            print(f"Image already exists for {story_part}, skipping generation")
            image_paths[story_part] = f"gs://{PRIVATE_BUCKET}/{gcs_image_path}"
            continue

        # Generate prompt
        try:
            prompt = generate_story_image_prompt(content)
        except Exception as e:
            print(f"Error generating image prompt for {story_part}: {e}")
            continue

        # Try to generate image
        try:
            image = generate_image(
                prompt,
                style=style,
                project_id=project_id,
                model_order=model_order,
            )

            if image is None:
                print(f"Failed to generate image for {story_part} with all providers")
                continue

            # Resize the image
            image = resize_image(image)  # 500 x 500

            # Upload the image to GCS
            gcs_uri = upload_file_to_gcs(
                obj=image,
                bucket_name=PRIVATE_BUCKET,
                file_name=gcs_image_path,
                content_type="image/png",
            )
            image_paths[story_part] = gcs_uri
            print(f"Successfully generated and uploaded image for {story_part} to {gcs_uri}")

        except Exception as e:
            print(f"Error processing {story_part}: {e}")
            continue

    return image_paths


def add_image_paths(
    story_dict: Dict[str, any],
    image_dir: str
) -> Dict[str, any]:
    """Add image paths to story dictionary based on English phrases.

    Args:
        story_dict: Dictionary containing story data with translated_phrase_list
        image_dir: Directory containing the images

    Returns:
        Updated dictionary with image_path added for each story part

    Note:
        For each story part, expects translated_phrase_list to be a list of tuples
        where each tuple is (english_text, target_text)
    """
    updated_dict = copy.deepcopy(story_dict)

    for story_part, data in tqdm(updated_dict.items(), desc="Processing story parts"):
        # Initialize image_path list for this story part
        data["image_path"] = []

        # Get the phrases from translated_phrase_list
        phrase_list = data.get("translated_phrase_list", [])

        for eng_phrase, _ in tqdm(
            phrase_list, desc=f"Adding image paths for {story_part}", leave=False
        ):
            # Generate the expected image filename from English phrase
            clean_name = clean_filename(eng_phrase)
            image_filename = f"{clean_name}.png"
            full_path = os.path.join(image_dir, image_filename)

            # Check if the image exists and is readable
            if os.path.isfile(full_path) and os.access(full_path, os.R_OK):
                data["image_path"].append(full_path)
            else:
                print(f"Warning: Image not found or not readable: {full_path}")
                data["image_path"].append(None)

        # Verify lengths match
        if len(data["image_path"]) != len(phrase_list):
            raise ValueError(
                f"Mismatch in {story_part}: {len(data['image_path'])} images "
                f"vs {len(phrase_list)} phrases"
            )

    return updated_dict
