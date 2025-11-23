"""Image generation and management module."""

from src.images.generator import (
    generate_image,
    generate_images_from_phrases,
    generate_and_save_story_images,
    add_image_paths,
)
from src.images.styles import add_image_style, load_image_styles, get_style_description
from src.images.manipulation import resize_image, create_png_of_html
from src.images.providers import ImagenProvider, StabilityProvider, DeepAIProvider

__all__ = [
    # Generator functions
    "generate_image",
    "generate_images_from_phrases",
    "generate_and_save_story_images",
    "add_image_paths",
    # Style management
    "add_image_style",
    "load_image_styles",
    "get_style_description",
    # Image manipulation
    "resize_image",
    "create_png_of_html",
    # Providers
    "ImagenProvider",
    "StabilityProvider",
    "DeepAIProvider",
]
