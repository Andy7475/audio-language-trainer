"""Image style management and application."""

import json
from pathlib import Path
from typing import Dict, Optional


def load_image_styles(styles_file: Optional[Path] = None) -> Dict[str, str]:
    """Load image styles from JSON file.

    Args:
        styles_file: Path to styles JSON file. If None, uses default location.

    Returns:
        Dict[str, str]: Dictionary mapping style names to style descriptions

    Raises:
        FileNotFoundError: If styles file doesn't exist
        json.JSONDecodeError: If styles file is invalid JSON
    """
    if styles_file is None:
        styles_file = Path(__file__).parent / "styles.json"

    if not styles_file.exists():
        raise FileNotFoundError(f"Styles file not found: {styles_file}")

    with open(styles_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_style_description(style: str, styles: Optional[Dict[str, str]] = None) -> str:
    """Get style description, falling back to the style name if not found.

    Args:
        style: Style name or key to look up
        styles: Pre-loaded styles dictionary. If None, will load from default file.

    Returns:
        str: Style description or the original style string if not found

    Raises:
        FileNotFoundError: If styles dictionary can't be loaded
    """
    if styles is None:
        styles = load_image_styles()

    # Look up in styles dictionary, case-insensitive
    style_lower = style.lower()
    if style_lower in styles:
        return styles[style_lower]

    # If not found, return the original style string
    return style


def add_image_style(prompt: str, style: str, styles: Optional[Dict[str, str]] = None) -> str:
    """Add an art style to an image generation prompt.

    Args:
        prompt: The image generation prompt (without style)
        style: Style name (key in styles.json) or custom style description
        styles: Pre-loaded styles dictionary. If None, will load from default file.

    Returns:
        str: The prompt with style appended

    Raises:
        FileNotFoundError: If styles dictionary can't be loaded
    """
    # Remove any trailing periods and whitespace
    prompt = prompt.rstrip(". ")

    # Get the style description
    style_description = get_style_description(style, styles)

    # Append style to prompt
    return f"{prompt} in the style of {style_description}"
