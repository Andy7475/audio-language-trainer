"""Image manipulation and conversion utilities."""

import os
import time
from pathlib import Path

from PIL import Image


def resize_image(image: Image.Image, height: int = 500, width: int = 500) -> Image.Image:
    """Resize a PIL Image to specified dimensions if needed.

    Args:
        image: PIL Image object
        height: Desired height (default 500)
        width: Desired width (default 500)

    Returns:
        PIL.Image.Image: Image with desired dimensions
    """
    if image.size != (width, height):
        image = image.resize((width, height))
    return image


def create_png_of_html(
    path_to_html: str,
    output_path: str = None,
    width: int = 375,
    height: int = 1100
) -> str:
    """Render an HTML file at smartphone screen size and save it as PNG using Selenium.

    Args:
        path_to_html: Path to the HTML file to render
        output_path: Path where the PNG should be saved. If None, uses the HTML filename with .png extension
        width: Width of the smartphone screen in pixels (default: iPhone X width = 375)
        height: Height of the smartphone screen in pixels (default: 1100)

    Returns:
        str: Path to the saved PNG file

    Raises:
        ImportError: If Selenium or required drivers are not available
        Exception: If rendering fails
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    if output_path is None:
        # Use the same filename but with .png extension
        output_path = os.path.splitext(path_to_html)[0] + ".png"

    # Setup Chrome in headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument(f"--window-size={width},{height}")

    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Convert to absolute path for file:// URL
        abs_path = os.path.abspath(path_to_html)
        driver.get(f"file://{abs_path}")

        # Wait for any JS to render
        driver.implicitly_wait(2)
        time.sleep(0.5)

        # Take screenshot and save
        driver.save_screenshot(output_path)

        return output_path
    finally:
        # Always clean up
        driver.quit()
