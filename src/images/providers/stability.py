"""Stability AI image generation provider."""

import io
import os
from typing import Literal, Optional

import requests
from PIL import Image

from .base import ImageProvider


class StabilityProvider(ImageProvider):
    """Stability AI image generation provider."""

    def __init__(
        self,
        endpoint: str = "https://api.stability.ai/v2beta/stable-image/generate/core",
    ):
        """Initialize Stability provider.

        Args:
            endpoint: API endpoint URL
        """
        self.endpoint = endpoint
        self.api_key = os.getenv("STABILITY_API_KEY")
        if not self.api_key:
            raise ValueError("STABILITY_API_KEY environment variable not set")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        style_preset: Optional[
            Literal[
                "3d-model",
                "analog-film",
                "anime",
                "cinematic",
                "comic-book",
                "digital-art",
                "enhance",
                "fantasy-art",
                "isometric",
                "line-art",
                "low-poly",
                "modeling-compound",
                "neon-punk",
                "origami",
                "photographic",
                "pixel-art",
                "tile-texture",
            ]
        ] = None,
        **kwargs,
    ) -> Optional[Image.Image]:
        """Generate an image using Stability AI API.

        Args:
            prompt: Text description of the image to generate
            negative_prompt: Text description of what to avoid in the image
            style_preset: Style preset to apply
            **kwargs: Additional arguments (ignored)

        Returns:
            Optional[Image.Image]: Generated PIL Image object, or None if generation fails
        """
        try:
            self._validate_prompt(prompt)

            # Prepare headers
            headers = {"Accept": "image/*", "Authorization": f"Bearer {self.api_key}"}

            # Prepare form data
            files = {
                "prompt": (None, prompt),
            }

            # Add optional parameters if provided
            if negative_prompt:
                files["negative_prompt"] = (None, negative_prompt)

            if style_preset:
                files["style_preset"] = (None, style_preset)

            # Make the API request
            response = requests.post(
                self.endpoint,
                headers=headers,
                files=files,
            )

            # Check if request was successful
            if response.status_code != 200:
                print(
                    f"Stability API error: {response.status_code} - {response.content}"
                )
                return None

            # Check for content filtering
            finish_reason = response.headers.get("finish-reason")
            if finish_reason == "CONTENT_FILTERED":
                print("Generation failed: NSFW classifier triggered")
                return None

            # Return the image
            return Image.open(io.BytesIO(response.content))

        except requests.RequestException as e:
            print(f"Stability API request failed: {e}")
            return None
        except Exception as e:
            print(f"Stability image generation failed: {e}")
            return None
