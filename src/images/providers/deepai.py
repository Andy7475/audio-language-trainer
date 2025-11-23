"""DeepAI image generation provider."""

import io
import os
from typing import Literal, Optional, Union

import requests
from PIL import Image

from src.images.providers.base import ImageProvider


class DeepAIProvider(ImageProvider):
    """DeepAI text2img image generation provider."""

    def __init__(self):
        """Initialize DeepAI provider."""
        self.api_key = os.getenv("DEEPAI_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPAI_API_KEY environment variable not set")

    def generate(
        self,
        prompt: str,
        width: Union[str, int] = "512",
        height: Union[str, int] = "512",
        model: Literal["standard", "hd"] = "hd",
        negative_prompt: Optional[str] = None,
        **kwargs
    ) -> Optional[Image.Image]:
        """Generate an image using DeepAI's text2img API.

        Args:
            prompt: Text description of the image to generate
            width: Image width in pixels (128-1536, default 512)
            height: Image height in pixels (128-1536, default 512)
            model: Model version ("standard" or "hd")
            negative_prompt: Text describing what to avoid in the image
            **kwargs: Additional arguments (ignored)

        Returns:
            Optional[Image.Image]: Generated PIL Image object, or None if generation fails
        """
        try:
            self._validate_prompt(prompt)

            # Convert width and height to strings if they're integers
            width = str(width)
            height = str(height)

            # Prepare the API request data
            data = {
                "text": prompt,
                "width": width,
                "height": height,
                "image_generator_version": model,
            }

            # Add negative prompt if provided
            if negative_prompt:
                data["negative_prompt"] = negative_prompt

            # Make the API request
            response = requests.post(
                "https://api.deepai.org/api/text2img",
                data=data,
                headers={"api-key": self.api_key},
            )

            # Check if the request was successful
            response.raise_for_status()

            # Get the URL of the generated image from the response
            result = response.json()
            if "output_url" not in result:
                print(f"Unexpected DeepAI API response: {result}")
                return None

            # Download the image from the URL
            image_response = requests.get(result["output_url"])
            image_response.raise_for_status()

            # Convert to PIL Image
            return Image.open(io.BytesIO(image_response.content))

        except requests.exceptions.RequestException as e:
            print(f"DeepAI API request failed: {e}")
            return None
        except Exception as e:
            print(f"DeepAI image generation failed: {e}")
            return None
