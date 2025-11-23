"""Base class for image generation providers."""

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class ImageProvider(ABC):
    """Abstract base class for image generation providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> Optional[Image.Image]:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image to generate
            **kwargs: Provider-specific parameters

        Returns:
            Optional[Image.Image]: Generated PIL Image object, or None if generation fails
        """
        pass

    @staticmethod
    def _validate_prompt(prompt: str) -> None:
        """Validate that prompt is not empty.

        Args:
            prompt: The prompt to validate

        Raises:
            ValueError: If prompt is empty or invalid
        """
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
