"""Vertex AI Imagen image generation provider."""

import io
import os
from typing import Literal, Optional

import vertexai
from PIL import Image
from vertexai.preview.vision_models import ImageGenerationModel
from src.connections.gcloud_auth import setup_authentication, _project_id
from src.images.providers.base import ImageProvider


class ImagenProvider(ImageProvider):
    """Vertex AI Imagen image generation provider."""

    def __init__(
        self,
        model: Literal[
            "imagen-4.0-generate-001",
            "imagen-4.0-fast-generate-001",
            "imagen-4.0-ultra-generate-001",
            "imagen-3.0-generate-002",
            "imagen-3.0-generate-001",
            "imagen-3.0-fast-generate-001",
        ] = "imagen-4.0-generate-001",
        region: Optional[str] = None,
    ):
        """Initialize Imagen provider.

        Args:
            model: Imagen model version to use (defaults to imagen-3.0-generate-002)
            project_id: GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var if not provided)
            region: GCP region (defaults to "us-central1" if not provided)

        Raises:
            ValueError: If project_id cannot be determined
        """
        self.model = model
        if _project_id is None:
            credentials, project_id = setup_authentication()
        else:
            project_id = _project_id
        # Get project ID from parameter, environment, or raise error
        self.project_id = project_id
        if not self.project_id:
            raise ValueError(
                "project_id must be provided or GOOGLE_CLOUD_PROJECT environment variable must be set"
            )

        # Get region from parameter or environment, default to us-central1
        self.region = region or os.getenv("VERTEX_REGION", "us-central1")

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        safety_filter_level: Optional[str] = None,
        **kwargs,
    ) -> Optional[Image.Image]:
        """Generate an image using Vertex AI Imagen.

        Args:
            prompt: Text description of the image to generate
            aspect_ratio: Image aspect ratio (default "1:1")
            safety_filter_level: Safety filter level (optional)
            **kwargs: Additional arguments (ignored)

        Returns:
            Optional[Image.Image]: Generated PIL Image object, or None if generation fails
        """
        try:
            self._validate_prompt(prompt)

            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.region)
            generation_model = ImageGenerationModel.from_pretrained(self.model)

            # Generate the image
            images = generation_model.generate_images(
                prompt=prompt,
                negative_prompt="text, words",
                number_of_images=1,
                aspect_ratio=aspect_ratio,
            )

            if len(images.images) > 0:
                return Image.open(io.BytesIO(images.images[0]._image_bytes))
            else:
                print(f"No image generated using {self.model} with prompt: {prompt}")
                return None

        except Exception as e:
            print(f"Imagen generation failed: {e}")
            return None
