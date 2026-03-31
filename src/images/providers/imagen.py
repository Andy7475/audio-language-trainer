"""Vertex AI Imagen image generation provider using the google-genai SDK."""

import io
import os
from typing import Literal, Optional

from google import genai
from google.genai import types
from PIL import Image
from src.connections.gcloud_auth import setup_authentication, _project_id
from src.images.providers.base import ImageProvider


class ImagenProvider(ImageProvider):
    """Vertex AI Imagen image generation provider."""

    def __init__(
        self,
        model: Literal[
            "gemini-2.5-flash-image",
        ] = "gemini-2.5-flash-image",
        region: Optional[str] = None,
    ):
        """Initialize Imagen provider.

        Args:
            model: Model version to use (defaults to gemini-2.5-flash-image)
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
        """Generate an image using the Gemini image generation API via Vertex AI.

        Args:
            prompt: Text description of the image to generate
            aspect_ratio: Image aspect ratio (default "1:1") — passed via kwargs if supported
            safety_filter_level: Safety filter level (optional, unused for Gemini models)
            **kwargs: Additional arguments (ignored)

        Returns:
            Optional[Image.Image]: Generated PIL Image object, or None if generation fails
        """
        try:
            self._validate_prompt(prompt)

            # Initialise the google-genai client pointed at Vertex AI
            client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.region,
            )

            # Generate the image
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=[types.Modality.IMAGE],
                ),
            )

            # Extract the first image part from the response
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    return Image.open(io.BytesIO(part.inline_data.data))

            print(f"No image generated using {self.model} with prompt: {prompt}")
            return None

        except Exception as e:
            print(f"Imagen generation failed: {e}")
            return None
