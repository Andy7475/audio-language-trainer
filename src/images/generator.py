"""High-level image generation orchestration."""

from typing import List, Literal, Optional

from PIL import Image

from src.images.providers import ImagenProvider, StabilityProvider, DeepAIProvider
from src.images.styles import add_image_style, load_image_styles


def generate_image(
    prompt: str,
    style: str = "default",
    model_order: List[Literal["imagen", "stability", "deepai"]] = None,
) -> Optional[Image.Image]:
    """Generate an image using multiple providers in specified order.

    Args:
        prompt: The image generation prompt
        style: Art style to apply (defaults to "picture book illustration")
        model_order: List of models to try in order (default: ["imagen", "stability", "deepai"])

    Returns:
        Optional[Image.Image]: Generated image or None if all attempts fail
    """
    if model_order is None:
        model_order = ["imagen", "stability", "deepai"]

    # Apply style to prompt
    try:
        styles_dict = load_image_styles()
        prompt = add_image_style(prompt, style, styles_dict)
    except Exception as e:
        print(f"Warning: Could not apply style '{style}': {e}")
        # Continue with unstyled prompt

    print("🎨 Starting image generation process")
    print(f"   Prompt: {prompt}")
    print(f"   Will try providers in order: {model_order}")

    for model in model_order:
        try:
            print(f"🔄 Attempting image generation with {model}...")

            if model == "imagen":
                provider = ImagenProvider()
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

    print("🚫 All image generation attempts failed")
    return None
