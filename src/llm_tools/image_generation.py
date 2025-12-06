"""LLM tool for generating image prompts from language learning phrases."""

from src.llm_tools.base import (
    load_prompt_template,
    get_anthropic_client,
    DEFAULT_MODEL,
)


def generate_phrase_image_prompt(
    phrase: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 500,
    temperature: float = 0.7,
) -> str:
    """Generate an image prompt for a language learning phrase.

    Args:
        phrase: The English phrase to create an image prompt for
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for generation

    Returns:
        str: A detailed image generation prompt

    Raises:
        RuntimeError: If prompt generation fails
    """
    try:
        # Load prompts from template files
        system_template = load_prompt_template("image_generation", "system")
        user_template = load_prompt_template("image_generation", "user")

        # Substitute variables
        system_prompt = system_template.substitute()
        user_prompt = user_template.substitute(phrase=phrase)

        # Get Anthropic client and make API call
        client = get_anthropic_client()
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract text from response
        if response.content and len(response.content) > 0:
            image_prompt = response.content[0].text.strip()
            # Clean up any surrounding quotes or periods
            image_prompt = image_prompt.strip("\"'.")
            return image_prompt

        raise RuntimeError("No response content from Anthropic API")

    except Exception as e:
        raise RuntimeError(f"Failed to generate image prompt: {e}")
