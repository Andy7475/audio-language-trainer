"""LLM tool for generating image prompts from story dialogue."""

from typing import Dict, List, Union

from src.llm_tools.base import (
    load_prompt_template,
    get_anthropic_client,
    DEFAULT_MODEL,
)


def generate_story_image_prompt(
    story_part: Union[Dict, List[Dict]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 500,
    temperature: float = 0.7
) -> str:
    """Generate an image prompt from a story part containing dialogue.

    Args:
        story_part: Either a dictionary containing a 'dialogue' key, or a list of such dictionaries.
            Each dialogue entry should be a list of speaker/text pairs.
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for generation

    Returns:
        str: A detailed image generation prompt for the scene

    Raises:
        RuntimeError: If prompt generation fails
    """
    try:
        # Convert single part to list for consistent processing
        story_parts = story_part if isinstance(story_part, list) else [story_part]

        # Extract all dialogue text, removing speaker information
        all_dialogue = []
        for part in story_parts:
            if "dialogue" in part:
                all_dialogue.extend([utterance["text"] for utterance in part["dialogue"]])

        dialogue_text = " ".join(all_dialogue)

        # Load prompts from template files
        system_template = load_prompt_template("story_image_generation", "system")
        user_template = load_prompt_template("story_image_generation", "user")

        # Substitute variables
        system_prompt = system_template.substitute()
        user_prompt = user_template.substitute(dialogue=dialogue_text)

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
            image_prompt = image_prompt.strip('"\'.')
            return image_prompt

        raise RuntimeError("No response content from Anthropic API")

    except Exception as e:
        raise RuntimeError(f"Failed to generate story image prompt: {e}")
