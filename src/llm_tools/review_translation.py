"""LLM tool for reviewing and refining a single translation."""

from typing import Optional

from src.llm_tools.base import (
    load_prompt_template,
    get_anthropic_client,
    extract_tool_response,
    DEFAULT_MODEL,
)


# Tool definition
TOOL_SCHEMA = {
    "name": "review_translation",
    "description": "Review and improve a translation from English to target language",
    "input_schema": {
        "type": "object",
        "properties": {
            "translation": {"type": "string"},
            "modified": {"type": "boolean"},
        },
        "required": ["translation", "modified"],
    },
}


def refine_translation(
    english_phrase: str,
    initial_translation: str,
    target_language_name: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.2
) -> str:
    """Refine a translation using Claude API.

    Args:
        english_phrase: The English phrase being translated
        initial_translation: The initial Google Translate translation
        target_language_name: Display name of target language (e.g., "French")
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for generation

    Returns:
        str: The refined translation text

    Raises:
        RuntimeError: If refinement fails
    """
    try:
        # Load prompts from template files
        system_template = load_prompt_template("review_translation", "system")
        user_template = load_prompt_template("review_translation", "user")

        # Substitute variables
        system_prompt = system_template.substitute(
            target_language_name=target_language_name
        )
        user_prompt = user_template.substitute(
            english_phrase=english_phrase,
            target_language_name=target_language_name,
            initial_translation=initial_translation
        )

        # Get Anthropic client and make API call
        client = get_anthropic_client()
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            tools=[TOOL_SCHEMA],
            tool_choice={
                "type": "tool",
                "name": "review_translation",
            },
        )

        # Extract tool response
        tool_input = extract_tool_response(response, "review_translation")
        if tool_input:
            return tool_input["translation"]

        # If we didn't get a tool response, return the original translation
        print("Warning: No tool response found, returning original translation")
        return initial_translation

    except Exception as e:
        raise RuntimeError(f"Failed to refine translation with Anthropic: {e}")
