"""LLM tool for generating verb-based phrases for language learners."""

from typing import Any

from src.base import (
    load_prompt_template,
    get_anthropic_client,
    extract_tool_response,
    DEFAULT_MODEL,
)


# Tool definition for verb phrase generation
TOOL_SCHEMA = {
    "name": "generate_verb_phrases",
    "description": "Generate English phrases featuring a specific verb in different tenses and meanings",
    "input_schema": {
        "type": "object",
        "properties": {
            "base_phrases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "phrase": {"type": "string"},
                        "additional_words": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["phrase", "additional_words"],
                },
            },
            "meaning_variations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "phrase": {"type": "string"},
                        "meaning": {"type": "string"},
                        "additional_words": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["phrase", "meaning", "additional_words"],
                },
            },
        },
        "required": ["base_phrases", "meaning_variations"],
    },
}


def generate_verb_phrases(
    verb: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1500,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Generate phrases featuring a specific verb in different tenses and meanings.

    Generates base phrases (present, past, future tenses) and additional phrases
    showing different meanings of the verb. Also extracts additional words used
    in the phrases to support vocabulary list management.

    Args:
        verb: The English verb to generate phrases for (e.g., "run", "want", "break")
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for generation

    Returns:
        Dict containing:
            {
                "verb": str,
                "base_phrases": [
                    {
                        "phrase": str,
                        "additional_words": [str]
                    },
                    ...
                ],
                "meaning_variations": [
                    {
                        "phrase": str,
                        "meaning": str,
                        "additional_words": [str]
                    },
                    ...
                ],
                "all_additional_words": [str]  # Flattened list of all additional words
            }

    Raises:
        RuntimeError: If phrase generation fails
    """
    try:
        # Load prompts from template files
        system_template = load_prompt_template("verb_phrase_generation", "system")
        user_template = load_prompt_template("verb_phrase_generation", "user")

        # Substitute variables
        system_prompt = system_template.substitute()
        user_prompt = user_template.substitute(verb=verb)

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
                "name": "generate_verb_phrases",
            },
        )

        # Extract tool response
        tool_input = extract_tool_response(response, "generate_verb_phrases")
        if not tool_input:
            raise RuntimeError("No tool response found from Anthropic API")

        # Flatten additional words
        all_additional_words = []
        for phrase_data in tool_input.get("base_phrases", []):
            all_additional_words.extend(phrase_data.get("additional_words", []))
        for phrase_data in tool_input.get("meaning_variations", []):
            all_additional_words.extend(phrase_data.get("additional_words", []))

        return {
            "verb": verb,
            "base_phrases": tool_input.get("base_phrases", []),
            "meaning_variations": tool_input.get("meaning_variations", []),
            "all_additional_words": list(
                set(all_additional_words)
            ),  # Remove duplicates
        }

    except Exception as e:
        raise RuntimeError(f"Failed to generate phrases for verb '{verb}': {e}")
