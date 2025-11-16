"""LLM tool for reviewing and refining batch translations."""

from typing import Any, Dict, List

from src.llm_tools.base import (
    load_prompt_template,
    get_anthropic_client,
    extract_tool_response
)


# Tool definition
TOOL_SCHEMA = {
    "name": "review_translations",
    "description": "Review and improve translations from source language to target language",
    "input_schema": {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "english": {"type": "string"},
                        "translation": {"type": "string"},
                        "modified": {"type": "boolean"},
                    },
                    "required": ["english", "translation", "modified"],
                },
            }
        },
        "required": ["translations"],
    },
}


def review_batch_translations(
    phrase_pairs: List[Dict[str, str]],
    target_language_name: str = "target language",
    source_language_name: str = "English",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4000,
    temperature: float = 0.2
) -> List[Dict[str, Any]]:
    """Review and improve a batch of translations using Claude API.

    Args:
        phrase_pairs: List of dicts with 'english' and 'translation' keys
        target_language_name: Display name of target language (e.g., "French")
        source_language_name: Display name of source language (default: "English")
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for generation

    Returns:
        List of dicts with 'english', 'translation', and 'modified' keys

    Raises:
        RuntimeError: If review fails
    """
    try:
        # Load prompts from template files
        system_template = load_prompt_template("review_translations", "system")
        user_template = load_prompt_template("review_translations", "user")

        # Format phrase pairs for the prompt
        formatted_pairs = "\n".join([
            f"English: {pair['english']}\n{target_language_name}: {pair['translation']}\n"
            for pair in phrase_pairs
        ])

        # Substitute variables
        system_prompt = system_template.substitute(
            target_language_name=target_language_name,
            source_language_name=source_language_name
        )
        user_prompt = user_template.substitute(
            formatted_pairs=formatted_pairs
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
                "name": "review_translations",
            },
        )

        # Extract tool response
        tool_input = extract_tool_response(response, "review_translations")
        if tool_input:
            return tool_input["translations"]

        # If we didn't get a tool response, return empty list
        print("Warning: No tool response found, returning empty list")
        return []

    except Exception as e:
        raise RuntimeError(f"Failed to review translations with Anthropic: {e}")
