"""LLM tool for reviewing and refining story dialogue translations."""

import json
from typing import Any, Dict, List

from src.llm_tools.base import (
    load_prompt_template,
    get_anthropic_client,
    extract_tool_response,
    DEFAULT_MODEL,
)


# Tool definition
TOOL_SCHEMA = {
    "name": "review_story_translations",
    "description": "Review and improve translations from source language to target language for a story's dialogue",
    "input_schema": {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": ["Sam", "Alex"]},
                        "text": {"type": "string"},
                        "translation": {"type": "string"},
                        "modified": {"type": "boolean"},
                    },
                    "required": ["speaker", "text", "translation", "modified"],
                },
            }
        },
        "required": ["translations"],
    },
}


def review_story_dialogue(
    flattened_dialogue: List[Dict[str, Any]],
    target_language_name: str = "target language",
    source_language_name: str = "English",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> List[Dict[str, Any]]:
    """Review and improve story dialogue translations using Claude API.

    Args:
        flattened_dialogue: List of dialogue items with 'speaker', 'text', 'translation' keys
        target_language_name: Display name of target language (e.g., "French")
        source_language_name: Display name of source language (default: "English")
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for generation

    Returns:
        List of dicts with 'speaker', 'text', 'translation', 'modified' keys

    Raises:
        RuntimeError: If review fails
        ValueError: If speakers are not 'Sam' or 'Alex'
    """
    # Validate speakers
    for item in flattened_dialogue:
        if item.get("speaker") not in ["Sam", "Alex"]:
            raise ValueError(
                f"Invalid speaker: {item.get('speaker')}. Must be either 'Sam' or 'Alex'"
            )

    try:
        # Load prompts from template files
        system_template = load_prompt_template("review_story_translations", "system")
        user_template = load_prompt_template("review_story_translations", "user")

        # Substitute variables
        system_prompt = system_template.substitute(
            target_language_name=target_language_name,
            source_language_name=source_language_name,
        )
        user_prompt = user_template.substitute(
            story_json=json.dumps(flattened_dialogue, indent=2)
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
                "name": "review_story_translations",
            },
        )

        # Extract tool response
        tool_input = extract_tool_response(response, "review_story_translations")
        if tool_input:
            reviewed_translations = tool_input["translations"]

            # Validate speakers in response
            for item in reviewed_translations:
                if item["speaker"] not in ["Sam", "Alex"]:
                    raise ValueError(
                        f"Invalid speaker in response: {item['speaker']}. Must be either 'Sam' or 'Alex'"
                    )

            return reviewed_translations

        # If we didn't get a tool response, return empty list
        print("Warning: No tool response found, returning empty list")
        return []

    except Exception as e:
        raise RuntimeError(f"Failed to review story translations with Anthropic: {e}")
