"""LLM tool for reviewing and refining story dialogue translations."""

import json
from typing import Any, Dict, List

from src.models import BCP47Language
from src.llm_tools.base import (
    load_prompt_template,
    get_anthropic_client,
    extract_tool_response,
    DEFAULT_MODEL,
)

def return_story_part_schema() -> Dict[str, Any]:
    return {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "text": {"type": "string"},
                        "translation": {"type": "string"},
                    },
                    "required": ["speaker", "text", "translation"],
                },
            }
# Tool definition
TOOL_SCHEMA = {
    "name": "refine_story_translation",
    "description": "Review and improve translations for a story's dialogue, organized by parts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "part_1": return_story_part_schema(),
            "part_2": return_story_part_schema(),
            "part_3": return_story_part_schema(),
        },
        "required": ["part_1", "part_2", "part_3"],
    }
}


def refine_story_translation(
    story_parts: Dict[str, Any],
    language: BCP47Language,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> Dict[str, List[dict[str, str]]]:
    """Review and improve story dialogue translations using Claude API.

    Args:
        story_parts: Dictionary of story parts, each containing a list of dialogue items.
        language: Target language as a BCP47Language object.
        model: Anthropic model to use.
        max_tokens: Maximum tokens for response.
        temperature: Temperature for generation.

    Returns:
        Modified dictionary with improved translations.

    Raises:
        RuntimeError: If review fails.
    """
    try:
        # Load prompts from template files
        system_template = load_prompt_template("refine_story_translation", "system")
        user_template = load_prompt_template("refine_story_translation", "user")

        # Substitute variables
        system_prompt = system_template.substitute(
            target_language_name=language.display_name(),
            source_language_name="English",
        )
        user_prompt = user_template.substitute(
            story_json=json.dumps(story_parts, indent=2)
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
                "name": "refine_story_translation",
            },
        )

        # Extract tool response
        tool_input = extract_tool_response(response, "refine_story_translation")
        if tool_input:
            return tool_input

        # If we didn't get a tool response, return empty dictionary
        print("Warning: No tool response found, returning empty dictionary")
        return {"story_parts": {}}

    except Exception as e:
        raise RuntimeError(f"Failed to review story translations with Anthropic: {e}")
