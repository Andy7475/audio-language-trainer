"""Challenge generation LLM tool for creating language learning roleplay scenarios."""

from typing import Any, Dict, List

from src.llm_tools.base import (
    DEFAULT_MODEL,
    get_anthropic_client,
    extract_tool_response,
    load_prompt_template,
)
from src.models import BCP47Language


# Fixed tool schema - always 5 scenarios with 3 complications each
TOOL_SCHEMA = {
    "name": "generate_challenges",
    "description": "Generate language learning roleplay scenarios based on a story dialogue",
    "input_schema": {
        "type": "object",
        "properties": {
            "scenarios": {
                "type": "array",
                "description": "List of 5 roleplay scenarios",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "description": "Role for teacher to play (e.g., 'coffee shop staff')",
                        },
                        "context": {
                            "type": "string",
                            "description": "Brief setting description",
                        },
                        "challenge": {
                            "type": "string",
                            "description": "Main task to complete (e.g., 'Order a coffee')",
                        },
                        "information_task": {
                            "type": "string",
                            "description": "Specific information to discover (e.g., 'What is the price?')",
                        },
                        "complications": {
                            "type": "array",
                            "description": "Three realistic complications that could arise",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                        "success_criteria": {
                            "type": "string",
                            "description": "What constitutes successful completion",
                        },
                    },
                    "required": [
                        "role",
                        "context",
                        "challenge",
                        "information_task",
                        "complications",
                        "success_criteria",
                    ],
                },
                "minItems": 5,
                "maxItems": 5,
            }
        },
        "required": ["scenarios"],
    },
}


def generate_challenges(
    story_dialogue: Dict,
    target_language: BCP47Language,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 3000,
    temperature: float = 0.4,
) -> Dict:
    """Generate roleplay challenges based on a story dialogue.

    Args:
        story_dialogue: Dictionary with story parts as keys, dialogue as values
        target_language: BCP47Language for the target language (challenges will reference this)
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Generation temperature

    Returns:
        Dict with structure:
        {
            "scenarios": [
                {
                    "role": str,
                    "context": str,
                    "challenge": str,
                    "information_task": str,
                    "complications": [str, str, str],
                    "success_criteria": str
                },
                ... (5 total scenarios)
            ]
        }

    Raises:
        RuntimeError: If challenge generation fails
        ValueError: If response doesn't contain exactly 5 scenarios

    Examples:
        >>> import langcodes
        >>> target_lang = langcodes.get("fr-FR")
        >>> story_dialogue = {
        ...     "setup": {"dialogue": [{"speaker": "Alex", "text": "Hello"}]},
        ...     "resolution": {"dialogue": [{"speaker": "Sam", "text": "Goodbye"}]}
        ... }
        >>> result = generate_challenges(story_dialogue, target_lang)
        >>> len(result["scenarios"])
        5
        >>> all(len(s["complications"]) == 3 for s in result["scenarios"])
        True
    """
    try:
        # Extract dialogue text from all story parts
        dialogue_lines = []
        for part in story_dialogue.values():
            if "dialogue" in part:
                for utterance in part["dialogue"]:
                    speaker = utterance.get("speaker", "")
                    text = utterance.get("text", "")
                    dialogue_lines.append(f"{speaker}: {text}")

        story_context = "\n".join(dialogue_lines)

        if not story_context.strip():
            raise ValueError("Story dialogue is empty - cannot generate challenges")

        # Get target language display name
        target_language_name = target_language.display_name()

        # Load prompt templates
        system_template = load_prompt_template("challenge_generation", "system")
        user_template = load_prompt_template("challenge_generation", "user")

        # Substitute variables
        system_prompt = system_template.substitute(
            target_language_name=target_language_name
        )
        user_prompt = user_template.substitute(
            target_language_name=target_language_name,
            story_context=story_context,
        )

        # Get Anthropic client and create message
        client = get_anthropic_client()
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "generate_challenges"},
        )

        # Extract tool response
        tool_input = extract_tool_response(response, "generate_challenges")
        if not tool_input:
            raise RuntimeError(
                "No tool response found from Anthropic API - challenge generation failed"
            )

        # Validate response structure
        scenarios = tool_input.get("scenarios", [])
        if len(scenarios) != 5:
            raise ValueError(
                f"Expected exactly 5 scenarios, got {len(scenarios)}"
            )

        # Validate each scenario has 3 complications
        for i, scenario in enumerate(scenarios):
            complications = scenario.get("complications", [])
            if len(complications) != 3:
                raise ValueError(
                    f"Scenario {i+1} has {len(complications)} complications, expected 3"
                )

        return tool_input

    except Exception as e:
        raise RuntimeError(f"Failed to generate challenges: {e}")
