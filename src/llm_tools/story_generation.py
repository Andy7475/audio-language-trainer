"""Story generation LLM tool for creating English dialogue-based stories."""

from typing import Any, Dict, List, Tuple

from src.llm_tools.base import (
    DEFAULT_MODEL,
    get_anthropic_client,
    extract_tool_response,
    load_prompt_template,
)


def _build_story_schema(verb_count: int) -> Dict[str, Any]:
    """Build dynamic tool schema based on verb count.

    Args:
        verb_count: Number of verbs to incorporate

    Returns:
        Tool schema dictionary
    """
    # Base dialogue item schema (reused across all parts)
    dialogue_item_schema = {
        "type": "object",
        "properties": {
            "speaker": {"type": "string", "enum": ["Alex", "Sam"]},
            "text": {"type": "string"},
        },
        "required": ["speaker", "text"],
    }

    # Part schema template
    def part_schema():
        return {
            "type": "object",
            "properties": {
                "dialogue": {
                    "type": "array",
                    "items": dialogue_item_schema,
                }
            },
            "required": ["dialogue"],
        }

    # Build schema based on verb count
    if verb_count < 10:
        # Single "story" part
        properties = {
            "story_name": {
                "type": "string",
                "description": "Engaging 3-word title for the story",
            },
            "story": part_schema(),
        }
        required = ["story_name", "story"]

    elif verb_count < 30:
        # Two parts: setup, resolution
        properties = {
            "story_name": {
                "type": "string",
                "description": "Engaging 3-word title for the story",
            },
            "setup": part_schema(),
            "resolution": part_schema(),
        }
        required = ["story_name", "setup", "resolution"]

    else:
        # Three parts: introduction, development, resolution
        properties = {
            "story_name": {
                "type": "string",
                "description": "Engaging 3-word title for the story",
            },
            "introduction": part_schema(),
            "development": part_schema(),
            "resolution": part_schema(),
        }
        required = ["story_name", "introduction", "development", "resolution"]

    return {
        "name": "generate_story",
        "description": "Generate an English dialogue-based story for language learning",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _get_structure_description(verb_count: int) -> Tuple[str, int, str]:
    """Get story structure details based on verb count.

    Args:
        verb_count: Number of verbs

    Returns:
        Tuple of (description, part_count, target_length)
    """
    if verb_count < 10:
        description = "A single focused scene showing natural interaction"
        part_count = 1
        target_length = "30-45 seconds (about 75-100 words)"
    elif verb_count < 30:
        description = """Two parts:
   - setup: Establish the situation and context
   - resolution: Complete the interaction"""
        part_count = 2
        target_length = "~1 minute per part (about 100-150 words each)"
    else:
        description = """Three parts:
   - introduction: Set up the situation and characters
   - development: Present a challenge or complication
   - resolution: Resolve the situation"""
        part_count = 3
        target_length = "1-2 minutes per part (about 150-300 words each)"

    return description, part_count, target_length


def generate_story(
    verbs: List[str],
    vocab: List[str],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4000,
    temperature: float = 0.3,
) -> Tuple[str, Dict]:
    """Generate an English dialogue-based story for language learning.

    Stories are ALWAYS generated in English. Translation happens separately.

    Dynamically determines story structure based on verb count:
    - <10 verbs: Single "story" part
    - 10-29 verbs: "setup" and "resolution" parts
    - >=30 verbs: "introduction", "development", "resolution" parts

    Args:
        verbs: List of verbs to incorporate in story
        vocab: List of other vocabulary words to use
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Generation temperature

    Returns:
        Tuple of (story_name, story_dialogue)
        where story_dialogue has keys for each story part containing dialogue

    Raises:
        RuntimeError: If story generation fails
        ValueError: If response doesn't match expected structure

    Examples:
        >>> story_name, dialogue = generate_story(
        ...     verbs=["go", "see", "want"],
        ...     vocab=["coffee", "table", "friend"]
        ... )
        >>> story_name
        'Coffee Shop Meeting'
        >>> "story" in dialogue  # Single part for <10 verbs
        True
    """
    try:
        verb_count = len(verbs)
        vocab_count = len(vocab)

        # Build dynamic schema based on verb count
        story_schema = _build_story_schema(verb_count)

        # Get structure description for prompt
        structure_desc, part_count, target_length = _get_structure_description(
            verb_count
        )

        # Format verb and vocab lists for prompt
        verbs_str = ", ".join(verbs)
        vocab_str = ", ".join(vocab)

        # Load prompt templates
        system_template = load_prompt_template("story_generation", "system")
        user_template = load_prompt_template("story_generation", "user")

        # Substitute variables
        system_prompt = system_template.substitute()
        user_prompt = user_template.substitute(
            verbs=verbs_str,
            vocab=vocab_str,
            verb_count=str(verb_count),
            vocab_count=str(vocab_count),
            structure_description=structure_desc,
            part_count=str(part_count),
            target_length=target_length,
        )

        # Get Anthropic client and create message
        client = get_anthropic_client()
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            tools=[story_schema],
            tool_choice={"type": "tool", "name": "generate_story"},
        )

        # Extract tool response
        tool_input = extract_tool_response(response, "generate_story")
        if not tool_input:
            raise RuntimeError(
                "No tool response found from Anthropic API - story generation failed"
            )

        # Extract story name and dialogue
        story_name = tool_input.get("story_name")
        if not story_name:
            raise ValueError("Missing story_name in response")

        # Build dialogue dictionary (exclude story_name)
        story_dialogue = {k: v for k, v in tool_input.items() if k != "story_name"}

        # Validate that we have at least one dialogue part
        if not story_dialogue:
            raise ValueError("No dialogue parts found in response")

        # Validate structure matches expected part count
        expected_parts = (
            ["story"]
            if verb_count < 10
            else (
                ["setup", "resolution"]
                if verb_count < 30
                else ["introduction", "development", "resolution"]
            )
        )

        if set(story_dialogue.keys()) != set(expected_parts):
            raise ValueError(
                f"Expected parts {expected_parts}, got {list(story_dialogue.keys())}"
            )

        return story_name, story_dialogue

    except Exception as e:
        raise RuntimeError(f"Failed to generate story: {e}")
