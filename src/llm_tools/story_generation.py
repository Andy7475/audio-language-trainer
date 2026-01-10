"""Story generation LLM tool for creating English dialogue-based stories."""

import random
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from src.phrases.phrase_model import Phrase
from src.llm_tools.base import (
    DEFAULT_MODEL,
    get_anthropic_client,
    extract_tool_response,
    load_prompt_template,
)


def _build_story_schema() -> Dict[str, Any]:
    """Build dynamic tool schema.

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
    properties = {
        "story_name": {
            "type": "string",
            "description": "Engaging title for the story",
        },
        "summary": {
            "type": "string",
            "description": "Brief 1 - 2 sentence summary of the story"
        },
        "part_1": part_schema(),
        "part_2": part_schema(),
        "part_3": part_schema(),
    }
    required = ["story_name", "part_1", "part_2", "part_3"]

    return {
        "name": "generate_story",
        "description": "Generate an English dialogue-based story for language learning",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def _get_structure_description() -> Tuple[str, int, str]:
    """Get story structure details based on verb count.

    Args:
        verb_count: Number of verbs

    Returns:
        Tuple of (description, part_count, target_length)
    """

    description = """Three parts:
   - introduction (part 1): Set up the situation and characters
   - development (part 2): Present a challenge or complication
   - resolution (part 3): Resolve the situation"""
    part_count = 3
    target_length = "1-2 minutes per part (about 150-300 words each)"

    return description, part_count, target_length


def generate_story(
    phrase_list: List[Phrase],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4000,
    temperature: float = 0.4,
    num_phrases:int = 5
) -> Tuple[str, Dict]:
    """Generate an English dialogue-based story for language learning.

    Stories are ALWAYS generated in English. Translation happens separately.

    Args:
        phrase_list: List of phrases to include in the story
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Generation temperature

    Returns:
        Tuple of (story_name, story_dialogue)
        where story_dialogue has keys for each story part containing dialogue

    Raises:
        RuntimeError: If story generation fails
        ValueError: If response doesn't match expected structure

    """
    try:

        # Build dynamic schema based on verb count
        story_schema = _build_story_schema()

        # Get structure description for prompt
        structure_desc, part_count, target_length = _get_structure_description(
            
        )

        verbs = list({verb for phrase in phrase_list for verb in phrase.verbs})
        vocab = list({word for phrase in phrase_list for word in phrase.vocab})
        # Load prompt templates
        system_template = load_prompt_template("story_generation", "system")
        user_template = load_prompt_template("story_generation", "user")

        phrase_list_text =list(map(str, phrase_list))
        phrases_to_use = ", ".join(random.choices(phrase_list_text, k=num_phrases))
        random.shuffle(verbs)
        random.shuffle(vocab)
        # Substitute variables
        system_prompt = system_template.substitute()
        user_prompt = user_template.substitute(
            phrases_to_use=phrases_to_use,
            structure_description=structure_desc,
            part_count=str(part_count),
            target_length=target_length,
            verbs=", ".join(verbs),
            vocab=", ".join(vocab),
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

        summary = story_dialogue.pop("summary")
        return story_name, summary, story_dialogue

    except Exception as e:
        raise RuntimeError(f"Failed to generate story: {e}")
