"""Challenge generation LLM tool for creating language learning roleplay scenarios."""

from typing import Dict, List, Literal

from src.llm_tools.base import (
    DEFAULT_MODEL,
    get_anthropic_client,
    extract_tool_response,
    load_prompt_template,
)
from src.models import BCP47Language

from pydantic import BaseModel, Field


class QandA(BaseModel):
    question: str
    answer: str


class Scenario(BaseModel):
    role_user: str = Field(
        ..., description="Role for user to play (e.g., 'coffee shop staff')"
    )
    role_agent: str = Field(
        ..., description="Role for AI agent to play (e.g., 'coffee shop staff')"
    )
    situation: str = Field(
        ..., description="Setting description (e.g., 'A coffee shop')"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty level (e.g., 'easy', 'medium', 'hard')"
    )
    task: str = Field(..., description="Main task to complete (e.g., 'Order a coffee')")
    find_out: List[QandA] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Specific information to discover (e.g., 'What is the price?')",
    )


class Challenge(BaseModel):
    story_title_hash: str = Field(..., description="Hash of the parent story")
    scenarios: List[Scenario] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="List of 3 roleplay scenarios, one at each difficult",
    )


# TOOL_SCHEMA = transform_schema(Challenge)
# Fixed tool schema - always 5 scenarios with 3 complications each


def test_tool_use():
    client = get_anthropic_client()
    response = client.beta.messages.parse(
        model=DEFAULT_MODEL,
        betas=["structured-outputs-2025-11-13"],
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Please genereate a challenge for coffee shop situations, just to test the tool use in the SDK",
            }
        ],
        output_format=Challenge,
    )

    challenge = response.parsed_output
    return challenge


def generate_challenges(
    story_dialogue: Dict,
    target_language: BCP47Language,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 3000,
    temperature: float = 0.4,
) -> Dict:
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
            # tools=[TOOL_SCHEMA],
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
            raise ValueError(f"Expected exactly 5 scenarios, got {len(scenarios)}")

        # Validate each scenario has 3 complications
        for i, scenario in enumerate(scenarios):
            complications = scenario.get("complications", [])
            if len(complications) != 3:
                raise ValueError(
                    f"Scenario {i + 1} has {len(complications)} complications, expected 3"
                )

        return tool_input

    except Exception as e:
        raise RuntimeError(f"Failed to generate challenges: {e}")
