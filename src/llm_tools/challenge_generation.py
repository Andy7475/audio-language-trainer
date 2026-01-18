"""Challenge generation LLM tool for creating language learning roleplay scenarios."""

from src.challenges.models import Challenge
from src.llm_tools.base import (
    DEFAULT_MODEL,
    get_anthropic_client,
)


def generate_challenge_content(system_prompt: str, user_prompt: str):
    client = get_anthropic_client()
    response = client.beta.messages.parse(
        model=DEFAULT_MODEL,
        betas=["structured-outputs-2025-11-13"],
        max_tokens=3000,
        temperature=0.4,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        output_format=Challenge,
    )

    challenge = response.parsed_output
    return challenge
