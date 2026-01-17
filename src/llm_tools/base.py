"""Base utilities for LLM tools and prompt management."""

import os
from string import Template
from typing import Dict, Optional

from src.connections.anthropic_auth import get_anthropic_client as _get_anthropic_client


# Default model for all LLM tools
# Using 'claude-sonnet-4' which is Anthropic's recommended Sonnet model
DEFAULT_MODEL = "claude-sonnet-4-5"


def load_prompt_template(tool_name: str, prompt_type: str = "system") -> Template:
    """Load a prompt template from the prompts directory.

    Args:
        tool_name: Name of the tool (e.g., "review_translation")
        prompt_type: Type of prompt ("system" or "user")

    Returns:
        Template: A string.Template object ready for substitution

    Raises:
        FileNotFoundError: If prompt file doesn't exist

    Example:
        >>> template = load_prompt_template("review_translation", "system")
        >>> prompt = template.substitute(target_language_name="French")
    """
    prompts_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
    prompt_file = os.path.join(prompts_dir, tool_name, f"{prompt_type}.txt")

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_file}\n"
            f"Expected location: prompts/{tool_name}/{prompt_type}.txt"
        )

    with open(prompt_file, "r", encoding="utf-8") as f:
        return Template(f.read())


def get_anthropic_client():
    """Get an Anthropic API client.

    Returns:
        Anthropic: Configured Anthropic client

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not found in environment
    """
    return _get_anthropic_client()


def extract_tool_response(response, tool_name: str) -> Optional[Dict]:
    """Extract tool use response from Anthropic API response.

    Args:
        response: Anthropic API response object
        tool_name: Name of the tool to extract

    Returns:
        Optional[Dict]: Tool input dictionary if found, None otherwise
    """
    for content in response.content:
        if content.type == "tool_use" and content.name == tool_name:
            return content.input

    return None
