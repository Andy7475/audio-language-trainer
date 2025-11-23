"""Anthropic API authentication utilities."""

import os
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv


# Singleton Anthropic client
_anthropic_client: Optional[Anthropic] = None


def get_anthropic_client() -> Anthropic:
    """Get an Anthropic API client instance (singleton).

    Returns:
        Anthropic: Configured Anthropic client

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not found in environment
    """
    global _anthropic_client

    if _anthropic_client is not None:
        return _anthropic_client

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def reset_client() -> None:
    """Reset the cached Anthropic client instance (useful for testing)."""
    global _anthropic_client
    _anthropic_client = None
