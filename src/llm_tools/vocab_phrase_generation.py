"""LLM tool for generating vocab-based phrases (no verbs) for language learners."""

from typing import Any

from src.base import (
    load_prompt_template,
    get_anthropic_client,
    extract_tool_response,
    DEFAULT_MODEL,
)


# Tool definition for vocab phrase generation
TOOL_SCHEMA = {
    "name": "generate_vocab_phrases",
    "description": "Generate descriptive English phrases for multiple vocabulary words without verbs",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target_word": {"type": "string"},
                        "phrase": {"type": "string"},
                        "additional_words": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["target_word", "phrase", "additional_words"],
                },
                "description": "List of phrase results, one per target word",
            }
        },
        "required": ["results"],
    },
}


def generate_vocab_phrases(
    target_words: list[str],
    context_words: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Generate descriptive phrases for multiple vocabulary words (no verbs).

    Generates one phrase per target word using descriptive language, adjectives,
    and positional words. Also tracks additional words used for vocabulary list management.

    Args:
        target_words: List of vocabulary words to create phrases around (e.g., ["apple", "table", "red"])
        context_words: Optional list of nearby vocabulary words that can be used in phrases
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for generation

    Returns:
        Dict containing:
            {
                "results": [
                    {
                        "target_word": str,
                        "phrase": str,
                        "additional_words": [str]
                    },
                    ...
                ],
                "all_additional_words": [str]  # Flattened list of all additional words
            }

    Raises:
        RuntimeError: If phrase generation fails
    """
    try:
        # Format target words for the prompt
        target_words_str = "\n".join([f"- {word}" for word in target_words])

        # Format context words for the prompt
        if context_words:
            context_str = ", ".join(context_words[:25])  # Limit to next 25 words
        else:
            context_str = "(No context words provided)"

        # Load prompts from template files
        system_template = load_prompt_template("vocab_phrase_generation", "system")
        user_template = load_prompt_template("vocab_phrase_generation", "user")

        # Substitute variables
        system_prompt = system_template.substitute()
        user_prompt = user_template.substitute(
            target_words=target_words_str, context_words=context_str
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
                "name": "generate_vocab_phrases",
            },
        )

        # Extract tool response
        tool_input = extract_tool_response(response, "generate_vocab_phrases")
        if not tool_input:
            raise RuntimeError("No tool response found from Anthropic API")

        # Flatten additional words
        all_additional_words = []
        for result in tool_input.get("results", []):
            all_additional_words.extend(result.get("additional_words", []))

        return {
            "results": tool_input.get("results", []),
            "all_additional_words": list(
                set(all_additional_words)
            ),  # Remove duplicates
        }

    except Exception as e:
        raise RuntimeError(f"Failed to generate phrases for words {target_words}: {e}")
