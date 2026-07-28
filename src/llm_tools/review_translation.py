"""LLM tool for reviewing and refining translations."""

from llm_tools.base import (
    load_prompt_template,
    get_anthropic_client,
    extract_tool_response,
    DEFAULT_MODEL,
)


# Tool definition
TOOL_SCHEMA = {
    "name": "review_translation",
    "description": "Review and improve a translation from English to target language",
    "input_schema": {
        "type": "object",
        "properties": {
            "translation": {"type": "string"},
            "modified": {"type": "boolean"},
        },
        "required": ["translation", "modified"],
    },
}

# Tool definition for batch review
BATCH_TOOL_SCHEMA = {
    "name": "review_translations_batch",
    "description": "Review and improve a batch of translations between a source and target language",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "translation": {"type": "string"},
                        "modified": {"type": "boolean"},
                    },
                    "required": ["translation", "modified"],
                },
                "description": "One entry per input pair, in the same order they were given",
            }
        },
        "required": ["results"],
    },
}


def refine_translation(
    english_phrase: str,
    initial_translation: str,
    target_language_name: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.2,
) -> str:
    """Refine a translation using Claude API.

    Args:
        english_phrase: The English phrase being translated
        initial_translation: The initial Google Translate translation
        target_language_name: Display name of target language (e.g., "French")
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for generation

    Returns:
        str: The refined translation text

    Raises:
        RuntimeError: If refinement fails
    """
    try:
        # Load prompts from template files
        system_template = load_prompt_template("review_translation", "system")
        user_template = load_prompt_template("review_translation", "user")

        # Substitute variables
        system_prompt = system_template.substitute(
            target_language_name=target_language_name
        )
        user_prompt = user_template.substitute(
            english_phrase=english_phrase,
            target_language_name=target_language_name,
            initial_translation=initial_translation,
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
                "name": "review_translation",
            },
        )

        # Extract tool response
        tool_input = extract_tool_response(response, "review_translation")
        if tool_input:
            return tool_input["translation"]

        # If we didn't get a tool response, return the original translation
        print("Warning: No tool response found, returning original translation")
        return initial_translation

    except Exception as e:
        raise RuntimeError(f"Failed to refine translation with Anthropic: {e}")


def refine_translations_batch(
    pairs: list[tuple[str, str]],
    target_language_name: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4000,
    temperature: float = 0.2,
) -> list[str]:
    """Refine a batch of translations in a single Claude API call.

    Use this instead of calling refine_translation() in a loop when translating many
    phrases at once - it trades N sequential API round trips for one.

    Args:
        pairs: List of (source_phrase, initial_translation) tuples
        target_language_name: Display name of the translation's language (e.g., "French")
        model: Anthropic model to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for generation

    Returns:
        List[str]: Refined translations, same length and order as `pairs`. Falls back to
            the original initial translations if the model's response doesn't line up
            one-to-one.

    Raises:
        RuntimeError: If refinement fails
    """
    if not pairs:
        return []

    try:
        # Load prompts from template files
        system_template = load_prompt_template("review_translation_batch", "system")
        user_template = load_prompt_template("review_translation_batch", "user")

        # Substitute variables
        system_prompt = system_template.substitute(
            target_language_name=target_language_name
        )
        pairs_str = "\n".join(
            f"{i}. Source: {source}\n   {target_language_name}: {initial}"
            for i, (source, initial) in enumerate(pairs, 1)
        )
        user_prompt = user_template.substitute(
            target_language_name=target_language_name, pairs=pairs_str
        )

        # Get Anthropic client and make API call
        client = get_anthropic_client()
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            tools=[BATCH_TOOL_SCHEMA],
            tool_choice={
                "type": "tool",
                "name": "review_translations_batch",
            },
        )

        # Extract tool response
        tool_input = extract_tool_response(response, "review_translations_batch")
        results = tool_input.get("results", []) if tool_input else []

        if len(results) != len(pairs):
            print(
                f"Warning: review_translations_batch returned {len(results)} results for "
                f"{len(pairs)} input pairs, returning initial translations unchanged"
            )
            return [initial for _, initial in pairs]

        return [result["translation"] for result in results]

    except Exception as e:
        raise RuntimeError(f"Failed to refine translation batch: {e}")
