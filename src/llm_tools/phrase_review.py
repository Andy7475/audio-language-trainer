"""LLM tool for reviewing generated language-learner phrases for correctness."""

from langcodes import Language

from llm_tools.base import (
    load_prompt_template,
    get_anthropic_client,
    extract_tool_response,
    PREMIUM_MODEL,
)
from models import get_language


# Tool definition for phrase review
TOOL_SCHEMA = {
    "name": "review_phrases",
    "description": "Review a batch of language-learning phrases for grammatical correctness and register, correcting any that are wrong, slang, or unsuitable",
    "input_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "original_phrase": {"type": "string"},
                        "reviewed_phrase": {"type": "string"},
                        "modified": {"type": "boolean"},
                    },
                    "required": ["original_phrase", "reviewed_phrase", "modified"],
                },
                "description": "One entry per input phrase, in the same order they were given",
            }
        },
        "required": ["results"],
    },
}


def review_phrases(
    phrases: list[str],
    language: Language | str | None = None,
    model: str = PREMIUM_MODEL,
    max_tokens: int = 4000,
) -> list[str]:
    """Review a batch of generated phrases for grammatical correctness and register.

    Intended for phrases generated in a non-English target language, since the source
    vocabulary is drawn from real articles and subtitles and can carry slang or narrowly
    idiomatic phrasing that isn't suitable to teach directly.

    Args:
        phrases: List of phrases (in the target language) to review
        language: Target language of the phrases (default: en-GB)
        model: Anthropic model to use (default: PREMIUM_MODEL for higher-quality review)
        max_tokens: Maximum tokens for response

    Returns:
        List[str]: Reviewed phrases, same length and order as input. Falls back to the
            original phrases if the model's response doesn't line up one-to-one.

    Raises:
        RuntimeError: If review fails
    """
    if not phrases:
        return []

    try:
        lang = get_language(language)
        language_name = (
            Language.get(lang.language or "en").display_name() if lang else "English"
        )

        # Load prompts from template files
        system_template = load_prompt_template("phrase_review", "system")
        user_template = load_prompt_template("phrase_review", "user")

        # Substitute variables
        system_prompt = system_template.substitute(language_name=language_name)
        phrases_str = "\n".join(f"{i}. {phrase}" for i, phrase in enumerate(phrases, 1))
        user_prompt = user_template.substitute(
            language_name=language_name, phrases=phrases_str
        )

        # Get Anthropic client and make API call
        client = get_anthropic_client()
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            tools=[TOOL_SCHEMA],
            tool_choice={
                "type": "tool",
                "name": "review_phrases",
            },
        )

        # Extract tool response
        tool_input = extract_tool_response(response, "review_phrases")
        results = tool_input.get("results", []) if tool_input else []

        if len(results) != len(phrases):
            print(
                f"Warning: review_phrases returned {len(results)} results for "
                f"{len(phrases)} input phrases, returning originals unchanged"
            )
            return phrases

        return [result["reviewed_phrase"] for result in results]

    except Exception as e:
        raise RuntimeError(f"Failed to review phrases: {e}")
