import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from dotenv import load_dotenv
from anthropic import Anthropic
from google.cloud import language_v1
from google.cloud import translate_v2 as translate

from src.config_loader import config


def review_translations_with_anthropic(
    phrase_pairs: List[Dict[str, str]],
    target_language: str = None,
    model: str = "claude-3-5-sonnet-latest",
) -> List[Dict[str, Any]]:
    """
    Use Anthropic API to review and improve translations using the tool interface.

    Args:
        phrase_pairs: List of dictionaries containing {'english': 'phrase', 'translation': 'current translation'}
        target_language: The target language for translations (defaults to config.TARGET_LANGUAGE_NAME)
        model: Anthropic model to use

    Returns:
        List of dictionaries with {'english': str, 'translation': str, 'modified': bool}
    """
    # Default to config target language if not specified
    if target_language is None:
        target_language = config.TARGET_LANGUAGE_NAME.lower()

    # Set up the Anthropic client
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    client = Anthropic(api_key=api_key)

    # Define the translation review tool
    tools = [
        {
            "name": "review_translations",
            "description": f"Review and improve translations from {config.SOURCE_LANGUAGE_NAME} to {target_language}",
            "input_schema": {
                "type": "object",
                "properties": {
                    "translations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "english": {"type": "string"},
                                "translation": {"type": "string"},
                                "modified": {"type": "boolean"},
                            },
                            "required": ["english", "translation", "modified"],
                        },
                    }
                },
                "required": ["translations"],
            },
        }
    ]

    # Construct the prompt
    system_prompt = f"""You are a professional translator specializing in natural-sounding {target_language}.
Review the provided {config.SOURCE_LANGUAGE_NAME} phrases and their {target_language} translations.
Improve translations to sound more natural for everyday spoken {target_language}.
You are supporting a language learner, so keep to the {config.SOURCE_LANGUAGE_NAME} vocabulary, but use a natural word choice or phrase that a native speaker would use.

For each translation pair:
1. Assess if the current translation sounds natural in {target_language}
2. If it doesn't sound natural for speech, provide an improved translation
3. Set 'modified' to true if you changed the translation, false if original was good

Only change translations that need improvement to sound more natural in speech - maintain the exact meaning."""

    # Convert phrase pairs to the format expected in the prompt
    formatted_pairs = "\n".join(
        [
            f"English: {pair['english']}\n{target_language}: {pair['translation']}\n"
            for pair in phrase_pairs
        ]
    )

    user_prompt = f"""Review these translation pairs and return your assessment using the review_translations tool:
<phrases>{formatted_pairs}</phrases>

For each translation, determine if it needs improvement and provide a more natural-sounding translation if needed."""

    # Make the API call
    try:
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=4000,
            temperature=0.2,
            tools=tools,
            tool_choice={
                "type": "tool",
                "name": "review_translations",
            },  # Explicitly require tool use
        )

        # Extract the tool use from the response
        for content in response.content:
            if content.type == "tool_use":
                if content.name == "review_translations":
                    return content.input["translations"]

        # If we didn't get a tool response, try to parse from text
        print("Warning: No tool response found, attempting to parse JSON from text")
        for content in response.content:
            if content.type == "text":
                # Try to extract JSON from text response
                text = content.text
                json_match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass

        print("Could not extract valid translation data from response")
        return []

    except Exception as e:
        print(f"API call error: {e}")
        return []


def batch_translate(
    texts,
    batch_size=128,
    target_language=None,
    source_language=None,
):
    """Translate texts in batches."""
    if target_language is None:
        target_language = config.TARGET_LANGUAGE_ALPHA2
    if source_language is None:
        source_language = config.SOURCE_LANGUAGE_ALPHA2

    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = translate.Client().translate(
            batch,
            target_language=target_language,
            source_language=source_language,
            format_="text",
        )
        translated_texts.extend([item["translatedText"] for item in result])
    return translated_texts


def translate_from_english(
    text: Union[List[str], str], target_language: str = None
) -> List[str]:
    """translates text into the target_language, returns translated text. text can be a str or List[str]"""
    if target_language is None:
        target_language = config.TARGET_LANGUAGE_ALPHA2

    result = translate.Client().translate(
        text, target_language=target_language, source_language=config.SOURCE_LANGUAGE_ALPHA2
    )

    if isinstance(result, list):
        result = [item["translatedText"] for item in result]
    else:
        result = [result["translatedText"]]
    return result


def translate_dialogue(dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """translates the 'text' part of the dialogue, keeping the speaker parts"""
    texts = [utterance["text"] for utterance in dialogue]
    translated_texts = batch_translate(texts)

    translated_dialogue = []
    for utterance, translated_text in zip(dialogue, translated_texts):
        translated_utterance = utterance.copy()
        translated_utterance["text"] = translated_text
        translated_dialogue.append(translated_utterance)

    return translated_dialogue


def translate_phrases(corrected_phrases: List[str]) -> List[Tuple[str, str]]:
    """translates a list of english phrases and returns a tuple of english, target_language
    phrases back as this is an easier format to pass into audio generation, and to manually inspect
    """
    translated_phrases = batch_translate(corrected_phrases)
    return list(zip(corrected_phrases, translated_phrases))


def tokenize_text(text: str, language_code: Optional[str] = None) -> List[str]:
    """
    Tokenize text using language-appropriate methods.

    For space-separated languages: Simply split on spaces
    For other languages: Use Google Cloud Natural Language API

    Args:
        text: Text to tokenize
        language_code: Two-letter language code (e.g. 'en', 'ja', defaults to current config language)

    Returns:
        List of tokens suitable for TTS breaks and Wiktionary lookups
    """
    # Evaluate language_code at runtime to pick up config changes
    if language_code is None:
        language_code = config.TARGET_LANGUAGE_CODE

    if not text:
        return []

    # For languages that use spaces
    if " " in text and not any(
        ord(char) > 0x4E00 for char in text
    ):  # Quick check for CJK
        return [token for token in text.split() if token]

    # For languages needing API
    try:
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT,
            language=language_code,
        )

        response = client.analyze_syntax(
            request={
                "document": document,
                "encoding_type": language_v1.EncodingType.UTF8,
            }
        )

        tokens = [token.text.content for token in response.tokens]
        return tokens if tokens else text.split()

    except Exception as e:
        print(f"API Tokenization failed: {str(e)}")
        # Fallback: split on spaces if present, otherwise return whole text as one token
        return text.split() if " " in text else [text]


def review_translated_phrases_batch(
    translated_phrases: Dict[str, Dict[str, str]],
    target_language: str = None,
    batch_size: int = 20,
    model: str = "claude-3-5-sonnet-latest",
) -> Dict[str, Dict[str, Any]]:
    """
    Wrapper function to batch review translations from the translated_phrases format.

    Args:
        translated_phrases: Dictionary with phrase_keys containing {'english': 'phrase', target_language: 'translation'}
        target_language: The target language for translations (defaults to config.TARGET_LANGUAGE_NAME)
        batch_size: Number of phrases to process in each batch (default: 20)
        model: Anthropic model to use

    Returns:
        Dictionary with same structure but with 'modified' field added to each translation
    """
    if target_language is None:
        target_language = config.TARGET_LANGUAGE_NAME.lower()

    # Convert to list of (phrase_key, phrase_data) tuples for batching
    phrase_items = list(translated_phrases.items())
    result = {}

    # Process in batches
    for i in range(0, len(phrase_items), batch_size):
        batch_items = phrase_items[i : i + batch_size]

        # Convert batch to phrase_pairs format
        phrase_pairs = []
        batch_keys = []

        for phrase_key, phrase_data in batch_items:
            phrase_pairs.append(
                {
                    "english": phrase_data["english"],
                    "translation": phrase_data[target_language],
                }
            )
            batch_keys.append(phrase_key)

        print(
            f"Processing batch {i//batch_size + 1} with {len(phrase_pairs)} phrases..."
        )

        # Call the original function
        reviewed_batch = review_translations_with_anthropic(
            phrase_pairs=phrase_pairs, target_language=target_language, model=model
        )

        # Convert results back to original format
        for j, reviewed_translation in enumerate(reviewed_batch):
            phrase_key = batch_keys[j]
            result[phrase_key] = {
                "english": reviewed_translation["english"],
                target_language: reviewed_translation["translation"],
                "modified": reviewed_translation["modified"],
            }

    return result


def review_story_dialogue_translations(
    story_dialogue: Dict[str, Dict[str, Any]],
    target_language: str = None,
    model: str = "claude-3-5-sonnet-latest",
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Review and improve translations for a story's dialogue using Anthropic API.

    Args:
        story_dialogue: Dictionary where each key is a story part containing dialogue and translated_dialogue
        target_language: The target language for translations (defaults to config.TARGET_LANGUAGE_NAME)
        model: Anthropic model to use
        verbose: If True, print out old and new translations when changes are made

    Returns:
        Dictionary with same structure but with improved translations
    """
    # Default to config target language if not specified
    if target_language is None:
        target_language = config.TARGET_LANGUAGE_NAME.lower()

    # Set up the Anthropic client
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    client = Anthropic(api_key=api_key)

    # Define the translation review tool
    tools = [
        {
            "name": "review_story_translations",
            "description": f"Review and improve translations from {config.SOURCE_LANGUAGE_NAME} to {target_language} for a story's dialogue",
            "input_schema": {
                "type": "object",
                "properties": {
                    "translations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "speaker": {"type": "string", "enum": ["Sam", "Alex"]},
                                "text": {"type": "string"},
                                "translation": {"type": "string"},
                                "modified": {"type": "boolean"},
                            },
                            "required": ["speaker", "text", "translation", "modified"],
                        },
                    }
                },
                "required": ["translations"],
            },
        }
    ]

    # Construct the prompt
    system_prompt = f"""You are a professional translator specializing in natural-sounding {target_language}.
Review the provided story dialogue and its {target_language} translations.
Improve translations to sound more natural for everyday spoken {target_language}.
You are supporting a language learner, so keep to the {config.SOURCE_LANGUAGE_NAME} vocabulary as much as possible, but use a natural word choice or phrase that a native speaker would use.

For each dialogue exchange:
1. Assess if the current translation sounds natural in {target_language}
2. If it doesn't sound natural for speech, provide an improved translation
3. Set 'modified' to true if you changed the translation, false if original was good

Only change translations that need improvement to sound more natural in speech - maintain the exact meaning.
Consider the context of the entire story when reviewing translations.
Note: Speakers must be either 'Sam' or 'Alex'."""

    # Flatten the dialogue for processing
    flattened_dialogue = []
    part_lengths = {}  # Keep track of how many utterances are in each part

    for part_name, part_data in story_dialogue.items():
        if "dialogue" in part_data and "translated_dialogue" in part_data:
            part_lengths[part_name] = len(part_data["dialogue"])
            for eng, trans in zip(
                part_data["dialogue"], part_data["translated_dialogue"]
            ):
                # Validate speaker
                if eng["speaker"] not in ["Sam", "Alex"]:
                    raise ValueError(
                        f"Invalid speaker: {eng['speaker']}. Must be either 'Sam' or 'Alex'"
                    )

                flattened_dialogue.append(
                    {
                        "speaker": eng["speaker"],
                        "text": eng["text"],
                        "translation": trans["text"],
                        "modified": False,
                    }
                )

    if not flattened_dialogue:
        print("No dialogue found to review")
        return story_dialogue

    user_prompt = f"""Review this story's dialogue translations and return your assessment using the review_story_translations tool:
<story>{json.dumps(flattened_dialogue, indent=2)}</story>

For each dialogue exchange, determine if it needs improvement and provide a more natural-sounding translation if needed.
Remember: Speakers must be either 'Sam' or 'Alex'."""

    # Make the API call
    try:
        response = client.messages.create(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=4000,
            temperature=0.2,
            tools=tools,
            tool_choice={
                "type": "tool",
                "name": "review_story_translations",
            },
        )

        # Extract the tool use from the response
        for content in response.content:
            if content.type == "tool_use":
                if content.name == "review_story_translations":
                    reviewed_translations = content.input["translations"]

                    # Validate speakers in response
                    for item in reviewed_translations:
                        if item["speaker"] not in ["Sam", "Alex"]:
                            raise ValueError(
                                f"Invalid speaker in response: {item['speaker']}. Must be either 'Sam' or 'Alex'"
                            )

                    # Reconstruct the story structure
                    result = {}  # Don't copy, build fresh
                    current_index = 0

                    for part_name, length in part_lengths.items():
                        if part_name in story_dialogue:
                            # Get the slice of reviewed translations for this part
                            part_reviewed = reviewed_translations[
                                current_index : current_index + length
                            ]

                            # Create new part data
                            result[part_name] = {
                                "dialogue": story_dialogue[part_name]["dialogue"],
                                "translated_dialogue": [
                                    {
                                        "speaker": item["speaker"],
                                        "text": item["translation"],
                                    }
                                    for item in part_reviewed
                                ],
                            }

                            # Print modified translations if verbose is True
                            if verbose:
                                for i, (old, reviewed) in enumerate(
                                    zip(
                                        story_dialogue[part_name][
                                            "translated_dialogue"
                                        ],
                                        part_reviewed,
                                    )
                                ):
                                    if (
                                        reviewed["modified"]
                                        and old["text"] != reviewed["translation"]
                                    ):
                                        print(
                                            f"English: {story_dialogue[part_name]['dialogue'][i]['text']}"
                                        )
                                        print(f"Old: {old['text']}")
                                        print(f"New: {reviewed['translation']}")
                                        print()

                        current_index += length

                    return result

        print("Could not extract valid translation data from response")
        return story_dialogue

    except Exception as e:
        print(f"API call error: {e}")
        return story_dialogue
