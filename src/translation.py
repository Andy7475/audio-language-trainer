import json
import os
import re
from typing import Any, Dict, List, Tuple, Union
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


def batch_translate(texts, batch_size=128):
    """Translate texts in batches."""
    translated_texts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = translate.Client().translate(
            batch,
            target_language=config.TARGET_LANGUAGE_ALPHA2,
            source_language="en",
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
        text, target_language=target_language, source_language="en"
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


def tokenize_text(
    text: str, language_code: str = config.TARGET_LANGUAGE_CODE
) -> List[str]:
    """
    Tokenize text using language-appropriate methods.

    For space-separated languages: Simply split on spaces
    For other languages: Use Google Cloud Natural Language API

    Args:
        text: Text to tokenize
        language_code: Two-letter language code (e.g. 'en', 'ja')

    Returns:
        List of tokens suitable for TTS breaks and Wiktionary lookups
    """
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
