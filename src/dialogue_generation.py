import json
import os
import random
import re
import subprocess
import sys
from typing import Dict, List, Set, Tuple

import pysnooper
import spacy
from dotenv import load_dotenv

from src.config_loader import config
from src.utils import (
    anthropic_generate,
    extract_json_from_llm_response,
)


load_dotenv()  # so we can use environment variables for various global settings


def get_story_prompt(verbs: list, vocab: list) -> str:
    """Creates a prompt for story generation, adapting length and complexity based on vocabulary size.
    Primarily uses verb count to determine story structure, as verbs drive the action.

    Args:
        verbs: List of verbs to incorporate
        vocab: List of other vocabulary words

    Returns:
        str: Prompt for story generation including output format specification
    """

    # Determine story structure based primarily on verb count
    verb_count = len(verbs)

    if verb_count < 10:
        parts = {"story": "A single focused scene showing natural interaction"}
        target_length = "30-45 seconds (about 75-100 words)"

    elif verb_count < 30:
        parts = {
            "setup": "Establish the situation and context",
            "resolution": "Complete the interaction",
        }
        target_length = "~1 minute per part (about 100 - 150 words each)"

    else:
        parts = {
            "introduction": "Set up the situation and characters",
            "development": "Present a challenge or complication",
            "resolution": "Resolve the situation",
        }
        target_length = "1-2 minutes per part (about 150-300 words each)"

    parts_description = "\n   - ".join(f"{k} ({v})" for k, v in parts.items())

    prompt = f"""Create a story named in the format {{story_name: "descriptive title here"}} followed by dialogue between Alex and Sam.
The story_name must be 3 words long and describe the main theme or setting, yet be engaging.

Story Requirements:
1. The story should have {len(parts)} part(s):
   - {parts_description}
   Each part should be about {target_length} when spoken.

2. Use vocabulary naturally from these lists ({verb_count} verbs and {len(vocab)} other words):
   Verbs: {', '.join(verbs)}
   Other words: {', '.join(vocab)}
   
3. Guidelines:
   - Using the above vocabulary, use whatever words make an engaging story (don't force a story to include all the words)
   - Only add essential connecting words (e.g., greetings, pronouns, or basic verbs (am, have, want, like etc) if necessary)
   - Keep dialogue exchanges realistic (about 10-15 words per utterance)
   - Alternate between Alex and Sam
   - Focus on quality over length - better a shorter, coherent story than a forced longer one
   - Focus on quality over using all the vocab - better a coherent story using less vocab

4. Output Format:
{{
    "story_name": "Engaging Title (3 words long)",
    {','.join(f'''
    "{part}": {{
        "dialogue": [
            {{"speaker": "Alex", "text": "..."}},
            {{"speaker": "Sam", "text": "..."}}
        ]
    }}''' for part in parts.keys())}
}}

Create a natural, engaging story that uses the provided vocabulary while maintaining good flow and coherence."""

    return prompt


def generate_story(vocab_dict: Dict[str, List[str]]) -> Dict:
    """Extract dialogue from an LLM response and validate structure.

    Args:
        vocab_dict: Dictionary with keys 'verbs' and 'vocab' containing word lists

    Returns:
        Dictionary with story parts as keys and dialogue lists as values, or None if invalid
    """

    prompt = get_story_prompt(verbs=vocab_dict["verbs"], vocab=vocab_dict["vocab"])
    llm_response = anthropic_generate(prompt, max_tokens=4000)
    story_data = extract_json_from_llm_response(llm_response)

    if not story_data:
        print("No valid JSON found in the response")
        return None

    story_name = story_data.get("story_name")
    if not story_name:
        print("Missing story_name in response")
        return None

    # Extract dialogue parts, excluding story_name
    story_dialogue = {k: v for k, v in story_data.items() if k != "story_name"}

    if not story_dialogue:
        print("No valid dialogue found in the response")
        return None

    # Verify the structure and speakers in each part
    valid_speakers = {"Sam", "Alex"}

    for part, content in story_dialogue.items():
        # Check basic structure
        if not isinstance(content, dict) or "dialogue" not in content:
            print(f"Invalid dialogue structure in part: {part}")
            return None

        # Check each dialogue entry has valid speakers
        for utterance in content["dialogue"]:
            if not isinstance(utterance, dict) or "speaker" not in utterance:
                print(f"Missing speaker in dialogue: {utterance}")
                return None

            if utterance["speaker"] not in valid_speakers:
                print(
                    f"Invalid speaker '{utterance['speaker']}' in dialogue. Must be 'Sam' or 'Alex'"
                )
                return None

    # Return the validated data with story_name included
    print(f"generated story: {story_name}")
    return story_name, story_dialogue


def add_usage_to_words(word_list: List[str], category: str) -> str:
    """adds the number of times that words has been used to the word_list - getting
    this data from the vocab_usage file - this can then be issued as string into the prompt
    """
    # Load the current usage
    with open(config.VOCAB_USAGE_PATH, "r") as f:
        vocab_usage = json.load(f)

    # make word_list all lower case
    word_list = [word.lower() for word in word_list]

    # Check if the category exists in vocab_usage
    if category not in vocab_usage:
        raise ValueError(f"Category '{category}' not found in vocabulary usage data.")

    # Add usage count to each word
    word_usage = [(word, vocab_usage[category].get(word, 0)) for word in word_list]

    # Sort by usage count (least used first)
    word_usage.sort(key=lambda x: x[1])

    # Format as a string
    formatted_string = (
        "{" + ", ".join(f"'{word}': {count}" for word, count in word_usage) + "}"
    )

    return formatted_string
