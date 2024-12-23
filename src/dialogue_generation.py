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
    load_json,
    save_json,
    update_vocab_usage,
)


load_dotenv()  # so we can use environment variables for various global settings


def get_story_prompt(verbs: list, vocab: list) -> str:
    prompt = f"""Create a three-part story told through dialogue between Alex and Sam. Each part should be about 2 minutes long when spoken (approximately 300 words or 20-25 dialogue exchanges).

Story Requirements:
1. The story must have a clear narrative arc across three parts:
   - Introduction (2 mins): Set up the situation and characters
   - Development (2 mins): Present a challenge or complication
   - Resolution (2 mins): Resolve the situation

2. Use vocabulary naturally from these lists:
   Verbs: {', '.join(verbs)}
   Other words: {', '.join(vocab)}
   
3. Guidelines:
   - Use at least 50% of the provided vocabulary across the story
   - Mix the vocabulary naturally - don't force all words
   - Keep dialogue exchanges realistic (about 10-15 words per utterance)
   - Alternate between Alex and Sam

4. Output Format:
{{
    "introduction": {{
        "dialogue": [
            {{"speaker": "Alex", "text": "..."}},
            {{"speaker": "Sam", "text": "..."}}
        ]
    }},
    "development": {{
        "dialogue": [
            {{"speaker": "Alex", "text": "..."}},
            {{"speaker": "Sam", "text": "..."}}
        ]
    }},
    "resolution": {{
        "dialogue": [
            {{"speaker": "Alex", "text": "..."}},
            {{"speaker": "Sam", "text": "..."}}
        ]
    }}
}}

Create a natural, engaging story that primarily uses the provided vocabulary but allows for essential connecting words. The story should feel cohesive across all three parts while maintaining natural dialogue flow."""

    return prompt


def generate_story(vocab_dict: Dict[str, List[str]]) -> Dict:
    """Extract dialogue from an LLM response.
    Returns a dictionary with story parts as keys and dialogue lists as values.
    """

    prompt = get_story_prompt(verbs=vocab_dict["verbs"], vocab=vocab_dict["vocab"])
    llm_response = anthropic_generate(prompt, max_tokens=4000)
    extracted_json = extract_json_from_llm_response(llm_response)

    if not extracted_json:
        print("No valid dialogue found in the response")
        return None

    # Verify the structure of the extracted JSON
    for part, content in extracted_json.items():
        if not isinstance(content, dict) or "dialogue" not in content:
            print(f"Invalid dialogue structure in part: {part}")
            return None

    return extracted_json


def get_least_used_words(category, count):
    # Load the current usage
    with open(config.VOCAB_USAGE_PATH, "r") as f:
        vocab_usage = json.load(f)

    # Calculate weights (inverse of usage count + 1 to avoid division by zero)
    words = list(vocab_usage[category].keys())
    weights = [1 / (usage + 1) for usage in vocab_usage[category].values()]

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Perform weighted random sampling
    selected_words = random.choices(words, weights=normalized_weights, k=count)

    # now we should update that they have been used once (to prevent instances of words remaining at 0)
    # for any mismatch in the lemmma or the vocab list
    if category == "verbs":
        pos = "VERB"
    else:
        pos = "vocab"

    words_with_pos = [(word, pos) for word in selected_words]
    update_vocab_usage(words_with_pos)

    return selected_words


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
