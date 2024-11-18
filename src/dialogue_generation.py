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

from src.nlp import extract_vocab_and_pos

load_dotenv()  # so we can use environment variables for various global settings


def generate_story_plan(
    story_guide: str,
    verb_list: List[str],
    vocab_list: List[str],
    output_dir: str,
    story_name: str,
    test=False,
) -> dict:
    """Generate an outline story following a standard story structure.

    Args:
        story_guide: Brief description of the story
        verb_list: List of verbs to incorporate
        vocab_list: List of vocabulary words to incorporate
        output_dir: Directory to save the story plan
        story_name: Clean story name (already processed for file naming)
        test: If True, returns test data instead of generating

    Returns:
        Dictionary containing the story plan

    Raises:
        Exception: If story plan generation fails
    """
    story_structure = [
        "exposition: Introduce the main characters (Alex and Sam) and setting",
        "rising_action: Present a challenge or conflict",
        "climax: The turning point where the conflict reaches its peak",
        "falling_action: Events following the climax",
        "resolution: The conclusion of the story",
    ]

    prompt = (
        f"""Create a brief story plan (story guide: {story_guide}) for a language learning dialogue series."""
        """The story should be simple enough for beginners to follow but interesting enough to maintain engagement."""
        """Use the following structure:\n"""
        f"""{' '.join(story_structure)}\n"""
        """Keep each part of the story plan to 1-2 sentences. The entire plan should be no more than 200 words.\n"""
        f"""You should consider the following verbs and vocab list to write the plan: verbs: ({", ".join(verb_list)}), vocab:({", ".join(vocab_list)}).\n"""
        """Output the story plan as a JSON object with keys for each part of the story structure."""
    )

    if test:
        llm_response = """{
            "exposition": "Two friends, Alex and Sam, decide to learn a new language together.",
            "rising_action": "They face challenges in their studies and personal lives that test their commitment.",
            "climax": "A language competition is announced, pushing them to their limits.",
            "falling_action": "They prepare for the competition, supporting each other through difficulties.",
            "resolution": "They participate in the competition, growing closer as friends and more confident in their language skills."
        }"""
    else:
        llm_response = anthropic_generate(prompt)

    # Extract and validate the JSON part from the response
    story_plan = extract_json_from_llm_response(llm_response)
    if not story_plan:
        raise Exception("Failed to extract valid JSON from LLM response")

    # Normalize keys to lowercase
    story_plan = {k.lower(): v for k, v in story_plan.items()}

    # Validate required story parts
    required_parts = {
        "exposition",
        "rising_action",
        "climax",
        "falling_action",
        "resolution",
    }
    if not all(part in story_plan for part in required_parts):
        raise Exception(
            f"Story plan missing required parts. Found: {list(story_plan.keys())}"
        )

    # Save the story plan
    os.makedirs(output_dir, exist_ok=True)
    story_plan_path = os.path.join(output_dir, f"story_plan_{story_name}.json")

    try:
        with open(story_plan_path, "w") as f:
            json.dump(story_plan, f, indent=2)
        print(f"Story plan saved to: {story_plan_path}")
    except Exception as e:
        print(f"Warning: Failed to save story plan: {e}")

    return story_plan


def generate_recap(dialogue, test=True):
    prompt = f"""
    Based on the following dialogue, provide a brief recap of the events and character interactions. 
    Keep the recap to 2-3 sentences, focusing on the main points that are relevant for continuing the story.
    
    Output should JSON of the form {{"recap" : "your recap here"}}

    Dialogue:
    {json.dumps(dialogue, indent=2)}
    """

    # Here, you would send this prompt to your LLM and get the response
    if test:
        llm_response = """{"recap" : "Alex and Sam met to study together. They discussed their progress and challenges in learning the new language. They also made plans to attend a language exchange event next week."}"""
    else:
        llm_response = anthropic_generate(prompt)

    recap = extract_json_from_llm_response(llm_response)["recap"]

    return recap


def generate_dialogue_prompt(
    story_part: str,
    story_part_outline: str,
    last_recap: str,
    verb_usage_str: str,
    vocab_usage_str: str,
    verb_use_count=3,
    vocab_use_count=10,
    grammar_concept_count=10,
    grammar_use_count=3,
):
    # Load the JSON files
    grammar_concepts = load_json(config.GRAMMAR_USAGE_PATH)

    # Select grammar concepts
    selected_concepts = select_grammar_concepts(grammar_concepts, grammar_concept_count)
    used_concepts = selected_concepts[:grammar_use_count]

    # Update grammar concept usage
    update_grammar_concept_usage(grammar_concepts, used_concepts)

    # Generate the prompt
    prompt = f"""Create a brief dialogue for language learners using the following guidelines:
    1. This is for language practice. Prioritze words from the lists provided below, but add additional words if required for the story. Preferentially use words that haven't been used as much (with a smaller number)
    2. Pick from about {verb_use_count} of these verbs:
    {verb_usage_str}
    You can use other verbs if required to make the tenses work (e.g., auxiliary verbs).
    3. Use at least {vocab_use_count} words from this vocabulary list, again picking words with smaller numbers by preference:
    {vocab_usage_str}
    4. Focus on these grammatical concepts (use each at least once, but no more than twice):
    {', '.join(used_concepts)}
    5. Create a conversation with 6-8 lines of dialogue total (about 20-30 seconds when spoken).
    6. The characters are Alex and Sam. Maintain their personalities and relationship from previous dialogues.
    7. Recap from last episode: {last_recap}. You can (if necessary) reuse vocab words from the recap to maintain continuity, but prioritise the vocab list provided.
    8. Output the dialogue in JSON format with Sam and Alex clearly labeled as 'speaker'. For example:
    {{
        "dialogue": [
        {{"speaker": "Alex", "text": "Hello! How are you today?"}},
        {{"speaker": "Sam", "text": "I'm doing well, thanks for asking."}}
        ]
    }}
    9. Do not give speaking parts to other characters. Just use Alex and Sam.
    10. Current story phase and plan: {story_part}: {story_part_outline}
    11. Create dialogue for the current story plan.
    Remember:
    - While you should prioritse words from the provided lists, you can use additional words if required for the story.
    - Balance practicing the specified concepts and vocabulary (about 45%) with advancing the storyline (about 55%).
    - Keep the language simple and appropriate for beginner language learners.
    """
    return prompt


def select_grammar_concepts(
    grammar_concepts: Dict, count: int
) -> List[Tuple[str, str]]:
    usable_concepts = [
        (category, concept)
        for category, concepts in grammar_concepts.items()
        for concept, details in concepts.items()
        if details["use"]
    ]

    # Sort concepts by usage (least used first)
    usable_concepts.sort(key=lambda x: grammar_concepts[x[0]][x[1]]["times_seen"])

    # Format the concepts as "category - concept"
    formatted_concepts = [
        f"{category} - {concept}" for category, concept in usable_concepts[:count]
    ]

    return formatted_concepts


def update_grammar_concept_usage(
    grammar_concepts: Dict, used_concepts: List[str]
) -> None:
    for used_concept in used_concepts:
        category, concept = used_concept.split(" - ")
        if category in grammar_concepts and concept in grammar_concepts[category]:
            grammar_concepts[category][concept]["times_seen"] += 1

    with open(config.GRAMMAR_USAGE_PATH, "w") as f:
        json.dump(grammar_concepts, f, indent=2)


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


def generate_complete_dialogue_prompt(
    story_plan: Dict[str, str],
    verbs_for_story_usage: str,
    vocab_for_story_usage: str,
    verb_use_count: int = 3,
    vocab_use_count: int = 10,
    grammar_concept_count: int = 10,
    grammar_use_count: int = 3,
) -> str:
    """Generate a single prompt for creating dialogue for all story parts."""
    # Load and select grammar concepts
    grammar_concepts = load_json(config.GRAMMAR_USAGE_PATH)
    selected_concepts = select_grammar_concepts(grammar_concepts, grammar_concept_count)
    used_concepts = selected_concepts[:grammar_use_count]
    update_grammar_concept_usage(grammar_concepts, used_concepts)

    # Create story plan string
    story_plan_str = "\n".join(
        [f"{part}: {outline}" for part, outline in story_plan.items()]
    )

    # Create the example JSON structure with double curly braces
    example_json = """
    {
        "exposition": {
            "dialogue": [
                {"speaker": "Alex", "text": "Hello! How are you today?"},
                {"speaker": "Sam", "text": "I'm doing well, thanks for asking."}
            ]
        },
        "rising_action": {
            "dialogue": [
                ...
            ]
        },
        ...
    }
    """

    prompt = f"""Create a complete multi-part dialogue for language learners following these guidelines:

    1. Create separate dialogues for each part of this story plan:
    {story_plan_str}

    2. For each story part, create dialogue using:
    - About {verb_use_count} verbs from: {verbs_for_story_usage}
    - At least {vocab_use_count} words from: {vocab_for_story_usage}
    Prioritize words with lower usage numbers.

    3. Each part should incorporate these grammatical concepts:
    {', '.join(used_concepts)}

    4. For each story part:
    - Create 6-8 lines of dialogue (20-30 seconds spoken)
    - Use only Alex and Sam as speakers
    - Keep language simple and appropriate for beginners
    - Ensure natural flow between story parts

    5. Output format should be a JSON object with story parts as keys:
    {example_json}

    Remember:
    - Each part should advance the story while practicing vocabulary
    - Balance concept practice (45%) with story progression (55%)
    - Maintain consistent personalities for Alex and Sam
    - Keep the story cohesive across all parts
    - You can use additional words if needed, but prioritize the provided lists

    Generate the complete dialogue for all parts in a single JSON response."""

    return prompt


def generate_dialogue(dialogue_prompt: str) -> Dict:
    """Extract dialogue from an LLM response.
    Returns a dictionary with story parts as keys and dialogue lists as values.
    """
    llm_response = anthropic_generate(dialogue_prompt, max_tokens=4000)
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


def get_vocab_from_dialogue(dialogue: List[Dict[str, str]]) -> Set[Tuple[str, str]]:
    """
    For a given English dialogue, extracts the vocab used as the lemmas.
    Returns a set of tuples of the form (lemma, POS) e.g. ("go", "VERB")
    Excludes punctuation, persons identified by spaCy, and the names 'sam' and 'alex'.
    """

    english_phrases = []
    for utterance in dialogue:
        english_phrases.append(utterance["text"])

    return extract_vocab_and_pos(english_phrases)
