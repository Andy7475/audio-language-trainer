import json
import os
import random
import re
import subprocess
import sys
from typing import Dict, List, Set, Tuple

import pysnooper
import spacy
from anthropic import AnthropicVertex
from dotenv import load_dotenv

from src.config_loader import config
from src.utils import anthropic_generate, extract_json_from_llm_response

load_dotenv()  # so we can use environment variables for various global settings

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")

GRAMMAR_USAGE_PATH = "../data/grammar_concepts_usage.json"
VOCAB_USAGE_PATH = "../data/vocab_usage.json"
STORY_PLAN_PATH = "../data/story_plan.json"
RECAP_PATH = "../data/story_recap.json"


def generate_story_plan(story_guide: str = None, test=True) -> dict:
    story_structure = [
        "Exposition: Introduce the main characters and setting",
        "Rising_Action: Present a challenge or conflict",
        "Climax: The turning point where the conflict reaches its peak",
        "Falling_Action: Events following the climax",
        "Resolution: The conclusion of the story",
    ]

    prompt = f"""
    Create a brief story plan (using {story_guide} as a guide) for a language learning dialogue series. The story should be simple enough for beginners to follow but interesting enough to maintain engagement. Use the following structure:

    {' '.join(story_structure)}

    Keep each part of the story plan to 1-2 sentences. The entire plan should be no more than 200 words. You should keep it relatively vague as you do not yet know what vocabulary will be available to use.
    Output the story plan as a JSON object with keys for each part of the story structure.

    """

    # Here, you would send this prompt to your LLM and get the response
    # For this example, I'll provide a placeholder response
    if test:
        llm_response = """{
            "Exposition": "Two friends, Alex and Sam, decide to learn a new language together.",
            "Rising_Action": "They face challenges in their studies and personal lives that test their commitment.",
            "Climax": "A language competition is announced, pushing them to their limits.",
            "Falling_Action": "They prepare for the competition, supporting each other through difficulties.",
            "Resolution": "They participate in the competition, growing closer as friends and more confident in their language skills."
        }"""
    else:
        llm_response = anthropic_generate(prompt)
    # Extract the JSON part from the response
    json_match = re.search(r"\{[\s\S]*\}", llm_response)
    if json_match:
        json_str = json_match.group(0)
        try:
            story_plan = json.loads(json_str)
        except json.JSONDecodeError:
            print("Error: Unable to parse JSON from LLM response")
            story_plan = {}
    else:
        print("Error: No JSON found in LLM response")
        story_plan = {}

    # Normalize keys to lowercase
    story_plan = {k.lower(): v for k, v in story_plan.items()}

    with open(STORY_PLAN_PATH, "w") as f:
        json.dump(story_plan, f, indent=2)

    return story_plan


def generate_recap(dialogue, test=True):
    prompt = f"""
    Based on the following dialogue, provide a brief recap of the events and character interactions. 
    Keep the recap to 2-3 sentences, focusing on the main points that are relevant for continuing the story.
    
    Dialogue:
    {json.dumps(dialogue, indent=2)}
    """

    # Here, you would send this prompt to your LLM and get the response
    if test:
        llm_response = "Alex and Sam met to study together. They discussed their progress and challenges in learning the new language. They also made plans to attend a language exchange event next week."
    else:
        llm_response = anthropic_generate(prompt)

    # Load existing recaps
    if os.path.exists(RECAP_PATH):
        with open(RECAP_PATH, "r") as f:
            recaps = json.load(f)
    else:
        recaps = {"recaps": []}

    # Append new recap
    recaps["recaps"].append(llm_response)

    # Save updated recaps
    with open(RECAP_PATH, "w") as f:
        json.dump(recaps, f, indent=2)

    return llm_response


def get_story_recap(num_recaps=None):
    """
    Retrieve the specified number of most recent recaps.
    If num_recaps is None, return all recaps.

    :param num_recaps: Number of most recent recaps to retrieve (optional)
    :return: List of recaps
    """
    if os.path.exists(RECAP_PATH):
        with open(RECAP_PATH, "r") as f:
            recaps = json.load(f)["recaps"]
        if num_recaps is not None:
            return recaps[-num_recaps:]
        return recaps
    else:
        return []


def get_story_plan():
    try:
        with open(STORY_PLAN_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return generate_story_plan()


def get_current_story_part(recaps: List[str], story_plan: dict) -> str:
    """
    Determine which part of the story we're currently in based on the number of recaps.

    :param recaps: List of existing recaps
    :param story_plan: Dictionary containing the story plan
    :return: Current part of the story
    """
    DIALOGUES_PER_STORY_PART = 1  # UNIT TEST will fail if this is changed
    story_parts = list(story_plan.keys())
    current_part_index = min(
        len(recaps) // DIALOGUES_PER_STORY_PART, len(story_parts) - 1
    )
    return story_parts[current_part_index]


def get_last_recap(recaps: List[str]) -> str:
    """
    Get the last recap or a default message if no recaps exist.

    :param recaps: List of existing recaps
    :return: Last recap or default message
    """
    return recaps[-1] if recaps else "This is the beginning of the story."


def load_story_plan():
    """
    Load the story plan from the JSON file.

    :return: Dictionary containing the story plan
    """
    try:
        with open(STORY_PLAN_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}  # Return an empty dict if file not found


def load_recaps():
    """
    Load the recaps from the JSON file.

    :return: List of recaps
    """
    try:
        with open(RECAP_PATH, "r") as f:
            return json.load(f)["recaps"]
    except FileNotFoundError:
        return []


@pysnooper.snoop(output="pysnoop.txt")
def generate_dialogue_prompt(
    verb_count=10,
    verb_use_count=3,
    vocab_count=100,
    vocab_use_count=10,
    grammar_concept_count=10,
):
    # Load the JSON files
    with open(GRAMMAR_USAGE_PATH, "r") as f:
        grammar_concepts = json.load(f)

    # Select verbs
    verbs = get_least_used_words("verbs", verb_count)
    verbs_str = ", ".join(verbs)

    # Select vocabulary
    vocab = get_least_used_words("vocab", vocab_count)
    vocab_str = ", ".join(vocab)

    # Select grammar concepts
    usable_concepts = [
        concept
        for category in grammar_concepts.values()
        for concept, details in category.items()
        if details["use"]
    ]
    selected_concepts = random.sample(
        usable_concepts, min(grammar_concept_count, len(usable_concepts))
    )

    # Sort concepts by usage
    selected_concepts.sort(
        key=lambda x: sum(
            details["times_seen"]
            for category in grammar_concepts.values()
            for concept, details in category.items()
            if concept == x
        )
    )

    # Get story plan and recap
    story_plan = get_story_plan()
    recaps = get_story_recap()

    # Determine which part of the story we're currently in
    current_part = get_current_story_part(recaps, story_plan)

    # Get the last recap
    last_recap = get_last_recap(recaps)

    # Generate the prompt
    prompt = f"""
    Create a brief dialogue for language learners using the following guidelines:
    1. This is for language practice. Only use words from the lists provided below.
    2. Pick from about {verb_use_count} of these verbs:
    {verbs_str}
    You can use other verbs if required to make the tenses work (e.g., auxiliary verbs).
    3. Use at least {vocab_use_count} words from this vocabulary list:
    {vocab_str}
    4. Focus on these grammatical concepts (use each at least once, but no more than twice):
    {', '.join(selected_concepts[:3])}
    5. Create a conversation with 6-8 lines of dialogue total (about 20-30 seconds when spoken).
    6. The characters are Alex and Sam. Maintain their personalities and relationship from previous dialogues.
    7. Output the dialogue in JSON format with speakers clearly labeled. For example:
    {{
        "dialogue": [
        {{"speaker": "Alex", "text": "Hello! How are you today?"}},
        {{"speaker": "Sam", "text": "I'm doing well, thanks for asking."}}
        ]
    }}
    8. Current story phase and plan: {current_part}: {story_plan[current_part]}
    9. Recap from last episode: {last_recap}
    10. Create dialogue for the current story plan.
    Remember:
    - Only use words from the provided lists.
    - Balance practicing the specified concepts and vocabulary (about 60%) with advancing the storyline (about 40%).
    - Keep the language simple and appropriate for beginners.
    """
    return prompt, verbs + vocab


def update_vocab_usage(used_words):
    # Load the current usage
    with open(VOCAB_USAGE_PATH, "r") as f:
        vocab_usage = json.load(f)

    # Update the usage count for each used word
    for word in used_words:
        if word in vocab_usage["verbs"]:
            vocab_usage["verbs"][word] += 1
        elif word in vocab_usage["vocab"]:
            vocab_usage["vocab"][word] += 1

    # Save the updated usage dictionary
    with open(VOCAB_USAGE_PATH, "w") as f:
        json.dump(vocab_usage, f, indent=2)

    print("vocab_usage.json has been updated.")


def get_least_used_words(category, count):
    # Load the current usage
    with open(VOCAB_USAGE_PATH, "r") as f:
        vocab_usage = json.load(f)

    # Calculate weights (inverse of usage count + 1 to avoid division by zero)
    words = list(vocab_usage[category].keys())
    weights = [1 / (usage + 1) for usage in vocab_usage[category].values()]

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Perform weighted random sampling
    selected_words = random.choices(words, weights=normalized_weights, k=count)

    return selected_words


def get_dialogue(llm_response: str):
    """
    Extract dialogue from an LLM response.

    :param llm_response: String containing the LLM's response
    :return: List of dialogue turns, or None if no valid dialogue is found
    """
    extracted_json = extract_json_from_llm_response(llm_response)
    if extracted_json and "dialogue" in extracted_json:
        return extracted_json["dialogue"]
    else:
        print("No valid dialogue found in the response")
        return None



def ensure_spacy_model(model_name="en_core_web_md"):
    try:
        spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model {model_name}...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])

def get_vocab_from_dialogue(dialogue: List[Dict[str, str]]) -> Set[Tuple[str, str]]:
    """For a given Engish dialogue, extracts the vocab used as the lemmas, the reason is so the
    known_vocab_list can be updated with usage information."""
    # Load the English language model
    ensure_spacy_model()
    nlp = spacy.load("en_core_web_md")
    
    # Initialize an empty set to store unique (lemma, pos) pairs
    vocab_set = set()
    
    # Iterate through each utterance in the dialogue
    for utterance in dialogue:
        # Process the text with spaCy
        doc = nlp(utterance['text'])
        
        # Iterate through each token in the processed document
        for token in doc:
            # Skip punctuation
            if token.pos_ != "PUNCT":
                # Add the lemma and its POS tag to the set
                vocab_set.add((token.lemma_.lower(), token.pos_))
    
    return vocab_set

