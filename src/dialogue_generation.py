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
)

load_dotenv()  # so we can use environment variables for various global settings

GRAMMAR_USAGE_PATH = "../data/grammar_concepts_usage.json"
VOCAB_USAGE_PATH = "../data/vocab_usage.json"
STORY_PLAN_PATH = "../outputs/story_plan.json"
RECAP_PATH = "../outputs/story_recap.json"


def generate_story_plan(story_guide: str = None, test=True) -> dict:
    """This function generates an outline story following the story_structure written below.
    When we are creating dialgoue, we will feed the story_plan into the function a bit at a time for subsequent dialogues
    so there is some continuity.

    returns: a JSON story plan, but it also saves it to STORY_PLAN_PATH"""
    story_structure = [
        "Exposition: Introduce the main characters (Alex and Sam) and setting",
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
    story_plan = extract_json_from_llm_response(llm_response)

    # Normalize keys to lowercase
    story_plan = {k.lower(): v for k, v in story_plan.items()}

    save_json(story_plan, STORY_PLAN_PATH)
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
    verb_count=10,
    verb_use_count=3,
    vocab_count=30,
    vocab_use_count=10,
    grammar_concept_count=10,
    grammar_use_count=3,
):
    # Load the JSON files
    grammar_concepts = load_json(GRAMMAR_USAGE_PATH)

    # Select verbs
    verbs = get_least_used_words("verbs", verb_count)
    verbs_str = ", ".join(verbs)

    # Select vocabulary
    vocab = get_least_used_words("vocab", vocab_count)
    vocab_str = ", ".join(vocab)

    # Select grammar concepts
    selected_concepts = select_grammar_concepts(grammar_concepts, grammar_concept_count)
    used_concepts = selected_concepts[:grammar_use_count]

    # Update grammar concept usage
    update_grammar_concept_usage(grammar_concepts, used_concepts)

    # Generate the prompt
    prompt = f"""Create a brief dialogue for language learners using the following guidelines:
    1. This is for language practice. Only use words from the lists provided below.
    2. Pick from about {verb_use_count} of these verbs:
    {verbs_str}
    You can use other verbs if required to make the tenses work (e.g., auxiliary verbs).
    3. Use at least {vocab_use_count} words from this vocabulary list:
    {vocab_str}
    4. Focus on these grammatical concepts (use each at least once, but no more than twice):
    {', '.join(used_concepts)}
    5. Create a conversation with 6-8 lines of dialogue total (about 20-30 seconds when spoken).
    6. The characters are Alex and Sam. Maintain their personalities and relationship from previous dialogues.
    7. Recap from last episode: {last_recap}. You can (if necessary) reuse vocab words from the recap to maintain continuity, but prioritise the vocab list provided.
    8. Output the dialogue in JSON format with speakers clearly labeled. For example:
    {{
        "dialogue": [
        {{"speaker": "Alex", "text": "Hello! How are you today?"}},
        {{"speaker": "Sam", "text": "I'm doing well, thanks for asking."}}
        ]
    }}
    8. Current story phase and plan: {story_part}: {story_part_outline}
    9. Create dialogue for the current story plan.
    Remember:
    - Only use words from the provided lists.
    - Balance practicing the specified concepts and vocabulary (about 60%) with advancing the storyline (about 40%).
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

    with open(GRAMMAR_USAGE_PATH, "w") as f:
        json.dump(grammar_concepts, f, indent=2)


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


def generate_dialogue(dialogue_prompt: str):
    """
    Extract dialogue from an LLM response.

    :param llm_response: String containing the LLM's response
    :return: List of dialogue turns, or None if no valid dialogue is found
    """
    llm_response = anthropic_generate(dialogue_prompt)
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
    vocab_usage.json can be updated with usage information. Returns of the form (lemma, POS) e.g. ("go", "VERB")
    """
    # Load the English language model
    ensure_spacy_model()
    nlp = spacy.load("en_core_web_md")

    # Initialize an empty set to store unique (lemma, pos) pairs
    vocab_set = set()

    # Iterate through each utterance in the dialogue
    for utterance in dialogue:
        # Process the text with spaCy
        doc = nlp(utterance["text"])

        # Iterate through each token in the processed document
        for token in doc:
            # Skip punctuation
            if token.pos_ != "PUNCT":
                # Add the lemma and its POS tag to the set
                vocab_set.add((token.lemma_.lower(), token.pos_))

    return vocab_set


def update_vocab_usage(used_words: Set[Tuple[str, str]]):
    """Taking a list of (word, POS) e.g. ('can', 'VERB') we update the vocab_usage
    list, if the word doesn't exist we add it to list. This is used for sampling vocab for subsequent
    lessons. words that haven't been used have a higher chance of being sampled.

    No return statement"""
    # Load the current usage

    vocab_usage = load_json(VOCAB_USAGE_PATH)

    # Update the usage count for each used word
    for word, pos in used_words:
        if pos == "VERB":
            if word in vocab_usage["verbs"]:
                vocab_usage["verbs"][word] += 1
            else:
                vocab_usage["verbs"][word] = 1
        else:
            if word in vocab_usage["vocab"]:
                vocab_usage["vocab"][word] += 1
            else:
                vocab_usage["vocab"][word] = 1

    # Save the updated usage dictionary
    save_json(vocab_usage, VOCAB_USAGE_PATH)
