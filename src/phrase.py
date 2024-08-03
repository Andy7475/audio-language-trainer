import os
import requests
import spacy
from src.dialogue_generation import anthropic_generate, extract_json_from_llm_response
from typing import Dict, List
from src.config_loader import config


def get_text_from_dialogue(dialogue: List[Dict[str, str]]) -> List[str]:
    """ignoring the speaker, just gets all the utterances from a dialogue and puts
    them in a single list"""

    phrases = []
    for utterance in dialogue:
        phrases.append(utterance["text"])
    return phrases


def get_sentences_from_text(phrases: List[str]) -> List[str]:
    """Splits up phrases which might have more than one sentence per phrase and splits into a list of separate sentences.
    Returns a list of sentences.
    """

    nlp = spacy.load("en_core_web_md")
    sentences = []

    for phrase in phrases:
        doc = nlp(phrase)
        for sent in doc.sents:
            sentences.append(sent.text)
    return sentences


def generate_practice_phrases_from_dialogue(
    dialogue: List[Dict[str, str]]
) -> List[str]:
    """Uses an LLM call to create practice phrases from a longer dialogue"""
    phrases = get_text_from_dialogue(dialogue)

    llm_prompt = f"""
    I will provide you with a list of dialogue phrases. Your task is to create 15-20 new phrases based on this dialogue, following these strict rules:

    1. Use only the vocabulary, verbs, grammatical structures, and tenses present in the original dialogue.
    2. Do not introduce any new words, tenses, or grammatical concepts not present in the original.
    3. Change pronouns, verbs to accommodate new pronounds (e.g. they are to I am etc) rearrange words, and create shorter or simplified versions of the original phrases.
    4. Start with simple, short phrases (2-3 words) and gradually move to more complex ones (5-8 words). If the original dialogue has very long sentences then increase the length of the complex examples you give to match.
    5. Ensure that each new phrase is grammatically correct and makes sense on its own.
    6. Do not use any direct quotes from the original dialogue.
    7. Be creative in mixing elements from different parts of the dialogue.

    Here's a short example to illustrate the task:

    Original dialogue: 
    ["I love eating an apple every day.", "Apples are healthy and delicious."]

    Example output:
    {{
    "new_phrases": [
        "We love",
        "They love eating",
        "You love every day",
        "It's delicious",
        "We eat every day",
        "You're healthy",
        "You love apples",
        "A delicious apple",
        "Eating is healthy",
        "We love delicious apples",
        "You eat healthy apples every day",
        "They're delicious and healthy"
    ]
    }}

    Now, here's the dialogue for you to work with:
    {phrases}

    Please provide your output as a JSON object with a single key "new_phrases" whose value is an array of the new phrases. Start with simpler phrases and progress to more complex variations. Remember to stick strictly to the vocabulary and grammatical structures present in the original dialogue, while avoiding direct quotes. Your response should be valid JSON that can be parsed programmatically.
    """

    llm_response = anthropic_generate(llm_prompt)
    new_phrases = extract_json_from_llm_response(llm_response)["new_phrases"]

    # now add the original sentences onto those
    original_sentences = get_sentences_from_text(phrases)

    return new_phrases + original_sentences


def correct_phrases(phrases: List[str]) -> List[str]:
    """Uses an LLM to correct the phrases"""

    phrase_correct_prompt = f"""Given this list of phrases, 
    which are for language learning (so small parts of sentences are OK) please correct them slightly if they need it. 
    Maybe adjusting the grammar slightly or adding a word or two to make it more sensible phrase.
    The majority will probably not need any adjustment. Do not add any words not already present in the list. Return the modified list as a single element in JSON. {{\"corrected_phrases\" : [\"phrase one\", \"phrase two\", ...]}}.
    Phrases: {phrases} """

    response = anthropic_generate(
        phrase_correct_prompt, model=config.ANTHROPIC_SMALL_MODEL_NAME
    )
    corrected_phrases = extract_json_from_llm_response(response)
    return corrected_phrases["corrected_phrases"]
