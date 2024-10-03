import os
import requests
import spacy
from src.dialogue_generation import anthropic_generate, extract_json_from_llm_response
from typing import Dict, List
from src.config_loader import config

import json
import random
from typing import List, Dict, Tuple
from collections import defaultdict


def load_longman_data(file_path: str) -> List[Dict]:
    with open(file_path, "r") as file:
        return json.load(file)


def filter_s1_words(data: List[Dict]) -> Dict[str, List[str]]:
    s1_words = defaultdict(list)
    for entry in data:
        if "S1" in entry.get("frequencies", []):
            for word_class in entry.get("word_classes", []):
                s1_words[word_class].append(entry["word"])
    return dict(s1_words)


def generate_phrases_with_llm(
    word_list: List[str], num_phrases: int = 100
) -> List[str]:

    prompt = f"""
    Task: Generate {num_phrases} unique English phrases using words from the provided list. Each phrase should be 6-9 words long and use a variety of verb tenses.

    Word List: {', '.join(word_list)}

    Requirements:
    1. Use only words from the provided list, common articles (a, an, the), and basic prepositions (we, you, they etc).
    2. Vary the verb tenses (present, past, future) across the phrases.
    3. Create meaningful and diverse phrases that could be useful for language learners.
    4. Ensure you use each word at least once

    Please return your response in the following JSON format:
    {{
        "phrases": [
            "Phrase 1",
            "Phrase 2",
            ...
        ]
    }}
    """

    llm_response = anthropic_generate(prompt)
    # result = extract_json_from_llm_response(llm_response)

    return llm_response


def update_word_usage(data: List[Dict], used_words: List[str]) -> List[Dict]:
    for entry in data:
        if entry["word"] in used_words:
            entry["used"] = True
    return data


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
    I will provide you with a list of dialogue phrases. Your task is to create 20-30 new phrases based on this dialogue, this is to support language learning
     where we learn new ways of rearranging the vocabulary to create a wider range of phrases.
    To help yourself, first list the verbs, tenses and other vocab within the phrases this will help you adhere to the following rules:
    1. Use only the vocabulary, verbs, grammatical structures, and tenses present in the original dialogue.
    2. Ensure every verb and noun is used in a short phrase (2 - 4 words).
    3. Do not introduce any new words, tenses, or grammatical concepts not present in the original.
    4. You can change pronouns for more practice with verbs (e.g. 'they are' to 'I am' etc), rearrange words, and create shorter or simplified versions of the original phrases.
    5. Start with simple, short phrases (2-4 words) and gradually move to more complex ones (5-8 words). If the original dialogue has very long sentences then increase the length of the complex examples you give to match.
    6. Ensure that each new phrase is grammatically correct and makes sense on its own.
    7. Do not use any direct quotes from the original dialogue (as we will practice this later).
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

    Please include in your output a JSON object with a single key "new_phrases" whose value is an array of the new phrases. 
    """

    llm_response = anthropic_generate(llm_prompt)
    new_phrases = extract_json_from_llm_response(llm_response)["new_phrases"]

    # now add the original sentences onto those
    original_sentences = get_sentences_from_text(phrases)

    return new_phrases + original_sentences
