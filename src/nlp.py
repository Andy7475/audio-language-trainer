import re
import subprocess
import sys
import spacy
from typing import Dict, List, Set, Tuple
from spacy.tokens import Token

def load_spacy_model():
    """Load spaCy model with error handling."""
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_md")
        return spacy.load("en_core_web_md")
    
def extract_vocab_and_pos(english_phrases: List[str]) -> List[Tuple[str, str]]:
    """Returns the (lemma and POS) for feeding into update_vocab_usage, as a list."""
    # Process vocabulary
    nlp = load_spacy_model()

    vocab_set = set()
    excluded_names = {"sam", "alex"}

    for phrase in english_phrases:
        doc = nlp(phrase)

        for token in doc:
            if (
                token.pos_ != "PUNCT"
                and token.ent_type_ != "PERSON"
                and token.text.lower() not in excluded_names
            ):
                vocab_set.add((token.lemma_.lower(), token.pos_))

    return vocab_set


def extract_substring_matches(
    new_phrases: List[str], target_phrases: Set[str]
) -> Set[str]:
    # Convert all new phrases to lowercase
    lowercase_phrases = [phrase.lower() for phrase in new_phrases]

    # Convert all target phrases to lowercase
    lowercase_targets = [target.lower() for target in target_phrases]

    # Initialize a set to store matched substrings
    matched_substrings = set()

    # Check each target phrase against each new phrase
    for target in lowercase_targets:
        # Create a regex pattern that matches the target as a whole word or phrase
        pattern = r"\b" + re.escape(target) + r"\b"
        for phrase in lowercase_phrases:
            if re.search(pattern, phrase):
                matched_substrings.add(target)
                break  # Move to the next target once a match is found

    return matched_substrings


def extract_spacy_lowercase_words(new_phrases: List[str]) -> Set[str]:
    # Ensure the spaCy model is loaded
    nlp = load_spacy_model()

    # Initialize an empty set to store unique lowercase words
    lowercase_words = set()

    # Process each phrase with spaCy
    for phrase in new_phrases:
        doc = nlp(phrase)

        # Add the lowercase version of each token's text to the set
        lowercase_words.update(token.text.lower() for token in doc)

    return lowercase_words


def get_verb_and_vocab_lists(used_words: Set[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Separate the input set of (word, POS) tuples into verb and vocabulary lists.

    Args:
    used_words (Set[Tuple[str, str]]): A set of tuples containing (word, POS)

    Returns:
    Dict[str, List[str]]: A dictionary with 'verbs' and 'vocab' lists
    """
    verb_list = []
    vocab_list = []

    for word, pos in used_words:
        if pos in ["VERB", "AUX"]:
            verb_list.append(word)
        else:
            vocab_list.append(word)

    return {"verbs": verb_list, "vocab": vocab_list}

def extract_content_words(phrase: str, nlp) -> Set[Tuple[str, str]]:
    """
    Extract content words (verbs and meaningful vocabulary) from a phrase.
    Returns set of (lemma, pos) tuples.
    """
    doc = nlp(phrase.lower())
    
    # Define parts of speech to exclude
    exclude_pos = {'DET', 'PUNCT', 'SPACE', 'PART', 'CCONJ', 'SCONJ', 'ADP', 'PRON', 'PROPN'}
    
    content_words = set()
    for token in doc:
        # Only include if:
        # 1. Not in excluded POS tags
        # 2. Not a stop word (unless it's a verb)
        if (token.pos_ not in exclude_pos and 
            (not token.is_stop or token.pos_ == 'VERB')):
            content_words.add((token.lemma_.lower(), token.pos_))
    
    return content_words

def check_vocab_match(phrase_words: Set[Tuple[str, str]], vocab_dict: Dict[str, List[str]]) -> bool:
    """
    Check if the content words from the phrase match the vocabulary dictionary.
    Returns True if all content words are found in the vocabulary lists.
    """
    for lemma, pos in phrase_words:
        if pos in ['VERB', 'AUX']:
            if lemma not in vocab_dict.get('verbs', []):
                return False
        else:
            if lemma not in vocab_dict.get('vocab', []):
                return False
    return True

def phrase_matches_vocab(english_phrase: str, vocab_dictionary: Dict[str, List[str]]) -> bool:
    """
    Check if a phrase only uses words from the provided vocabulary dictionary.
    
    Args:
        english_phrase: The English phrase to check
        vocab_dictionary: Dictionary with 'verbs' and 'vocab' lists
        
    Returns:
        bool: True if all content words in the phrase are found in the vocabulary lists
    """
    nlp = load_spacy_model()
    
    # Convert vocabulary lists to lowercase for matching
    vocab_dict = {
        'verbs': [v.lower() for v in vocab_dictionary.get('verbs', [])],
        'vocab': [v.lower() for v in vocab_dictionary.get('vocab', [])]
    }
    
    # Extract content words from the phrase
    phrase_words = extract_content_words(english_phrase, nlp)
    
    # Check if all content words are in the vocabulary
    return check_vocab_match(phrase_words, vocab_dict)

def filter_matching_phrases(phrases: List[str], vocab_dictionary: Dict[str, List[str]]) -> List[str]:
    """
    Filter a list of phrases to only include those that match the vocabulary.
    
    Args:
        phrases: List of English phrases to check
        vocab_dictionary: Dictionary with 'verbs' and 'vocab' lists
        
    Returns:
        List[str]: Filtered list of phrases that only use words from the vocabulary
    """
    nlp = load_spacy_model()
    
    # Convert vocabulary lists to lowercase for matching
    vocab_dict = {
        'verbs': [v.lower() for v in vocab_dictionary.get('verbs', [])],
        'vocab': [v.lower() for v in vocab_dictionary.get('vocab', [])]
    }
    
    matching_phrases = []
    for phrase in phrases:
        phrase_words = extract_content_words(phrase, nlp)
        if check_vocab_match(phrase_words, vocab_dict):
            matching_phrases.append(phrase)
            
    return matching_phrases