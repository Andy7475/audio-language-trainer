from src.utils import load_json, save_json
from src.config_loader import config
import os

# This file takes grammar_concepts.json and creates a copy with usage information
# ready to be customised by the learner. It adds a boolean 'use' (whether or not to use this term)
# in dialogue, and a 'times_seen' which tracks how many times the learner has been exposed to it

CONCEPTS_FILE = "../data/grammar_concepts.json"
CONCEPTS_USAGE_FILE = "../data/grammar_concepts_usage.json"
VOCAB_LIST = "../data/known_vocab_list.json"
VOCAB_USAGE_FILE = "../data/vocab_usage.json"
DATA_DIR = "../data"


def initialise_usage_data(overwrite=False):
    """Creates two usage JSON files and sets use counts to 0 for all words.
    Only use once when setting up this project"""

    # Check if files exist and only proceed if overwrite is True
    if (
        os.path.exists(config.GRAMMAR_USAGE_PATH)
        or os.path.exists(config.VOCAB_USAGE_PATH)
    ) and not overwrite:
        print("Usage files already exist. Set overwrite=True to reinitialize.")
        return

    initialise_grammar_usage()
    initialise_vocab_usage()


def initialise_grammar_usage():

    concepts_data = load_json(config.GRAMMAR_CONCEPTS)
    usage_data = {}
    for category, items in concepts_data.items():
        usage_data[category] = {}
        for item in items:
            usage_data[category][item["name"]] = {
                "use": True,
                "times_seen": 0,
                "example": item["example"],
            }
    save_json(usage_data, config.GRAMMAR_USAGE_PATH)


def initialise_vocab_usage():
    # Load the known vocabulary
    known_vocab = load_json(config.VOCAB_LIST)
    # Initialize the usage dictionary
    vocab_usage = {
        "verbs": {verb: 0 for verb in known_vocab["verbs"]},
        "vocab": {word: 0 for word in known_vocab["vocab"]},
    }

    # Save the usage dictionary to a new JSON file
    save_json(vocab_usage, config.VOCAB_USAGE_PATH)
