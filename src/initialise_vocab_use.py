import json


def initialize_vocab_usage():
    # Load the known vocabulary
    with open("data/known_vocab_list.json", "r") as f:
        known_vocab = json.load(f)

    # Initialize the usage dictionary
    vocab_usage = {
        "verbs": {verb: 0 for verb in known_vocab["verbs"]},
        "vocab": {word: 0 for word in known_vocab["vocab"]},
    }

    # Save the usage dictionary to a new JSON file
    with open("data/vocab_usage.json", "w") as f:
        json.dump(vocab_usage, f, indent=2)

    print("vocab_usage.json has been initialized.")


if __name__ == "__main__":
    # Initialize the vocab_usage.json file
    initialize_vocab_usage()
