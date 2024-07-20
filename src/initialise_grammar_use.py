import json

# This file takes grammar_concepts.json and creates a copy with usage information
# ready to be customised by the learner. It adds a boolean 'use' (whether or not to use this term)
# in dialogue, and a 'times_seen' which tracks how many times the learner has been exposed to it


def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)


def initialize_usage_data(concepts_data):
    usage_data = {}
    for category, items in concepts_data.items():
        usage_data[category] = {}
        for item in items:
            usage_data[category][item["name"]] = {
                "use": True,
                "times_seen": 0,
                "example": item["example"],
            }
    return usage_data


def main():
    concepts_file = "data/grammar_concepts.json"
    usage_file = "data/grammar_concepts_usage.json"

    # Load the existing grammar concepts
    concepts_data = load_json(concepts_file)

    # Initialize or load the usage data
    try:
        usage_data = load_json(usage_file)
        print("Existing usage data loaded.")
    except FileNotFoundError:
        usage_data = initialize_usage_data(concepts_data)
        print("New usage data initialized.")

    # Save the usage data
    save_json(usage_data, usage_file)
    print(f"Usage data saved to {usage_file}")


if __name__ == "__main__":
    main()
