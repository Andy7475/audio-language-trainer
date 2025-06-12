import json
from typing import Dict, List, Optional

from dotenv import load_dotenv

from src.config_loader import config
from src.utils import (
    anthropic_generate,
    extract_json_from_llm_response,
)
from src.generate import add_translations
from src.gcs_storage import upload_to_gcs, get_story_translated_dialogue_path, get_story_dialogue_path

load_dotenv()  # so we can use environment variables for various global settings


def get_story_prompt(verbs: list, vocab: list) -> str:
    """Creates a prompt for story generation, adapting length and complexity based on vocabulary size.
    Primarily uses verb count to determine story structure, as verbs drive the action.

    Args:
        verbs: List of verbs to incorporate
        vocab: List of other vocabulary words

    Returns:
        str: Prompt for story generation including output format specification
    """

    # Determine story structure based primarily on verb count
    verb_count = len(verbs)

    if verb_count < 10:
        parts = {"story": "A single focused scene showing natural interaction"}
        target_length = "30-45 seconds (about 75-100 words)"

    elif verb_count < 30:
        parts = {
            "setup": "Establish the situation and context",
            "resolution": "Complete the interaction",
        }
        target_length = "~1 minute per part (about 100 - 150 words each)"

    else:
        parts = {
            "introduction": "Set up the situation and characters",
            "development": "Present a challenge or complication",
            "resolution": "Resolve the situation",
        }
        target_length = "1-2 minutes per part (about 150-300 words each)"

    parts_description = "\n   - ".join(f"{k} ({v})" for k, v in parts.items())

    prompt = f"""Create a story named in the format {{story_name: "descriptive title here"}} followed by dialogue between Alex and Sam.
The story_name must be 3 words long and describe the main theme or setting, yet be engaging.

Story Requirements:
1. The story should have {len(parts)} part(s):
   - {parts_description}
   Each part should be about {target_length} when spoken.

2. Use vocabulary naturally from these lists ({verb_count} verbs and {len(vocab)} other words):
   Verbs: {', '.join(verbs)}
   Other words: {', '.join(vocab)}
   
3. Guidelines:
   - Using the above vocabulary, use whatever words make an engaging story (don't force a story to include all the words)
   - Only add essential connecting words (e.g., greetings, pronouns, or basic verbs (am, have, want, like etc) if necessary)
   - Keep dialogue exchanges realistic (about 10-15 words per utterance)
   - Alternate between Alex and Sam
   - Focus on quality over length - better a shorter, coherent story than a forced longer one
   - Focus on quality over using all the vocab - better a coherent story using less vocab

4. Output Format:
{{
    "story_name": "Engaging Title (3 words long)",
    {','.join(f'''
    "{part}": {{
        "dialogue": [
            {{"speaker": "Alex", "text": "..."}},
            {{"speaker": "Sam", "text": "..."}}
        ]
    }}''' for part in parts.keys())}
}}

Create a natural, engaging story that uses the provided vocabulary while maintaining good flow and coherence."""

    return prompt


def generate_story(vocab_dict: Dict[str, List[str]]) -> Dict:
    """Extract dialogue from an LLM response and validate structure.

    Args:
        vocab_dict: Dictionary with keys 'verbs' and 'vocab' containing word lists

    Returns:
        Dictionary with story parts as keys and dialogue lists as values, or None if invalid
    """

    prompt = get_story_prompt(verbs=vocab_dict["verbs"], vocab=vocab_dict["vocab"])
    llm_response = anthropic_generate(prompt, max_tokens=4000)
    story_data = extract_json_from_llm_response(llm_response)

    if not story_data:
        print("No valid JSON found in the response")
        return None

    story_name = story_data.get("story_name")
    if not story_name:
        print("Missing story_name in response")
        return None

    # Extract dialogue parts, excluding story_name
    story_dialogue = {k: v for k, v in story_data.items() if k != "story_name"}

    if not story_dialogue:
        print("No valid dialogue found in the response")
        return None

    # Verify the structure and speakers in each part
    valid_speakers = {"Sam", "Alex"}

    for part, content in story_dialogue.items():
        # Check basic structure
        if not isinstance(content, dict) or "dialogue" not in content:
            print(f"Invalid dialogue structure in part: {part}")
            return None

        # Check each dialogue entry has valid speakers
        for utterance in content["dialogue"]:
            if not isinstance(utterance, dict) or "speaker" not in utterance:
                print(f"Missing speaker in dialogue: {utterance}")
                return None

            if utterance["speaker"] not in valid_speakers:
                print(
                    f"Invalid speaker '{utterance['speaker']}' in dialogue. Must be 'Sam' or 'Alex'"
                )
                return None

    # Return the validated data with story_name included
    print(f"generated story: {story_name}")
    return story_name, story_dialogue


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


def upload_dialogue_to_gcs(
    dialogue_dict: Dict,
    story_name: str,
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
) -> str:
    """
    Upload the dialogue.json file to Google Cloud Storage.

    Args:
        dialogue_dict: Dictionary containing the dialogue data
        story_name: Name of the story (e.g., 'murder_mystery')
        collection: Collection name (e.g., 'LM1000', 'LM2000')
        bucket_name: Optional bucket name. Defaults to config.GCS_PRIVATE_BUCKET

    Returns:
        GCS URI of the uploaded file
    """
    if bucket_name is None:
        from src.config_loader import config

        bucket_name = config.GCS_PRIVATE_BUCKET

    # Ensure story_name is properly formatted
    story_name = (
        f"story_{story_name}" if not story_name.startswith("story_") else story_name
    )

    # Upload the JSON data using the utility function
    gcs_uri = upload_to_gcs(
        obj=dialogue_dict,
        bucket_name=bucket_name,
        file_name=get_story_dialogue_path(story_name=story_name, collection=collection),
    )

    print(f"Dialogue uploaded to {gcs_uri}")
    return gcs_uri


def translate_and_upload_dialogue(
    dialogue_dict: Dict,
    story_name: str,
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
) -> str:
    """
    Translate dialogue and upload the translated JSON to GCS.

    Args:
        dialogue_dict: Dictionary containing the dialogue data
        story_name: Name of the story
        language_name: Name of the target language (e.g., 'french')
        collection: Collection name (e.g., 'LM1000', 'LM2000')
        bucket_name: Optional bucket name

    Returns:
        GCS URI of the uploaded translated file
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    # Ensure story_name is properly formatted
    story_name = (
        f"story_{story_name}" if not story_name.startswith("story_") else story_name
    )

    # Translate the dialogue
    translated_dict = add_translations(dialogue_dict)

    # Create the base prefix and filename
    language_name = config.TARGET_LANGUAGE_NAME.lower()
    file_name = get_story_translated_dialogue_path(story_name, collection=collection)

    # Upload the translated JSON data
    gcs_uri = upload_to_gcs(
        obj=translated_dict,
        bucket_name=bucket_name,
        file_name=file_name,
    )

    print(f"Translated dialogue uploaded to {gcs_uri}")
    return gcs_uri
