import asyncio
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import spacy
from tqdm import tqdm

from src.config_loader import config
from src.convert import clean_filename
from src.dialogue_generation import anthropic_generate, extract_json_from_llm_response
from src.gcs_storage import (
    get_phrase_audio_path,
    get_phrase_image_path,
    get_story_collection_path,
    get_translated_phrases_path,
    read_from_gcs,
)
from src.nlp import (
    extract_spacy_lowercase_words,
    extract_substring_matches,
    extract_vocab_and_pos,
    get_verb_and_vocab_lists,
    remove_matching_words,
)


def remove_phrases_with_no_new_words(
    known_phrases: List[str], new_phrases: List[str]
) -> List[str]:
    """
    Filter phrases from new_phrases that contain only words already present in known_phrases.

    Args:
        known_phrases: List of phrases containing already known words
        new_phrases: List of phrases to check for new words

    Returns:
        List of phrases from new_phrases that contain new words not in known_phrases

    Example:
        >>> known = ["the cat sat", "on the mat"]
        >>> new = ["the cat jumped", "a new day", "the mat sat"]
        >>> filter_phrases_with_new_words(known, new)
        ['the cat jumped', 'a new day']  # 'the mat sat' only uses known words
    """
    # Get set of known words by splitting phrases and converting to lowercase
    known_words = set()
    for phrase in known_phrases:
        known_words.update(word.lower() for word in phrase.split())

    # Filter phrases that contain at least one new word
    filtered_phrases = []
    for phrase in new_phrases:
        # Get set of words in this phrase
        phrase_words = {word.lower() for word in phrase.split()}

        # Check if phrase contains any new words
        if (
            phrase_words - known_words
        ):  # If there are words in phrase_words not in known_words
            filtered_phrases.append(phrase)

    return filtered_phrases


def get_phrase_indices(known_phrases: list[str], all_phrases: list[str]) -> set[int]:
    """
    Get the indices of known phrases within a list of all phrases, skipping any that aren't found.

    Args:
        known_phrases: List of phrases to find indices for
        all_phrases: Master list of phrases to search within

    Returns:
        Set of indices where known phrases were found
    """
    indices = set()
    for phrase in known_phrases:
        try:
            idx = all_phrases.index(phrase)
            indices.add(idx)
        except ValueError:
            continue

    return indices


def generate_scenario_vocab_building_phrases(
    scenario: str, localise: bool = False, num_phrases: str = "20-30"
) -> List[str]:
    """
    Generate vocabulary-rich phrases for a given scenario, focusing on nouns and adjectives.

    Args:
        scenario: Description of the scenario e.g. "at the restaurant", "taking a taxi"
        localise: Whether to include country-specific vocabulary
        num_phrases: Range of phrases to generate e.g. "20-30"

    Returns:
        List of English noun/adjective-focused phrases suitable for vocabulary building

    Raises:
        ValueError: If unable to generate valid phrases or extract JSON
    """
    localisation_phrase = ""
    localisation_phrase_2 = ""
    if localise:
        localisation_phrase = f"- Consider local items and descriptions specific to {config.TARGET_COUNTRY_NAME}"
        localisation_phrase_2 = f" when travelling to {config.TARGET_COUNTRY_NAME}"

    prompt = """Generate vocabulary-rich phrases for: '{scenario}' (write in UK English). These are for language learners{localisation_phrase_2} to build their vocabulary through memorable noun-adjective combinations.

Approach:
1. First, identify 6-8 key categories of things/objects in this scenario
   For each category list relevant:
   - Common objects/items
   - Descriptive adjectives (size, quality, temperature, etc.)
   - Associated equipment/furnishings
   - Typical problems or variations
   - Use relatively simple vocabulary that would be useful in spoken phrases

2. Combine these into natural 3-7 word phrases that:
   - Use multiple nouns/adjectives together
   - Create clear mental images
   - Avoid complex verbs (use 'with', 'in', 'on', 'for' to connect)
   - Use relatively simple vocabulary that would be useful in spoken phrases
   - Would be easily illustrated

Remember:
- Focus on physical objects and their descriptions
- Consider location/placement phrases
- Consider variations (size, quality, condition)  
- Consider typical problems or issues
{localisation_phrase}

Return {num_phrases} phrases in this JSON format:
Example input scenario: 'eating at a restaurant'
Example JSON output:
{{
    "phrases": [
        "a fresh green salad with olives",
        "some broken glass on a large table",
        "the spicy noodle soup with prawns"
    ]
}}

Each phrase should:
- Be 3-7 words long
- Contain at least 2 content words (nouns/adjectives)
- Create a clear mental image
- Focus on objects rather than actions"""

    # Format the prompt with the given parameters
    formatted_prompt = prompt.format(
        localisation_phrase=localisation_phrase,
        scenario=scenario,
        num_phrases=num_phrases,
        localisation_phrase_2=localisation_phrase_2,
    )

    # Generate response using Claude
    response = anthropic_generate(formatted_prompt, max_tokens=2000)

    # Extract JSON from response
    json_data = extract_json_from_llm_response(response)

    if not json_data or "phrases" not in json_data:
        raise ValueError("Failed to generate valid phrases")

    return json_data["phrases"]


def generate_scenario_phrases(
    scenario: str, localise: bool = False, num_phrases: str = "10-15"
) -> List[str]:
    """
    Generate a list of useful phrases for a given scenario and country.

    Args:
        country_name: Name of the country e.g. "France", "Japan"
        scenario: Description of the scenario e.g. "at the restaurant", "taking a taxi"

    Returns:
        List of English phrases suitable for the scenario

    Raises:
        ValueError: If unable to generate valid phrases or extract JSON
    """

    localisation_phrase = ""
    localisation_phrase_2 = ""
    if localise:
        localisation_phrase = (
            f"- Cultural considerations specific to {config.TARGET_COUNTRY_NAME}"
        )
        localisation_phrase_2 = f" travelling to {config.TARGET_COUNTRY_NAME}"

    # Base prompt template
    prompt = """Generate practical phrases for: '{scenario}' (write in UK english). These are for language learners{localisation_phrase_2}.

    Approach this by:
    1. Break down {scenario} into a common sequence of situations
    2. For each situation, consider:
    - What someone might need to ask for
    - How they might respond to questions
    - Common problems they might face
    - Typical interactions with locals

    For each situation, create 3-4 phrases that:
    - Are 4-7 words in length
    - Use common speaking patterns as a basis (e.g. "Could I have...", "Where is...", "Are there any...")
    - But never leave phrases incomplete (e.g. insert an example location 'when is the next train to <city>?' rather than just the stem 'when is the next train to...?')
    - Would be naturally used in conversation
    - Focus on clear, practical communication
    - Use shorter phrases where possible

    Consider:
    - The logical sequence of events
    - Both asking and responding
    - Common issues or special requests
    {localisation_phrase}
    - Local customs and expectations

    Return {num_phrases} phrases in this JSON format:
    Example input scenario: 'visiting tourist attractions'
    Example JSON output:
    {{
        "phrases": [
            "Which way to the museum, please?",
            "What time does the museum close?",
            "How much does a ticket cost?",
            "etc...",
        ]
    }}

    Important:
    - Each phrase should be complete and standalone
    - Include a mix of questions and statements
    - Focus on everyday language people actually use
    - Consider both formal and informal situations where appropriate"""

    # Format the prompt with the given parameters
    formatted_prompt = prompt.format(
        localisation_phrase=localisation_phrase,
        scenario=scenario,
        num_phrases=num_phrases,
        localisation_phrase_2=localisation_phrase_2,
    )

    # Generate response using Claude
    response = anthropic_generate(formatted_prompt, max_tokens=2000)

    # Extract JSON from response
    json_data = extract_json_from_llm_response(response)

    if not json_data or "phrases" not in json_data:
        raise ValueError("Failed to generate valid phrases")

    return json_data["phrases"]


def generate_phrases_from_vocab_dict(
    vocab_dict: Dict[str, List[str]],
    max_iterations: int = 10,
    length_phrase: str = "6-9 words long",
    verbs_per_phrase: str = "one or two verbs",
    num_phrases: int = 100,
    localise: bool = False,  # whether to make the phrases aligned to the country
) -> List[str]:
    """This takes a dict with keys 'verbs' and 'vocab'  and constructs phrases using them, iterating through until all words are exhausted.
    Desgined for Longman Communication vocab lists with a verb : vocab ratio of about 1:4 and 1000 words tends to generate around 850 phrases in
    a range of tenses. A list of english phrases are returned."""

    verb_list_set = set(vocab_dict["verbs"])
    vocab_list_set = set(vocab_dict["vocab"])

    LONGMAN_PHRASES = []
    all_verbs_used = set()
    iteration_count = 0

    while (len(vocab_list_set) >= 5) and iteration_count < max_iterations:
        iteration_count += 1

        if len(vocab_list_set) < 150:
            # Switch to minimal phrase generation and use any verbs
            response = generate_minimal_phrases_with_llm(
                list(vocab_list_set) + list(verb_list_set),
                length_phrase=length_phrase,
                verbs_per_phrase=verbs_per_phrase,
            )
            new_phrases = extract_json_from_llm_response(response)["phrases"]
            vocab_used = extract_vocab_and_pos(new_phrases)
            words_used = get_verb_and_vocab_lists(vocab_used)

            verb_list_set = remove_matching_words(words_used["verbs"], verb_list_set)
            vocab_list_set = remove_matching_words(words_used["vocab"], vocab_list_set)

            # match on text as well
            all_lowercase_words = extract_spacy_lowercase_words(new_phrases)
            verb_list_set = remove_matching_words(all_lowercase_words, verb_list_set)
            vocab_list_set = remove_matching_words(all_lowercase_words, vocab_list_set)

            # match on substring - to account for 'words' in the longman dictionary that are > 1 words like 'aware of'
            substring_matches = extract_substring_matches(new_phrases, vocab_list_set)
            vocab_list_set = remove_matching_words(substring_matches, vocab_list_set)

            LONGMAN_PHRASES.extend(new_phrases)

            print(f"Iteration {iteration_count}/{max_iterations}")
            print(f"Generated {len(new_phrases)} phrases - with minimal phrase prompt")
            print(
                f"We have {len(verb_list_set)} verbs and {len(vocab_list_set)} vocab words left"
            )
            continue
        else:
            if len(verb_list_set) < 10:
                num_phrases = min(len(vocab_list_set), 100)  # Focus on exhausting vocab
                verb_list_set.update(all_verbs_used)  # Reintroduce all used verbs
            elif len(verb_list_set) < 50 and len(vocab_list_set) > 100:
                num_phrases = min(
                    len(verb_list_set) * 2, 100
                )  # Ensure we use all verbs
            else:
                num_phrases = 100

        verb_sample_size = min(75, len(verb_list_set))
        vocab_sample_size = min(75 * 3, len(vocab_list_set))

        verb_list_for_prompt = random.sample(list(verb_list_set), k=verb_sample_size)
        vocab_list_for_prompt = random.sample(list(vocab_list_set), k=vocab_sample_size)

        response = generate_phrases_with_llm(
            verb_list=verb_list_for_prompt,
            vocab_list=vocab_list_for_prompt,
            num_phrases=num_phrases,
            length_phrase=length_phrase,
            verbs_per_phrase=verbs_per_phrase,
            localise=localise,
        )

        new_phrases = extract_json_from_llm_response(response)["phrases"]
        LONGMAN_PHRASES.extend(new_phrases)

        # we now pull out the POS and words used in the phrases we just generated and split them back into
        # a dictionary with keys 'verbs' and 'vocab'
        vocab_pos_used = extract_vocab_and_pos(new_phrases)
        words_used = get_verb_and_vocab_lists(vocab_pos_used)

        verb_list_set = remove_matching_words(words_used["verbs"], verb_list_set)
        vocab_list_set = remove_matching_words(words_used["vocab"], vocab_list_set)

        # match on exact word text as well as sometimes the word in the longman dictionary is not a 'lemma' as generated by spacy
        all_lowercase_words = extract_spacy_lowercase_words(new_phrases)
        vocab_list_set = remove_matching_words(all_lowercase_words, vocab_list_set)

        all_verbs_used.update(words_used["verbs"])

        print(f"Iteration {iteration_count}/{max_iterations}")
        print(f"Generated {len(new_phrases)} phrases")
        print(
            f"We have {len(verb_list_set)} verbs and {len(vocab_list_set)} vocab words left"
        )

    if iteration_count == max_iterations:
        print(
            f"Reached maximum number of iterations ({max_iterations}). Stopping phrase generation."
        )
    else:
        print("All words have been used. Phrase generation complete.")

    return LONGMAN_PHRASES


def generate_phrases_with_llm(
    verb_list: List[str],
    vocab_list: List[str],
    num_phrases: int = 100,
    length_phrase: str = "6-9 words long",
    verbs_per_phrase: str = "one or two verbs",
    localise: bool = False,
) -> List[str]:

    localise_prompt_segment = ""  # default is blank
    if localise:
        localise_prompt_segment = f"- Localisation: {config.TARGET_COUNTRY_NAME} (For applying any societal, cultural or location elements to the phrases, such as city names etc) - but do not localize every phrase, the phrases can be generic."
    prompt = f"""
    Task: Generate {num_phrases} unique British English phrases using words from the provided verb and vocabulary lists. Each phrase should be {length_phrase} and use {verbs_per_phrase} (no more) per phrase.

    Verb List: {', '.join(verb_list)}
    Vocabulary List: {', '.join(vocab_list)}

    Requirements:
    1. Use only words from the provided lists, common articles (a, an, the), basic prepositions, and common pronouns (I, we, you, they, etc.).
    2. Each phrase must contain {verbs_per_phrase} from the verb list, and be {length_phrase}.
    3. Vary the verb tenses (present, past, future) across the phrases. Stick mainly to first and second person.
    4. Vary the type of phrase:
        - Imperative ("Don't be late...")
        - Simple statements ("The traffic was terrible...")
        - First-person expressions ("I enjoy...")
        - Question ("Shall we...?", "Do you ...?", "Did they...?")
        {localise_prompt_segment}
    
    5. Make phrases active rather than passive, something you would commonly say rather than read.
    6. Ensure each phrase is gramatically correct (so you may extend the length if required to meet this condition)
    7. Try to use all the words provided to create the {num_phrases} phrases.
    8. Make the phrases memorable by creating interesting or slightly humorous scenarios.

    Please return your response in the following JSON format:
    {{
        "phrases": [
            "Phrase 1",
            "Phrase 2",
            ...
        ]
    }}
    """

    # Here you would call your LLM with the prompt
    llm_response = anthropic_generate(prompt, max_tokens=5000)
    return llm_response


def generate_minimal_phrases_with_llm(
    word_list: List[str],
    length_phrase: str = "6-9 words long",
    verbs_per_phrase: str = "one or two verbs",
) -> List[str]:
    """We don't localise the minimal phrases as we are trying to exhaust the vocab"""

    prompt = f"""
    Task: Create the minimum number of English phrases necessary to use all the words from the provided list at least once. Each phrase should be {length_phrase}.

    Word List: {', '.join(word_list)}

    Requirements:
    1. Use all words from the provided list at least once across all phrases.
    2. Create the minimum number of phrases possible while meeting requirement 1.
    3. Each phrase must contain {verbs_per_phrase} from the verb list, and be {length_phrase}.
    3a. Ensure each phrase is gramatically correct (so you may extend the length if required to meet this condition)
    4. You may use additional common words (articles, prepositions, pronouns, basic verbs) that a beginner language learner would know to complete phrases.
    5. Prioritize exhausting the provided word list over creating a large number of phrases.
    6. Vary the verb tenses (present, past, future) across the phrases. Stick mainly to first and second person.
    7. Vary the type of phrase:
        - Imperative ("Don't be late...")
        - Simple statements ("The traffic was terrible...")
        - First-person expressions ("I enjoy...")
        - Question ("Shall we...?", "Do you ...?", "Did they...?")
    8. Make phrases active rather than passive, something you would commonly say rather than read.
    9. Make the phrases memorable by creating interesting or slightly humorous scenarios when possible.

    Please return your response in the following JSON format:
    {{
        "phrases": [
            "Phrase 1",
            "Phrase 2",
            ...
        ]
    }}
    """

    # Here you would call your LLM with the prompt
    llm_response = anthropic_generate(prompt, max_tokens=5000)
    return llm_response


def update_word_usage(data: List[Dict], used_words: List[str]) -> List[Dict]:
    for entry in data:
        if entry["word"] in used_words:
            entry["used"] = True
    return data


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


def get_phrase_multimedia(
    phrase_key: str,
    bucket_name: str = config.GCS_PRIVATE_BUCKET,
) -> Dict[str, Any]:
    """
    Get all multimedia data for a single phrase from GCS.

    Args:
        phrase_key: Key identifying the phrase
        language: Target language code (e.g. 'french')
        bucket_name: GCS bucket name
        collection: Collection name (default: "LM1000")

    Returns:
        Dictionary containing:
        {
            "normal_audio": AudioSegment or None,
            "slow_audio": AudioSegment or None,
            "image": Image or None,
        }
    """
    try:
        # Get the translated phrases data

        normal_audio = None
        slow_audio = None
        try:
            normal_audio_path = get_phrase_audio_path(phrase_key, "normal")
            slow_audio_path = get_phrase_audio_path(phrase_key, "slow")

            normal_audio = read_from_gcs(bucket_name, normal_audio_path, "audio")
            slow_audio = read_from_gcs(bucket_name, slow_audio_path, "audio")
        except Exception as e:
            print(f"Error getting audio for phrase {phrase_key}: {str(e)}")

        # Get image
        image = None
        try:
            image_path = get_phrase_image_path(phrase_key)
            image = read_from_gcs(bucket_name, image_path, "image")
        except Exception as e:
            print(f"Error getting image for phrase {phrase_key}: {str(e)}")

        return {
            "normal_audio": normal_audio,
            "slow_audio": slow_audio,
            "image": image,
        }

    except Exception as e:
        print(f"Error getting data for phrase {phrase_key}: {str(e)}")
        return {}


def get_phrase_keys(
    story_name: str, collection: str, n: Optional[int] = None
) -> List[str]:
    """
    Get the list of phrase keys for a given story from the collection's story file.

    Args:
        story_name: Name of the story to get phrases for
        collection: Collection name (e.g. "LM1000")
        n: Optional number of phrases to return. If None, returns all phrases.

    Returns:
        List of phrase keys for the story
    """
    try:
        # Get the story collection data
        collection_path = get_story_collection_path(collection=collection)
        collection_data = read_from_gcs(
            config.GCS_PRIVATE_BUCKET, collection_path, "json"
        )

        if not collection_data or story_name not in collection_data:
            print(f"No story found: {story_name}")
            return []

        # Get English phrases for this story
        phrase_keys = [
            clean_filename(item["phrase"]) for item in collection_data[story_name]
        ]

        # Limit number of phrases if n is specified
        if n is not None:
            phrase_keys = phrase_keys[:n]

        return phrase_keys

    except Exception as e:
        print(f"Error getting phrase keys for story {story_name}: {str(e)}")
        return []


def build_phrase_dict_from_gcs(
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
    phrase_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a dictionary containing translated phrase data from GCS.

    Args:
        collection: Collection name (default: "LM1000")
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)
        phrase_keys: Optional list of phrase keys to include. If None, includes all phrases.

    Returns:
        Dictionary in format:
        {
            "phrase_key_1": {
                "english_text": str,
                "target_text": str,
                "audio_normal": AudioSegment or None,
                "audio_slow": AudioSegment or None,
                "image": Image or None,
                "wiktionary_links": str or None
            },
            "phrase_key_2": { ... },
            ...
        }
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    try:
        language = config.TARGET_LANGUAGE_NAME.lower()
        # Get the translated phrases data
        phrases_path = get_translated_phrases_path(collection=collection)
        phrases_data = read_from_gcs(bucket_name, phrases_path, "json")

        if not phrases_data:
            print("No translated phrases found")
            return {}

        # Filter phrases if keys are provided
        if phrase_keys is not None:
            phrases_data = {k: v for k, v in phrases_data.items() if k in phrase_keys}
            if not phrases_data:
                print(f"No matching phrases found for keys: {phrase_keys}")
                return {}

        # Build dictionary of phrase data
        phrase_dict = {}
        for phrase_key, phrase_info in tqdm(
            phrases_data.items(), desc="Building phrase dictionary"
        ):
            # Get multimedia data for this phrase
            multimedia_data = get_phrase_multimedia(
                phrase_key=phrase_key,
                bucket_name=bucket_name,
            )

            if not multimedia_data:
                continue

            # Create entry for this phrase
            phrase_dict[phrase_key] = {
                "english_text": phrase_info["english"],
                "target_text": phrase_info[language],
                "audio_normal": multimedia_data["normal_audio"],
                "audio_slow": multimedia_data["slow_audio"],
                "image": multimedia_data["image"],
                "wiktionary_links": phrase_info.get("wiktionary_links"),
            }

    except Exception as e:
        print(f"Error building phrase dictionary: {str(e)}")
        return {}

    return phrase_dict


def build_phrase_to_story_index(
    collection: str = "LM1000",
    bucket_name: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Build a phrase-to-story index by reading the story collection file and creating an inverse mapping.
    This index maps each phrase key to a list of story names where that phrase appears.

    Args:
        collection: Collection name (default: "LM1000")
        bucket_name: Optional GCS bucket name (defaults to config.GCS_PRIVATE_BUCKET)

    Returns:
        Dict[str, List[str]]: Mapping of phrase keys to lists of story names
    """
    if bucket_name is None:
        bucket_name = config.GCS_PRIVATE_BUCKET

    try:
        # Read the story collection file
        collection_path = get_story_collection_path(collection)
        story_collection = read_from_gcs(bucket_name, collection_path, "json")

        if not story_collection:
            print(f"No stories found in collection {collection}")
            return {}

        # Create inverse mapping
        phrase_to_stories = defaultdict(list)

        # For each story, process its phrases
        for story_name, phrase_info in story_collection.items():
            for phrase in phrase_info:
                # Clean the phrase to create a consistent key
                phrase_key = clean_filename(phrase["phrase"])
                # Add this story to the list for this phrase
                phrase_to_stories[phrase_key].append(story_name)

        # Convert defaultdict to regular dict for JSON serialization
        return dict(phrase_to_stories)

    except Exception as e:
        print(f"Error building phrase-to-story index: {str(e)}")
        return {}
