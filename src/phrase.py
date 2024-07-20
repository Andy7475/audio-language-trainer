import os
import requests
from src.dialogue_generation import anthropic_generate, extract_json_from_llm_response
from typing import List
from src.config_loader import config

sapling_api_key = os.getenv("SAPLING_API_KEY")


def correct_grammar(phrase: str) -> str:
    """Uses the Sapling AI endpoint to correct any grammatical errors (in English), this should change things like
    she love -> she loves, so that the translation endpoint works better"""
    try:
        response = requests.post(
            "https://api.sapling.ai/api/v1/edits",
            json={
                "key": sapling_api_key,
                "text": f"{phrase}",
                "session_id": "language_graph",
                "auto_apply": True,
            },
        )
        if response.ok:
            resp_json = response.json()
            return resp_json["applied_text"]
        else:
            raise ConnectionError(
                f"Could not connect to Sapling AI {response}. Content {response.content}"
            )
    except Exception as e:
        print(f"Error querying Sapling AI endpoint with phrase {phrase}: ", e)
        return None


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
