import os

import requests

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
