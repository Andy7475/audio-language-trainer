from anthropic import AnthropicVertex
import json
from dotenv import load_dotenv
import os

load_dotenv()  # so we can use environment variables for various global settings

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME")
ANTHROPIC_REGION = os.getenv("ANTHROPIC_REGION")


def anthropic_generate(prompt: str, max_tokens: int = 1024) -> str:

    client = AnthropicVertex(region=ANTHROPIC_REGION, project_id=PROJECT_ID)

    message = client.messages.create(
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=ANTHROPIC_MODEL_NAME,
    )
    response_json = message.model_dump_json(indent=2)

    response = json.loads(response_json)
    return response["content"][0]["text"]
