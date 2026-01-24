"""Challenge orchestration and formatting for language learning roleplay scenarios."""

import json
from typing import Dict, List, Optional
from string import Template

from src.convert import get_story_title, get_collection_title
from challenges.models import generate_challenges
from src.models import BCP47Language
from src.storage import (
    PRIVATE_BUCKET,
    PUBLIC_BUCKET,
    get_story_challenges_path,
    get_story_translated_challenges_path,
    upload_file_to_gcs,
)
from src.utils import load_template


def generate_roleplay_prompt(
    scenario_data: Dict,
    target_language: BCP47Language,
    complication_index: Optional[int] = None,
) -> str:
    """Generate a structured roleplay prompt for language learning scenarios.

    Args:
        scenario_data: Dictionary containing scenario information including
                      role, context, challenges, and complications
        target_language: BCP47Language object for the target language
        complication_index: Index of the complication to use (None for no complication)

    Returns:
        str: Formatted prompt template for the AI chatbot
    """
    language_name = target_language.display_name()

    complication_text = ""
    if complication_index is not None:
        complication = scenario_data["complications"][complication_index]
        complication_text = f"""
## Complication to Handle
A complication will arise during the roleplay: {complication}
Introduce this naturally at an appropriate moment."""

    return f"""## Personality
You are a {scenario_data["role"]} in {scenario_data["context"]}. You are helpful, patient, and speak naturally in {language_name}. Stay in character throughout the conversation.

## Environment
This is a spoken language learning roleplay session. The learner is practicing {language_name} conversation skills and may make mistakes - be encouraging and supportive.

## Tone
- Speak **only 1-2 sentences** at a time, then pause for the learner's response
- Use natural, everyday {language_name} appropriate for this setting
- If the learner switches to English, gently respond in {language_name} and continue naturally
- Include brief affirmations like "mm-hmm" or "I see" to sound conversational
- Speak clearly and at a moderate pace

## Goal
Help the learner complete this task: **{scenario_data["challenge"]}**

The learner must also discover: **{scenario_data["information_task"]}**
- Don't volunteer this information - wait for them to ask
- Be flexible about how they ask - they don't need perfect grammar
- When they ask appropriately, provide a helpful answer

{complication_text}

## Guardrails
- **Always speak in {language_name}** unless giving feedback
- Keep responses brief - maximum 2 sentences before pausing
- Stay in character as {scenario_data["role"]}
- If you don't understand, ask for clarification in {language_name}
- Be patient with learner mistakes and continue the conversation

## Success Criteria
The roleplay succeeds when: {scenario_data["success_criteria"]}

When this happens, switch to **English** and provide:
- Congratulations on what they accomplished
- The information they discovered (if applicable)
- 2-3 alternative phrases they could have used
- One encouraging comment about their {language_name}

**Start now by greeting the learner in {language_name} as {scenario_data["role"]} - just 1-2 sentences, then wait for their response.**"""


def get_html_challenge_inputs(
    scenario_dicts: Dict,
    target_language: BCP47Language,
) -> List[Dict]:
    """Format scenario dictionaries for HTML template.

    Takes a scenario dictionary of 5 challenges, each with 3 complications, and
    returns a list where each item is a group of related challenges
    (base scenario + complications).

    Args:
        scenario_dicts: Dictionary with "scenarios" key containing list of scenarios
        target_language: BCP47Language object for the target language

    Returns:
        List of challenge group dictionaries, each containing:
        - group_description: HTML formatted description
        - variants: List of challenge variants (base + 3 complications)
    """
    all_challenge_groups = []

    for scenario in scenario_dicts.get("scenarios", []):
        challenge_group = []

        # Base description for all variants (HTML formatted)
        base_description = f"""Setting: {scenario["context"]}<br>
Speaking with: {scenario["role"]}<br>
Task: <span class='font-bold'>{scenario["challenge"]}</span> and find out: <span class='font-bold'>{scenario["information_task"]}</span>"""

        # Add base scenario (no complications)
        challenge_group.append(
            {
                "challenge_description": base_description,
                "llm_prompt": generate_roleplay_prompt(scenario, target_language),
                "answer": "No complications to uncover - ask the AI assistant for the answer",
                "variant": "Base Scenario",
            }
        )

        # Add complication variants
        for i, complication in enumerate(scenario.get("complications", [])):
            challenge_group.append(
                {
                    "challenge_description": base_description,
                    "llm_prompt": generate_roleplay_prompt(
                        scenario, target_language, i
                    ),
                    "answer": f"The complication was: {complication}",
                    "variant": f"Complication {i + 1}",
                }
            )

        all_challenge_groups.append(
            {"group_description": base_description, "variants": challenge_group}
        )

    return all_challenge_groups


def create_html_challenges(
    challenges: List[Dict[str, str]],
    story_name: str,
    language: BCP47Language,
    collection: str = "LM1000",
    bucket_name: str = PUBLIC_BUCKET,
    component_path: str = "ChallengeViewer.js",
    template_path: str = "challenge_template.html",
) -> str:
    """Create and upload a standalone HTML file for language challenges.

    Args:
        challenges: List of challenge group dictionaries from get_html_challenge_inputs()
        story_name: Snake case story name (e.g., "story_coffee_shop_meeting")
        language: BCP47Language object for the target language
        collection: Collection name (e.g., "LM1000")
        bucket_name: GCS bucket for upload (default: PUBLIC_BUCKET)
        component_path: Path to the React component file
        template_path: Path to the HTML template file

    Returns:
        str: GCS URI of the uploaded HTML file
    """
    # Load templates
    react_component = load_template(component_path)
    api_handlers = load_template("ChallengeAPIHandlers.js")
    template = Template(load_template(template_path))

    # Get display values from language
    language_name = language.display_name()

    # Get story title
    title = get_story_title(story_name)

    # Create HTML content
    html_content = template.substitute(
        title=title,
        challenge_data=json.dumps(challenges),
        language=language_name,
        language_code=language.language,
        react_component=f"{api_handlers}\n\n{react_component}",
        collection_name=get_collection_title(collection),
        collection_raw=collection,
    )

    # Get upload path
    output_path = get_story_translated_challenges_path(story_name, language, collection)

    # Upload to GCS
    gcs_uri = upload_file_to_gcs(
        obj=html_content,
        bucket_name=bucket_name,
        file_path=output_path,
        content_type="text/html",
    )

    print(f"HTML challenges created at: {gcs_uri}")
    return gcs_uri


def generate_and_upload_challenges(
    story_name: str,
    story_dialogue: Dict,
    target_language: BCP47Language,
    collection: str = "LM1000",
    private_bucket: str = PRIVATE_BUCKET,
    public_bucket: str = PUBLIC_BUCKET,
) -> str:
    """End-to-end challenge generation: generate scenarios, format, create HTML, upload.

    This orchestrates the complete workflow:
    1. Generate scenarios using llm_tools.challenge_generation
    2. Save scenarios JSON to private bucket
    3. Format scenarios for HTML
    4. Create and upload HTML challenges page to public bucket

    Args:
        story_name: Name of the story (e.g., "story_coffee_shop_meeting")
        story_dialogue: Dictionary with story parts and dialogue
        target_language: BCP47Language object for the target language
        collection: Collection name (default: "LM1000")
        private_bucket: GCS bucket for JSON data (default: PRIVATE_BUCKET)
        public_bucket: GCS bucket for HTML (default: PUBLIC_BUCKET)

    Returns:
        str: GCS URI of uploaded HTML challenges page

    Example:
        >>> import langcodes
        >>> target_lang = langcodes.get("fr-FR")
        >>> story_dialogue = {
        ...     "setup": {"dialogue": [{"speaker": "Alex", "text": "Hello"}]},
        ...     "resolution": {"dialogue": [{"speaker": "Sam", "text": "Goodbye"}]}
        ... }
        >>> uri = generate_and_upload_challenges(
        ...     "story_coffee_shop_meeting",
        ...     story_dialogue,
        ...     target_lang,
        ...     "LM1000"
        ... )
        >>> print(f"Challenges at: {uri}")
    """
    # 1. Generate scenarios using LLM tool
    print(f"Generating challenges for {story_name}...")
    scenario_dicts = generate_challenges(story_dialogue, target_language)

    # 2. Save scenarios JSON to private bucket
    scenarios_path = get_story_challenges_path(story_name, target_language, collection)
    upload_file_to_gcs(
        obj=scenario_dicts,
        bucket_name=private_bucket,
        file_path=scenarios_path,
    )
    print(f"Saved scenarios to: gs://{private_bucket}/{scenarios_path}")

    # 3. Format scenarios for HTML
    challenges = get_html_challenge_inputs(scenario_dicts, target_language)

    # 4. Create and upload HTML
    html_uri = create_html_challenges(
        challenges=challenges,
        story_name=story_name,
        language=target_language,
        collection=collection,
        bucket_name=public_bucket,
    )

    return html_uri
