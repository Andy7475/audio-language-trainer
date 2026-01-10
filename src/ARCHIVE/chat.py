import json
from string import Template
from typing import Dict, List
from src.convert import get_story_title, get_collection_title
from src.anki_tools import load_template
from src.config_loader import config
from src.gcs_storage import get_story_translated_challenges_path
from storage import upload_to_gcs


def get_challenge_generation_prompt(story_dialogue: dict) -> str:
    """
    Generate a prompt for an LLM to create language learning scenarios where learners
    must complete a challenge while gathering specific information and potentially
    handling complications.

    These scenarios will be fed into another prompt generation script to reformat them
    ready for an online chat application (generate_roleplay_prompt())

    Args:
        story_dialogue: Dictionary containing dialogue parts ("setup", "resolution" etc)

    Returns:
        str: Formatted prompt for the LLM ( 5 x scenario data as scenario_dict)
    """

    dialogue_text = []
    for part in story_dialogue.values():
        if "dialogue" in part:
            for utterance in part["dialogue"]:
                dialogue_text.append(f'{utterance["speaker"]}: {utterance["text"]}')

    story_context = "\n".join(dialogue_text)

    prompt = f"""Analyze this dialogue and create 5 practical language learning scenarios.
Each scenario should have a main challenge plus information the learner must discover,
with possible complications that could arise.
Base the scenarios on this story context:

{story_context}

Create the scenarios in this JSON format:
{{
    "scenarios": [
        {{
            "role": "who the teacher will roleplay (e.g. coffee shop staff)",
            "context": "brief setting description",
            "challenge": "overall task to complete (e.g. 'Order a coffee')",
            "information_task": "specific information to discover (e.g. 'What is the price of a coffee?')",
            "complications": [
                "I'm sorry, our cups are all being washed right now, but I could put it in a mug?",
                "Unfortunately our coffee machine is just warming up, if you are happy waiting and I'll bring you the order when ready?",
                "We've run out of coffee cups, but we can serve you in a takeaway cup"
            ],
            "success_criteria": "Learner discovers the price of their chosen size while handling any complications with the order"
        }}
    ]
}}

Requirements:
1. Create 5 scenarios that relate to the story provided using similar vocabulary
2. Each scenario should have:
   - A main challenge (e.g. "order a coffee")
   - An information-seeking task about the variable (e.g. "what is the price of a coffee?")
   - Three realistic complications to completing the challenge
4. Information tasks should:
   - Be discoverable through conversation
5. Complications should:
   - Clearly explain why the original request can't be fulfilled
   - Suggest or hint at possible alternatives
   - Not prevent the information task completion
6. Success criteria should:
   - Include both completing the challenge and discovering the information
   - Allow for multiple solutions
   - Include accepting alternatives

Remember:
- Keep language practical and everyday
- Every complication should suggest a way forward
- Success means both understanding any problems and gathering required information

Output only the JSON with no additional text."""

    return prompt


def generate_roleplay_prompt(
    scenario_data: dict, complication_index: int = None
) -> str:
    """
    Generate a structured roleplay prompt for language learning scenarios.

    Args:
        scenario_data (dict): Dictionary containing scenario information including
                            role, context, challenges, and complications.
        complication_index (int): Index of the complication to use (defaults to None)

    Returns:
        str: Formatted prompt template with placeholders for dynamic content
    """
    complication_text = ""

    if complication_index is not None:
        complication_text = f"""
## Complication to Handle
A complication will arise during the roleplay: {scenario_data['complications'][complication_index]}
Introduce this naturally at an appropriate moment."""

    return f"""## Personality
You are a {scenario_data['role']} in {scenario_data['context']}. You are helpful, patient, and speak naturally in {config.TARGET_LANGUAGE_NAME}. Stay in character throughout the conversation.

## Environment
This is a spoken language learning roleplay session. The learner is practicing {config.TARGET_LANGUAGE_NAME} conversation skills and may make mistakes - be encouraging and supportive.

## Tone
- Speak **only 1-2 sentences** at a time, then pause for the learner's response
- Use natural, everyday {config.TARGET_LANGUAGE_NAME} appropriate for this setting
- If the learner switches to English, gently respond in {config.TARGET_LANGUAGE_NAME} and continue naturally
- Include brief affirmations like "mm-hmm" or "I see" to sound conversational
- Speak clearly and at a moderate pace

## Goal
Help the learner complete this task: **{scenario_data['challenge']}**

The learner must also discover: **{scenario_data['information_task']}**
- Don't volunteer this information - wait for them to ask
- Be flexible about how they ask - they don't need perfect grammar
- When they ask appropriately, provide a helpful answer

{complication_text}

## Guardrails
- **Always speak in {config.TARGET_LANGUAGE_NAME}** unless giving feedback
- Keep responses brief - maximum 2 sentences before pausing
- Stay in character as {scenario_data['role']}
- If you don't understand, ask for clarification in {config.TARGET_LANGUAGE_NAME}
- Be patient with learner mistakes and continue the conversation

## Success Criteria
The roleplay succeeds when: {scenario_data['success_criteria']}

When this happens, switch to **English** and provide:
- Congratulations on what they accomplished
- The information they discovered (if applicable)
- 2-3 alternative phrases they could have used
- One encouraging comment about their {config.TARGET_LANGUAGE_NAME}

**Start now by greeting the learner in {config.TARGET_LANGUAGE_NAME} as {scenario_data['role']} - just 1-2 sentences, then wait for their response.**"""


def process_scenario(scenario: dict) -> list:
    """Turns a single scenario dictionary into a list of challenge dictionaries
    that can be added into a list and fed into create_html_challenges

    Args:
        scenario (dict): A dictionary scenario originating from the output of
        get_challenge_generation_prompt()
    Raises:
        KeyError: This means the LLM has not responded correctly to the prompt

    Returns:
        list: Challenge dictionaries with keys challenge_description, llm_prompt, answer
    """
    all_challenges = []
    context = scenario.get("context")
    role = scenario.get("role")
    challenge = scenario.get("challenge")
    find_out = scenario.get("information_task")
    complications = scenario.get("complications")
    if not all([context, role, challenge, find_out, complications]):
        raise KeyError(
            f"Missing keys in scenario. Should have 'context', 'role', 'challenge', 'scenario', but has {scenario.keys()}"
        )
    challenge_description = f"Setting: {context}, speaking with {role}.\nYou must {challenge} and find out: {find_out}"

    # no complication variant

    all_challenges.append(
        {
            "challenge_description": challenge_description,
            "llm_prompt": generate_roleplay_prompt(scenario),
            "answer": "No complications to uncover - ask the AI assitant for the answer",
        }
    )

    for i, variant in enumerate(complications):
        answer = f"The complication was: {variant}"
        llm_prompt = generate_roleplay_prompt(scenario, i)
        all_challenges.append(
            {
                "challenge_description": challenge_description,
                "llm_prompt": llm_prompt,
                "answer": answer,
            }
        )

    return all_challenges


def get_html_challenge_inputs(scenario_dicts: dict) -> list:
    """Takes a scenario dictionary of 5 challenges, each with 3 complications, and returns a list
    where each item is a group of related challenges (base scenario + complications)
    """
    all_challenge_groups = []
    for scenario in scenario_dicts.get("scenarios"):
        challenge_group = []

        # Base description for all variants
        base_description = f"""Setting: {scenario['context']}<br>
Speaking with: {scenario['role']}<br>
Task: <span class='font-bold'>{scenario['challenge']}</span> and find out: <span class='font-bold'>{scenario['information_task']}</span>"""

        # Add base scenario (no complications)
        challenge_group.append(
            {
                "challenge_description": base_description,
                "llm_prompt": generate_roleplay_prompt(scenario),
                "answer": "No complications to uncover - ask the AI assistant for the answer",
                "variant": "Base Scenario",
            }
        )

        # Add complication variants
        for i, complication in enumerate(scenario.get("complications", [])):
            challenge_group.append(
                {
                    "challenge_description": base_description,
                    "llm_prompt": generate_roleplay_prompt(scenario, i),
                    "answer": f"The complication was: {complication}",
                    "variant": f"Complication {i+1}",
                }
            )

        all_challenge_groups.append(
            {"group_description": base_description, "variants": challenge_group}
        )

    return all_challenge_groups


def create_html_challenges(
    challenges: List[Dict[str, str]],
    story_name: str,
    component_path: str = "ChallengeViewer.js",
    template_path: str = "challenge_template.html",
    collection: str = "LM1000",
    language_name: str = None,
    language_code: str = None,
) -> str:
    """
    Create a standalone HTML file for language challenges using string.Template.

    Args:
        challenges: List of dictionaries, each containing:
            - challenge_description: Description of the challenge
            - llm_prompt: Prompt to send to the LLM (hidden from user)
            - answer: The answer to reveal
        output_path: Path to the HTML file
        story_name: snakecase of the form story_title_of_story
        language: Target language name
        component_path: Path to the React component file
        template_path: Path to the HTML template file
        collection: Collection name for organizing stories

    Returns:
        The GCS URI of the uploaded HTML file
    """
    # Load the React component and API handlers
    react_component = load_template(component_path)
    api_handlers = load_template("ChallengeAPIHandlers.js")
    template = Template(load_template(template_path))
    title = get_story_title(story_name)

    # Use provided language name/code or fall back to config
    if language_name is None:
        language_name = config.TARGET_LANGUAGE_NAME

    if language_code is None:
        language_code = config.TARGET_LANGUAGE_CODE

    print(f"Using language code: {language_code} for language: {language_name}")
    # Create the HTML content with both JS modules
    html_content = template.substitute(
        title=title,
        challenge_data=json.dumps(challenges),
        language=language_name,
        language_code=language_code,
        react_component=f"{api_handlers}\n\n{react_component}",
        collection_name=get_collection_title(collection),
        collection_raw=collection,
    )

    # Create output directory if it doesn't exist
    output_path = get_story_translated_challenges_path(
        story_name, collection=collection
    )

    gcs_uri = upload_to_gcs(html_content, config.GCS_PUBLIC_BUCKET, output_path)

    print(f"HTML challenges created at: {gcs_uri}")
    return gcs_uri
