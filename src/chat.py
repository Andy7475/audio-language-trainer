import json
from pathlib import Path
from string import Template
from typing import Dict, List, Tuple
from src.convert import get_story_title
from src.anki_tools import load_template
from src.config_loader import config


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
    complication_instruction = ""

    if complication_index is not None:
        complication_text = f"""[COMPLICATION TO HANDLE]
The roleplay has a complication for the learner.
You should explain: {scenario_data['complications'][complication_index]}"""
        complication_instruction = (
            "3. If there is a complication, introduce it at the appropriate moment"
        )

    return f"""You are a helpful {config.TARGET_LANGUAGE_NAME} language learning assistant. You will engage in roleplay scenarios to help users practice their {config.TARGET_LANGUAGE_NAME} conversation skills. Here are the details for this challenge:

[ROLEPLAY SCENARIO]
You are playing the role of: {scenario_data['role']}
Context: {scenario_data['context']}

[LEARNER'S TASK]
The learner needs to: {scenario_data['challenge']}
They must also find out (FIND_OUT): {scenario_data['information_task']}
DO NOT provide them with this unless they ask. But be flexible on how they ask - they don't have to be perfect.

[CORRECT INFORMATION TO PROVIDE]
When the learner asks appropriately, you should create a suitable answer to what they are trying to find out (FIND_OUT) - remembering this answer to reveal at the end of the role-play.

{complication_text}

[SCENARIO FLOW]
1. You should begin by greeting the learner in character
2. You should respond naturally to the learner, in character, allow the learner to extend or expand the roleplay - the aim is for them to practice!
{complication_instruction}

[CHALLENGE: SUCCESS CONDITIONS]
{scenario_data['success_criteria']}

ROLEPLAY GUIDELINES:
1. Begin each interaction in {config.TARGET_LANGUAGE_NAME} staying in character for the scenario
2. If the learner says "PAUSE ROLEPLAY", temporarily break character to:
   - Provide relevant phrases or vocabulary in English
   - Explain the current expectation
   - Then resume the roleplay in {config.TARGET_LANGUAGE_NAME}
3. Let the learner extend the roleplay if they want, you don't have to stick rigidly to the learner's task.
4. Use simple, clear {config.TARGET_LANGUAGE_NAME} appropriate for the learner's level
5. Stay in {config.TARGET_LANGUAGE_NAME} until the success conditions are met, or the learner says 'END SESSION'.
6. Once success conditions are met, provide constructive feedback in British English about:
   - Successful language usage
   - Reveal the answer to what they had to find out (FIND_OUT)
   - Areas for improvement
   - Alternative phrases they could have used

Start by introducing yourself in character, in {config.TARGET_LANGUAGE_NAME} appropriate to the challenge context and specified role."""


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
    output_dir: Path,
    story_name: str,
    component_path: str = "ChallengeViewer.js",
    template_path: str = "challenge_template.html",
) -> str:
    """
    Create a standalone HTML file for language challenges using string.Template.

    Args:
        challenges: List of dictionaries, each containing:
            - challenge_description: Description of the challenge
            - llm_prompt: Prompt to send to the LLM (hidden from user)
            - answer: The answer to reveal
        output_dir: Directory where the HTML file will be saved
        story_name: snakecase of the form story_title_of_story
        language: Target language name
        component_path: Path to the React component file
        template_path: Path to the HTML template file

    Returns:
        The HTML content as a string
    """
    # Load the React component
    react_component = load_template(component_path)
    template = Template(load_template(template_path))
    title = get_story_title(story_name)
    # Create the HTML content
    html_content = template.substitute(
        title=title,
        challenge_data=json.dumps(challenges),
        language=config.TARGET_LANGUAGE_NAME,
        react_component=react_component,
    )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = "challenges.html"
    # Create the HTML file
    output_path = output_dir / output_file
    output_path.write_text(html_content, encoding="utf-8")

    print(f"HTML challenges created at: {output_path}")
    return html_content
