from llm_tools.base import load_prompt_template
from llm_tools.challenge_generation import generate_challenge_content
from story import Story


from typing import Dict


def generate_challenges(
    story: Story,
) -> Dict:
        # Load prompt templates
    system_template = load_prompt_template("challenge_generation", "system")
    user_template = load_prompt_template("challenge_generation", "user")

    # Substitute variables
    system_prompt = system_template.substitute(
    )
    user_prompt = user_template.substitute(
        story_content=story.get_story_text(),
    )

    challenge = generate_challenge_content(system_prompt, user_prompt)
    return challenge