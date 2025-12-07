"""LLM tools and prompts for translation refinement and phrase generation."""

from src.llm_tools.base import DEFAULT_MODEL
from src.llm_tools.challenge_generation import generate_challenges
from src.llm_tools.review_translation import refine_translation
from src.llm_tools.review_story_translations import review_story_dialogue
from src.llm_tools.story_generation import generate_story
from src.llm_tools.verb_phrase_generation import generate_verb_phrases
from src.llm_tools.vocab_phrase_generation import generate_vocab_phrases

__all__ = [
    "DEFAULT_MODEL",
    "generate_challenges",
    "refine_translation",
    "review_story_dialogue",
    "generate_story",
    "generate_verb_phrases",
    "generate_vocab_phrases",
]
