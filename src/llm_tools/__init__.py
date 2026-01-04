"""LLM tools and prompts for translation refinement and phrase generation."""

from .base import DEFAULT_MODEL
from .challenge_generation import generate_challenges
from .review_translation import refine_translation
from .refine_story_translation import refine_story_translation
from .story_generation import generate_story
from .verb_phrase_generation import generate_verb_phrases
from .vocab_phrase_generation import generate_vocab_phrases

__all__ = [
    "DEFAULT_MODEL",
    "generate_challenges",
    "refine_translation",
    "refine_story_translation",
    "generate_story",
    "generate_verb_phrases",
    "generate_vocab_phrases",
]
