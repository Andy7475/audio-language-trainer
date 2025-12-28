from collections import defaultdict
import json
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple

from pydub import AudioSegment
from tqdm import tqdm

  # Keep for backward compatibility with existing functions
from src.convert import (
    get_story_title,
    get_collection_title
)
from src.gcs_storage import (
    check_blob_exists,

    read_from_gcs,
    upload_to_gcs,
    sanitize_path_component,
)
from src.llm_tools.story_generation import generate_story
from src.storage import (
    PRIVATE_BUCKET,
    PUBLIC_BUCKET,
    get_story_dialogue_path,
    upload_file_to_gcs
)
from src.utils import load_template

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, model_validator
from src.models import BCP47Language
from src.phrases.phrase_model import Phrase
from google.cloud.firestore import DocumentReference
from src.connections.gcloud_auth import get_firestore_client
from src.phrases.utils import generate_phrase_hash
from src.phrases.phrase_model import FirePhraseDataModel, Phrase, Translation

class StoryPhrase(Phrase):
    """Phrase model specifically for story dialogues with context-aware translations."""
    story_title_hash: str = Field(..., description="Hash of the parent story")
    story_part: str = Field(..., description="Part of the story (e.g., introduction, development)")
    sequence: int = Field(..., description="Utterance sequence within the story part", gte=0)

    @classmethod
    def create(cls, english_phrase: str, story_title_hash: str, story_part: str, sequence: int, collections:List[str] = []) -> "StoryPhrase":
        """Factory method to create a StoryPhrase with story-specific hash."""
        # Create base phrase for NLP processing
        base_phrase = Phrase.create(english_phrase=english_phrase)
        base_phrase.collections = collections
        # Generate story-specific phrase hash
        story_specific_hash = cls._generate_story_phrase_hash(english_phrase, story_title_hash, story_part, sequence)
        
        return cls(
            **base_phrase.model_dump(exclude={"key", "translations"}),
            key=story_specific_hash,
            story_title_hash=story_title_hash,
            story_part=story_part,
            sequence=sequence
        )
    
    @staticmethod
    def _generate_story_phrase_hash(english_phrase: str, story_title_hash: str, story_part: str, sequence: int) -> str:
        """Generate a unique hash combining phrase text and story context.
        
        Args:
            english_phrase: The English phrase text
            story_title_hash: Hash of the story title
            
        Returns:
            str: Combined hash like "hello_abc123_story_xyz789"
        """
        phrase_hash = generate_phrase_hash(english_phrase)
        # Combine: phrase_hash + "_story_" + story_title_hash
        return f"story_{story_title_hash}#{story_part}#{sequence}#{phrase_hash}"
    
StoryPhrase.model_rebuild()

class Utterance(BaseModel):
    sequence: int = Field(..., description="Sequence number of the dialogue line")
    speaker: str = Field(..., description="Name of the speaker")
    text: str = Field(..., description="English Text spoken by the speaker")
    story_phrase: Optional[StoryPhrase] = Field(None, exclude=True, description="Phrase hash for this dialogue line, uses story part and sequence")
    phrase_hash: Optional[str] = Field(None, description="Phrase hash for this dialogue line")

    @model_validator(mode="after")
    def assign_phrase_hash(self) -> "Utterance":
        """Assign phrase_hash from story_phrase if available."""
        if self.story_phrase:
            self.phrase_hash = self.story_phrase.key
        return self
    
class Story(FirePhraseDataModel):
    title: str = Field(..., description="Title of the story")
    summary: str = Field("", description="Brief summary of the story")
    story_parts: Dict[str, List[Utterance]] = Field(..., description="Dictionary of story parts (e.g., introduction, development)")
    collections: List[str] = Field(default_factory=list, description="List of collections this story belongs to")
   
    @classmethod
    def create(cls, title: str, summary: str, story_dialogue: Dict[str, List[Utterance]]) -> "Story":
        """Factory method to create a Story instance."""
        story_title_hash = generate_phrase_hash(title)

        story_parts = defaultdict(list)
        # create sequence and story part information for each piece of dialogue
        for part_name, part_dialogue in story_dialogue.items():
            story_parts[part_name] = [] # will be a list of utterances

            for seq, utterance in enumerate(part_dialogue["dialogue"]):

                story_phrase_obj = StoryPhrase.create(
                    english_phrase=utterance["text"],
                    story_title_hash=story_title_hash,
                    story_part=part_name,
                    sequence=seq,
                    collections=["stories", title],)
                
                utterance_obj = Utterance(
                    sequence=seq,
                    speaker=utterance["speaker"],
                    text=utterance["text"],
                    story_title_hash=story_title_hash,   
                    story_phrase=story_phrase_obj                
                )
                story_parts[part_name].append(utterance_obj)

        return cls(title=title, key=story_title_hash, summary=summary, story_parts=story_parts, firestore_collection="stories")

    def upload_phrase_entries(self, language:BCP47Language|None = None, overwrite:bool = False) -> List[DocumentReference]:
        """Upload all StoryPhrase entries to Firestore."""
        doc_refs = []
        for part_name, utterances in self.story_parts.items():
            for utterance in utterances:
                if utterance.story_phrase:
                    doc_ref = utterance.story_phrase.upload(language=language, overwrite=overwrite)
                    doc_refs.append(doc_ref)
        return doc_refs
    
    def upload(self) -> DocumentReference:
        """Upload the story to Firestore."""
        doc_ref = self._get_firestore_document_reference()
        doc_ref.set(self.model_dump(mode="json"))
        return doc_ref

def upload_styles_to_gcs():
    """Upload the styles.css file to the public GCS bucket."""

    # Load the CSS content with correct path handling
    try:
        # Try current directory first (when running from project root)
        styles_content = load_template("styles.css", "src/templates")
    except FileNotFoundError:
        # Fallback to relative path (when running from subdirectory)
        styles_content = load_template("styles.css", "../src/templates")

    # Upload to GCS
    public_url = upload_to_gcs(
        obj=styles_content,
        bucket_name=PUBLIC_BUCKET,
        file_name="styles.css",
        content_type="text/css",
    )

    print("✅ Styles uploaded successfully!")
    print(f"🌐 Public URL: {public_url}")

    return public_url





# ============================================================================
# NEW STORY GENERATION FUNCTIONS (using llm_tools pattern)
# ============================================================================


def generate_and_upload_story(
    verbs: List[str],
    vocab: List[str],
    collection: str = "LM1000",
    bucket_name: str = PRIVATE_BUCKET,
) -> Tuple[str, Dict, str]:
    """Generate English story and upload to GCS using modern llm_tools pattern.

    Stories are ALWAYS generated in English. Translation happens separately
    using existing translation functions.

    This is the recommended way to generate new stories. It uses:
    - src.llm_tools.story_generation for LLM calls
    - src.storage for path generation and uploads
    - Dynamic story structure based on verb count

    Args:
        verbs: List of verbs to incorporate in the story
        vocab: List of other vocabulary words to use
        collection: Collection name (default: "LM1000")
        bucket_name: GCS bucket for upload (default: PRIVATE_BUCKET)

    Returns:
        Tuple of (story_name, story_dialogue, gcs_uri)
        - story_name: 3-word story title
        - story_dialogue: Dictionary with story parts and dialogue
        - gcs_uri: GCS URI where dialogue JSON was uploaded

    Example:
        >>> story_name, dialogue, uri = generate_and_upload_story(
        ...     verbs=["go", "see", "want"],
        ...     vocab=["coffee", "table", "friend"],
        ...     collection="LM1000"
        ... )
        >>> print(f"Created story: {story_name}")
        >>> print(f"Saved to: {uri}")
    """
    # Generate story using LLM tool (always in English)
    print(f"Generating story from {len(verbs)} verbs and {len(vocab)} vocab words...")
    story_name, story_dialogue = generate_story(verbs, vocab)

    print(f"Generated story: {story_name}")

    # Upload dialogue to GCS
    dialogue_path = get_story_dialogue_path(story_name, collection)
    gcs_uri = upload_file_to_gcs(
        obj=story_dialogue,
        bucket_name=bucket_name,
        file_path=dialogue_path,
    )

    print(f"Uploaded story dialogue to: {gcs_uri}")

    return story_name, story_dialogue, gcs_uri
