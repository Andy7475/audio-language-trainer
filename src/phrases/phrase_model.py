from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Literal, Annotated

from pydantic import BaseModel, Field, PlainSerializer, BeforeValidator

from src.connections.gcloud_auth import get_firestore_client
from src.phrases.utils import generate_phrase_hash
from google.cloud.firestore import Client as FirestoreClient
from src.nlp import extract_lemmas_and_pos, get_tokens_from_lemmas_and_pos, get_verbs_from_lemmas_and_pos, get_vocab_from_lemmas_and_pos
from src.models import BCP47Language

LowercaseStr = Annotated[str, BeforeValidator(lambda v: v.lower().strip() )]

class Phrase(BaseModel):
    """Pydantic model representing a phrase in Firestore.

    This model corresponds to the phrases collection schema defined in firestore.md.
    Each phrase contains English text with linguistic analysis including tokens, lemmas,
    verbs, and vocabulary.
    """
    phrase_hash: str = Field(..., description="Unique hash identifier for the phrase")
    english: str = Field(..., description="Original English phrase with original capitalisation")
    english_lower: LowercaseStr = Field(..., description="Lowercase version for consistent lookups")
    tokens: List[str] = Field(..., description="Tokenised words from the phrase")
    verbs: List[str] = Field(default_factory=list, description="Lemmatised verb forms only")
    vocab: List[str] = Field(default_factory=list, description="Lemmatised non-verb words")
    source: Optional[Literal["manual", "tatoeba", "generated"]] = Field(
        default=None, description="Source of the phrase"
    )
    translations: List[Translation] = Field(default_factory=list, description="List of translations for this phrase")

    @classmethod
    def create_phrase(cls, english_phrase:str, source:str="generated")->Phrase:
        """Factory method to create a Phrase with NLP processing."""

        phrase_hash = generate_phrase_hash(english_phrase)
        lemmas_and_pos = extract_lemmas_and_pos([english_phrase], language_code="en")
        tokens=get_tokens_from_lemmas_and_pos(lemmas_and_pos)

        #we make an english 'translation' for audio / text lookup consistency
        english_translation = Translation(phrase_hash=phrase_hash,
                                          language= BCP47Language.get("en-GB"),
                                         text= english_phrase,
                                         text_lower= english_phrase.lower(),
                                         tokens=tokens,
                                         audio=[]
                                         )
        return cls(
            phrase_hash=phrase_hash,
            english=english_phrase,
            english_lower=english_phrase,
            tokens=tokens,
            verbs=get_verbs_from_lemmas_and_pos(lemmas_and_pos),
            vocab=get_vocab_from_lemmas_and_pos(lemmas_and_pos),
            source=source,
            translations=[english_translation]
        )

    def get_phrase_hash(self) -> str:
        """Generate a unique hash for this phrase based on English text.

        Returns:
            str: Phrase hash in format: {slug}_{hash_suffix}

        Example:
            >>> phrase = Phrase(english="She runs to the store daily", ...)
            >>> phrase.get_phrase_hash()
            'she_runs_to_the_store_daily_a3f8d2'
        """
        return generate_phrase_hash(self.english)


    def upload_phrase(self, firestore_client: FirestoreClient, database_name: str = "firephrases") -> str:
        """Upload a phrase to Firestore.

        Args:
            phrase: The Phrase object to upload
            database_name: Name of the Firestore database (default: "firephrases")

        Returns:
            str: The phrase hash (document ID) of the uploaded phrase

        Raises:
            RuntimeError: If upload fails
        """
        try:
            phrase_hash = self.get_phrase_hash()

            doc_ref = firestore_client.collection("phrases").document(phrase_hash)
            doc_ref.set(self.model_dump(mode="json"))

            return phrase_hash

        except Exception as e:
            raise RuntimeError(f"Failed to upload phrase: {e}")



def get_phrase(phrase_hash: str, database_name: str = "firephrases") -> Optional[Phrase]:
    """Fetch a phrase from Firestore by its hash.

    Args:
        phrase_hash: The phrase hash (document ID)
        database_name: Name of the Firestore database (default: "firephrases")

    Returns:
        Phrase: The phrase object if found, None otherwise

    Raises:
        RuntimeError: If Firestore query fails
    """
    try:
        client = get_firestore_client(database_name)
        doc_ref = client.collection("phrases").document(phrase_hash)
        doc = doc_ref.get()

        if doc.exists:
            return Phrase.model_validate(doc.to_dict())
        return None

    except Exception as e:
        raise RuntimeError(f"Failed to get phrase {phrase_hash}: {e}")





class PhraseAudio(BaseModel):
    """Pydantic model representing audio associated with a phrase."""
    setting: Literal["flashcard", "story"] = Field(..., description="Context where audio is used")
    speed: Literal["normal", "slow", "fast"] = Field(..., description="Speed of the audio")
    url: str = Field(..., description="URL to the audio file ignoring bucket name")
    voice_model_id: str = Field(..., description="Identifier of the voice model used")
    voice_provider: Literal["google", "aws", "azure"] = Field(..., description="Cloud provider for TTS")
    duration_seconds: float | None = Field(default=None, description="Duration of the audio in seconds")

class Translation(BaseModel):
    """Pydantic model representing a translation of a phrase in Firestore."""
    model_config = {"arbitrary_types_allowed": True} # allow us to use Language
    phrase_hash: str = Field(..., description="Hash of the associated English root phrase")
    language: BCP47Language = Field(..., description="BCP-47 language tag for the translation")
    text :str = Field(..., description="Translated text of the phrase")
    text_lower: LowercaseStr = Field(..., description="Lowercase version for consistent lookups")
    tokens: List[str] = Field(..., description="Tokenised words from the translated phrase")
    audio: List[PhraseAudio]

