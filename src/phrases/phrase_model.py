from datetime import datetime
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator

from src.connections.gcloud_auth import get_firestore_client
from src.phrases.utils import generate_phrase_hash
from google.cloud.firestore import Client as FirestoreClient

class Phrase(BaseModel):
    """Pydantic model representing a phrase in Firestore.

    This model corresponds to the phrases collection schema defined in firestore.md.
    Each phrase contains English text with linguistic analysis including tokens, lemmas,
    verbs, and vocabulary.
    """

    english: str = Field(..., description="Original English phrase with original capitalisation")
    english_lower: str = Field(..., description="Lowercase version for consistent lookups")
    tokens: List[str] = Field(..., description="Tokenised words from the phrase")
    lemmas: List[str] = Field(..., description="Lemmatised forms of all tokens")
    verbs: List[str] = Field(default_factory=list, description="Lemmatised verb forms only")
    vocab: List[str] = Field(default_factory=list, description="Lemmatised non-verb words")
    created: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    modified: Optional[datetime] = Field(None, description="Last modification timestamp")
    source: Optional[Literal["manual", "tatoeba", "generated"]] = Field(
        None, description="Source of the phrase"
    )

    @field_validator("english_lower", mode="before")
    @classmethod
    def ensure_lowercase(cls, v: str, info) -> str:
        """Ensure english_lower is actually lowercase."""
        if v and not v.islower():
            # If english_lower is not lowercase, create it from english field
            english = info.data.get("english", "")
            return english.lower()
        return v

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

            self.modified = datetime.utcnow()

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


