from __future__ import annotations

from typing import List, Optional, Literal, Annotated, Dict

from pydantic import BaseModel, Field, BeforeValidator

from src.connections.gcloud_auth import get_firestore_client
from src.phrases.utils import generate_phrase_hash
from google.cloud.firestore import Client as FirestoreClient
from src.nlp import extract_lemmas_and_pos, get_tokens_from_lemmas_and_pos, get_verbs_from_lemmas_and_pos, get_vocab_from_lemmas_and_pos, get_text_tokens
from src.models import BCP47Language
from src.translation import translate_with_google_translate, refine_translation_with_anthropic

LowercaseStr = Annotated[str, BeforeValidator(lambda v: v.lower().strip() )]

# Type aliases for audio structure
AudioSpeeds = Literal["slow", "normal", "fast"]
AudioSettings = Literal["flashcard", "story"]

class Phrase(BaseModel):
    """Pydantic model representing a phrase in Firestore.

    This model corresponds to the phrases collection schema defined in firestore.md.
    Each phrase contains English text with linguistic analysis including tokens, lemmas,
    verbs, and vocabulary. Translations are stored in a subcollection and fetched separately.
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
    translations: List[Translation] = Field(default_factory=list, description="List of translations for this phrase (loaded separately from subcollection)")

    @classmethod
    def create_phrase(cls, english_phrase: str, source: str = "generated") -> Phrase:
        """Factory method to create a Phrase with NLP processing.

        Args:
            english_phrase: The English phrase text
            source: Source of the phrase ("manual", "tatoeba", or "generated")

        Returns:
            Phrase: A new Phrase object with NLP analysis
        """
        phrase_hash = generate_phrase_hash(english_phrase)
        lemmas_and_pos = extract_lemmas_and_pos([english_phrase], language_code="en")
        tokens = get_tokens_from_lemmas_and_pos(lemmas_and_pos)

        return cls(
            phrase_hash=phrase_hash,
            english=english_phrase,
            english_lower=english_phrase,
            tokens=tokens,
            verbs=get_verbs_from_lemmas_and_pos(lemmas_and_pos),
            vocab=get_vocab_from_lemmas_and_pos(lemmas_and_pos),
            source=source,
            translations=[]  # Translations are loaded separately from subcollection
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

    def _has_translation(self, language: BCP47Language) -> bool:
        """Check if a translation exists for the given language.

        Args:
            language: BCP47 language tag to check for

        Returns:
            bool: True if translation exists, False otherwise
        """
        target_tag = language.to_tag()
        return any(t.language.to_tag() == target_tag for t in self.translations)

    def _get_translation(self, language: BCP47Language) -> Optional[Translation]:
        """Get the translation for a specific language if it exists.

        Args:
            language: BCP47 language tag to retrieve

        Returns:
            Optional[Translation]: The translation if found, None otherwise
        """
        target_tag = language.to_tag()
        return next(
            (t for t in self.translations if t.language.to_tag() == target_tag),
            None
        )

    def translate(
        self,
        target_language: BCP47Language,
        refine: bool = True,
        model: Optional[str] = None
    ) -> Translation:
        """Translate the phrase to a target language.

        This method will:
        1. Check if translation already exists and return it if found
        2. Translate using Google Translate API
        3. Optionally refine the translation using Anthropic's Claude
        4. Tokenize the translated text
        5. Create and return a Translation object
        6. Add the translation to the phrase's translations list

        Args:
            target_language: BCP47 language tag for the target language
            refine: Whether to refine the translation with Claude (default: True)
            model: Anthropic model to use for refinement (defaults to claude-sonnet-4-20250514)

        Returns:
            Translation: The translation object

        Raises:
            RuntimeError: If translation fails

        Example:
            >>> phrase = Phrase.create_phrase("Hello, how are you?")
            >>> fr_translation = phrase.translate(BCP47Language.get("fr-FR"))
            >>> print(fr_translation.text)
            'Bonjour, comment allez-vous ?'
        """
        # Check if translation already exists
        existing_translation = self._get_translation(target_language)
        if existing_translation is not None:
            print(f"Translation for {target_language.to_tag()} already exists")
            return existing_translation

        # Step 1: Translate with Google Translate
        translated_text = translate_with_google_translate(
            text=self.english,
            target_language=target_language,
            source_language="en"
        )

        # Step 2: Refine with Claude if requested
        if refine:
            translated_text = refine_translation_with_anthropic(
                source_phrase=self.english,
                initial_translation=translated_text,
                target_language=target_language,
                source_language=None,
                model=model
            )

        # Step 3: Tokenize the translated text
        # Extract language code for tokenization (e.g., "fr" from "fr-FR")
        language_code = target_language.language
        tokens = get_text_tokens(translated_text, language_code=language_code)

        # Step 4: Create the Translation object
        translation = Translation(
            phrase_hash=self.phrase_hash,
            language=target_language,
            text=translated_text,
            text_lower=translated_text.lower(),
            tokens=tokens,
            audio=None  # Audio is added separately when TTS is generated
        )

        # Step 5: Add the translation to the phrase's translations list
        self.translations.append(translation)

        return translation

    def upload_phrase(self, firestore_client: FirestoreClient, database_name: str = "firephrases") -> str:
        """Upload a phrase and its translations to Firestore.

        Uploads the phrase document to `phrases/{phrase_hash}` and each translation to
        `phrases/{phrase_hash}/translations/{language_code}` subcollection.

        Args:
            firestore_client: Firestore client instance
            database_name: Name of the Firestore database (default: "firephrases")

        Returns:
            str: The phrase hash (document ID) of the uploaded phrase

        Raises:
            RuntimeError: If upload fails
        """
        try:
            phrase_hash = self.get_phrase_hash()

            # Upload phrase document (without translations)
            phrase_data = self.model_dump(mode="json", exclude={"translations"})
            doc_ref = firestore_client.collection("phrases").document(phrase_hash)
            doc_ref.set(phrase_data)

            # Upload each translation to subcollection
            for translation in self.translations:
                language_tag = translation.language.to_tag()
                translation_data = translation.model_dump(mode="json")
                translations_ref = doc_ref.collection("translations").document(language_tag)
                translations_ref.set(translation_data)

            return phrase_hash

        except Exception as e:
            raise RuntimeError(f"Failed to upload phrase: {e}")



def get_phrase(phrase_hash: str, database_name: str = "firephrases") -> Optional[Phrase]:
    """Fetch a phrase from Firestore by its hash, including all translations.

    Fetches the phrase document from `phrases/{phrase_hash}` and all translations from
    the `phrases/{phrase_hash}/translations` subcollection.

    Args:
        phrase_hash: The phrase hash (document ID)
        database_name: Name of the Firestore database (default: "firephrases")

    Returns:
        Optional[Phrase]: The phrase object with translations if found, None otherwise

    Raises:
        RuntimeError: If Firestore query fails
    """
    try:
        client = get_firestore_client(database_name)
        doc_ref = client.collection("phrases").document(phrase_hash)
        doc = doc_ref.get()

        if not doc.exists:
            return None

        # Get phrase data
        phrase_data = doc.to_dict()

        # Fetch all translations from subcollection
        translations_docs = doc_ref.collection("translations").stream()
        translations = []
        for trans_doc in translations_docs:
            trans_data = trans_doc.to_dict()
            translations.append(Translation.model_validate(trans_data))

        # Add translations to phrase data
        phrase_data["translations"] = translations

        return Phrase.model_validate(phrase_data)

    except Exception as e:
        raise RuntimeError(f"Failed to get phrase {phrase_hash}: {e}")


def get_phrase_by_english(english_phrase: str, database_name: str = "firephrases") -> Optional[Phrase]:
    """Fetch a phrase from Firestore using its English text.

    Convenience wrapper that generates the phrase hash from English text and fetches
    the corresponding phrase document with all translations.

    Args:
        english_phrase: The English phrase text (e.g., "She runs to the store daily")
        database_name: Name of the Firestore database (default: "firephrases")

    Returns:
        Optional[Phrase]: The phrase object with translations if found, None otherwise

    Raises:
        RuntimeError: If Firestore query fails
    """
    phrase_hash = generate_phrase_hash(english_phrase)
    return get_phrase(phrase_hash, database_name=database_name)




class PhraseAudio(BaseModel):
    """Pydantic model representing audio metadata for a specific setting/speed combination.

    This represents a single audio file with its metadata (URL, voice, duration, etc).
    """
    url: str = Field(..., description="GCS URL to the audio file")
    voice_model_id: str = Field(..., description="Identifier of the voice model used")
    voice_provider: Literal["google", "aws", "azure"] = Field(..., description="Cloud provider for TTS")
    duration_seconds: Optional[float] = Field(default=None, description="Duration of the audio in seconds")


class Translation(BaseModel):
    """Pydantic model representing a translation of a phrase in Firestore.

    Audio is stored in a nested structure:
    {
      "flashcard": {
        "slow": PhraseAudio(...),
        "normal": PhraseAudio(...),
        "fast": PhraseAudio(...)
      },
      "story": {
        "slow": PhraseAudio(...),
        "normal": PhraseAudio(...),
        "fast": PhraseAudio(...)
      }
    }
    """
    model_config = {"arbitrary_types_allowed": True} # allow us to use Language
    phrase_hash: str = Field(..., description="Hash of the associated English root phrase")
    language: BCP47Language = Field(..., description="BCP-47 language tag for the translation")
    text: str = Field(..., description="Translated text of the phrase")
    text_lower: LowercaseStr = Field(..., description="Lowercase version for consistent lookups")
    tokens: List[str] = Field(..., description="Tokenised words from the translated phrase")
    audio: Optional[Dict[AudioSettings, Dict[AudioSpeeds, PhraseAudio]]] = Field(
        default=None, description="Nested audio structure by setting (flashcard/story) and speed (slow/normal/fast)"
    )

