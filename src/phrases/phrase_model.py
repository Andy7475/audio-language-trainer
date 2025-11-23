from __future__ import annotations

from typing import List, Optional, Literal, Annotated

from pydantic import BaseModel, Field, BeforeValidator
from pydub import AudioSegment
from PIL import Image

from src.connections.gcloud_auth import get_firestore_client
from src.phrases.utils import generate_phrase_hash
from google.cloud.firestore import Client as FirestoreClient
from src.nlp import extract_lemmas_and_pos, get_tokens_from_lemmas_and_pos, get_verbs_from_lemmas_and_pos, get_vocab_from_lemmas_and_pos, get_text_tokens
from src.models import BCP47Language
from src.translation import translate_with_google_translate, refine_translation_with_anthropic
from src.storage import (
    upload_file_to_gcs,
    download_from_gcs,
    get_phrase_audio_path,
    get_phrase_image_path,
    gcs_uri_from_file_path,
    file_path_from_gcs_uri,
    PRIVATE_BUCKET,
)
from src.audio.voices import get_voice_model
from src.audio.generation import generate_translation_audio, generate_fast_audio

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
            Phrase: A new Phrase object with NLP analysis and default en-GB translation
        """
        phrase_hash = generate_phrase_hash(english_phrase)
        lemmas_and_pos = extract_lemmas_and_pos([english_phrase], language_code="en")
        tokens = get_tokens_from_lemmas_and_pos(lemmas_and_pos)

        phrase = cls(
            phrase_hash=phrase_hash,
            english=english_phrase,
            english_lower=english_phrase,
            tokens=tokens,
            verbs=get_verbs_from_lemmas_and_pos(lemmas_and_pos),
            vocab=get_vocab_from_lemmas_and_pos(lemmas_and_pos),
            source=source,
            translations=[]  # Translations are loaded separately from subcollection
        )

        # Add default en-GB translation
        phrase.translations.append(phrase._get_english_translation())

        return phrase

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

    def _get_english_translation(self) -> Translation:
        """Create an English (en-GB) translation from the Phrase.

        This creates a Translation object for the default English phrase, copying
        linguistic data (tokens, text) from the Phrase model. No audio or image
        is included at this stage.

        Returns:
            Translation: An English translation with en-GB language tag
        """
        return Translation(
            phrase_hash=self.phrase_hash,
            language=BCP47Language.get("en-GB"),
            text=self.english,
            text_lower=self.english_lower,
            tokens=self.tokens,
            audio=None,
            image_file_path=None,
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

        Binary data (audio_segment, image) are automatically excluded from Firestore.

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
            # Note: model_dump automatically excludes fields marked with exclude=True (audio_segment, image)
            for translation in self.translations:
                language_tag = translation.language.to_tag()
                translation_data = translation.model_dump(mode="json")
                translations_ref = doc_ref.collection("translations").document(language_tag)
                translations_ref.set(translation_data)

            return phrase_hash

        except Exception as e:
            raise RuntimeError(f"Failed to upload phrase: {e}")

    def upload_all_audio(self, bucket_name: str = PRIVATE_BUCKET) -> None:
        """Upload all audio from all translations to GCS.

        Iterates through all translations and calls upload_all_audio() on each.

        Args:
            bucket_name: GCS bucket name for uploading audio. Defaults to PRIVATE_BUCKET.

        Raises:
            ValueError: If any audio upload fails
        """
        for translation in self.translations:
            translation.upload_all_audio(bucket_name)


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

    This represents a single audio file with its metadata (file path, voice, duration, etc).
    Audio is organized by context (flashcard/story) and speed (slow/normal/fast).

    Note: Stores only the file_path (relative path) in Firestore to save space and allow
    flexible bucket management. The full GCS URI can be reconstructed using gcs_uri_from_file_path()
    with the PRIVATE_BUCKET constant.

    Binary audio_segment data is excluded from Firestore serialization and loaded on demand.
    """
    model_config = {"arbitrary_types_allowed": True}  # allow us to use AudioSegment type
    file_path: str = Field(..., description="GCS file path (e.g., 'phrases/fr-FR/audio/flashcard/slow/hello_a3f8d2.mp3')")
    language_tag: str = Field(..., description="BCP-47 language tag (e.g., 'fr-FR', 'ja-JP')")
    context: Literal["flashcard", "story"] = Field(..., description="Context/setting for the audio (flashcard or story)")
    speed: Literal["slow", "normal", "fast"] = Field(..., description="Speaking speed of the audio")
    voice_model_id: str = Field(..., description="Identifier of the voice model used")
    voice_provider: Literal["google", "aws", "azure"] = Field(..., description="Cloud provider for TTS")
    duration_seconds: Optional[float] = Field(default=None, description="Duration of the audio in seconds")
    # Binary data excluded from Firestore serialization
    audio_segment: Optional[AudioSegment] = Field(default=None, exclude=True, description="Audio data (excluded from Firestore)")

    def get_gcs_uri(self, bucket_name: str = PRIVATE_BUCKET) -> str:
        """Get the full GCS URI for this audio file.

        Args:
            bucket_name: GCS bucket name (default: PRIVATE_BUCKET)

        Returns:
            str: Full GCS URI (e.g., 'gs://bucket-name/phrases/fr-FR/audio/flashcard/slow/hello_a3f8d2.mp3')
        """
        return gcs_uri_from_file_path(self.file_path, bucket_name)

    def upload_to_gcs(self, bucket_name: str = PRIVATE_BUCKET) -> None:
        """Upload this audio file to GCS.

        Uploads the audio_segment to the GCS path specified by file_path.

        Args:
            bucket_name: GCS bucket name (default: PRIVATE_BUCKET)

        Raises:
            ValueError: If audio_segment is not set or upload fails
        """
        if self.audio_segment is None:
            raise ValueError("audio_segment is not set, cannot upload")

        try:
            upload_file_to_gcs(
                obj=self.audio_segment,
                bucket_name=bucket_name,
                file_path=self.file_path,
                save_local=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to upload audio to {self.file_path}: {e}")

    def download_from_gcs(self, bucket_name: str = PRIVATE_BUCKET) -> AudioSegment:
        """Download this audio file from GCS.

        Downloads the audio from GCS using the stored file_path and stores it in audio_segment.

        Args:
            bucket_name: GCS bucket name (default: PRIVATE_BUCKET)

        Returns:
            AudioSegment: The downloaded audio

        Raises:
            ValueError: If download fails
        """
        try:
            audio_segment = download_from_gcs(
                bucket_name=bucket_name,
                file_path=self.file_path,
                expected_type="audio",
                use_local=True,
            )
            self.audio_segment = audio_segment
            return audio_segment
        except Exception as e:
            raise ValueError(f"Failed to download audio from {self.file_path}: {e}")


class Translation(BaseModel):
    """Pydantic model representing a translation of a phrase in Firestore.

    Audio is stored as a list of PhraseAudio objects, each containing metadata about
    a specific audio file (context, speed, voice, file path, etc.). This flat structure
    is simpler and more flexible than nested dicts.

    Binary data (image) are excluded from Firestore serialization.
    Audio segments are attached to PhraseAudio objects and loaded on demand.
    """
    model_config = {"arbitrary_types_allowed": True} # allow us to use Language and binary types
    phrase_hash: str = Field(..., description="Hash of the associated English root phrase")
    language: BCP47Language = Field(..., description="BCP-47 language tag for the translation")
    text: str = Field(..., description="Translated text of the phrase")
    text_lower: LowercaseStr = Field(..., description="Lowercase version for consistent lookups")
    tokens: List[str] = Field(..., description="Tokenised words from the translated phrase")
    audio: List[PhraseAudio] = Field(
        default_factory=list, description="List of audio files for this translation, each with metadata (context, speed, etc.)"
    )
    image_file_path: Optional[str] = Field(
        default=None,
        description="GCS file path for the phrase image (e.g., 'phrases/en-GB/images/hello_a3f8d2.png')"
    )
    # Binary data excluded from Firestore serialization
    image: Optional[Image.Image] = Field(default=None, exclude=True, description="Image data (excluded from Firestore)")


    def load_audio_from_file_path(
        self, file_path: str, bucket_name: str = PRIVATE_BUCKET
    ) -> AudioSegment:
        """Load audio from a file path or local cache.

        Finds the matching PhraseAudio in self.audio and calls download_from_gcs() on it.

        Args:
            file_path: GCS file path (e.g., 'phrases/fr-FR/audio/flashcard/slow/hello_a3f8d2.mp3')
            bucket_name: GCS bucket name (default: PRIVATE_BUCKET)

        Returns:
            AudioSegment: The loaded audio

        Raises:
            ValueError: If audio loading fails or file_path not found in self.audio
        """
        # Find the matching PhraseAudio object
        phrase_audio = next(
            (a for a in self.audio if a.file_path == file_path),
            None
        )
        if phrase_audio is None:
            raise ValueError(f"No audio found with file_path: {file_path}")

        # Download using PhraseAudio's method
        return phrase_audio.download_from_gcs(bucket_name)

    def upload_image(
        self,
        bucket_name: str = PRIVATE_BUCKET,
    ) -> str:
        """Upload image for this translation to GCS and return file_path.

        Args:
            bucket_name: GCS bucket name (default: PRIVATE_BUCKET)

        Returns:
            str: GCS file path of the uploaded image (e.g., 'phrases/en-GB/images/hello_a3f8d2.png')

        Raises:
            ValueError: If image is not set or upload fails
        """
        if self.image is None:
            raise ValueError(f"No image attached to translation ({self.language.to_tag()})")

        try:
            # Generate GCS path
            file_path = get_phrase_image_path(
                phrase_hash=self.phrase_hash,
                language=self.language,
            )

            # Upload image to GCS
            upload_file_to_gcs(
                obj=self.image,
                bucket_name=bucket_name,
                file_path=file_path,
                save_local=True,
            )

            # Store file path in the translation for later retrieval
            self.image_file_path = file_path

            return file_path

        except Exception as e:
            raise ValueError(f"Failed to upload image for translation ({self.language.to_tag()}): {e}")

    def load_image_from_file_path(
        self, file_path: str, bucket_name: str = PRIVATE_BUCKET
    ) -> Image.Image:
        """Load image from a file path or local cache.

        Args:
            file_path: GCS file path (e.g., 'phrases/en-GB/images/hello_a3f8d2.png')
            bucket_name: GCS bucket name (default: PRIVATE_BUCKET)

        Returns:
            Image.Image: The loaded PIL Image

        Raises:
            ValueError: If image loading fails
        """
        try:
            # Download from GCS (will use local cache if available)
            image = download_from_gcs(
                bucket_name=bucket_name,
                file_path=file_path,
                expected_type="image",
                use_local=True,
            )

            self.image = image
            return image

        except Exception as e:
            raise ValueError(f"Failed to load image from {file_path}: {e}")


    def generate_audio(
        self,
        context: Literal["flashcard", "story"],
        gender: Literal["MALE", "FEMALE"] = "FEMALE",
        bucket_name: str = PRIVATE_BUCKET,
    ) -> None:
        """Generate audio for this translation at appropriate speeds based on context.

        For flashcard context: generates both slow and normal speed audio.
        For story context: generates normal and fast speed audio.

        The appropriate voice model is loaded based on the translation's language and the audio context.
        Generated audio segments are automatically uploaded to GCS and their metadata is stored in self.audio.

        Args:
            context: Audio context - either "flashcard" or "story". Determines which speeds are generated.
            gender: Voice gender - either "MALE" or "FEMALE". Defaults to "FEMALE".
            bucket_name: GCS bucket name for uploading audio. Defaults to PRIVATE_BUCKET.

        Raises:
            ValueError: If language is not supported or audio generation fails
            RuntimeError: If audio upload fails

        Example:
            >>> phrase = Phrase.create_phrase("She runs to the store daily")
            >>> fr_translation = phrase.translate(BCP47Language.get("fr-FR"))
            >>> fr_translation.generate_audio(context="flashcard")
            >>> # Now fr_translation.audio contains slow and normal speed audio
            >>> fr_translation.generate_audio(context="story")
            >>> # Now fr_translation.audio also contains normal and fast speed audio for story context
        """
        # Get the voice model for this language and context
        # Note: voice model expects "flashcards" (plural) for flashcard context
        voice_model = get_voice_model(
            language_code=self.language.to_tag(),
            gender=gender,
            audio_type="flashcards" if context == "flashcard" else "story",
        )

        # Determine which speeds to generate based on context
        if context == "flashcard":
            speeds_to_generate = ["slow", "normal"]
        else:  # story
            speeds_to_generate = ["normal", "fast"]

        # Generate audio for each speed
        normal_audio = None
        for speed in speeds_to_generate:
            if speed == "fast":
                # Fast audio is generated from normal audio
                if normal_audio is None:
                    raise RuntimeError("Normal audio must be generated before fast audio")
                audio_segment = generate_fast_audio(normal_audio)
            else:
                # Generate normal or slow audio using TTS
                audio_segment = generate_translation_audio(
                    translated_text=self.text,
                    voice_model=voice_model,
                    speed=speed,
                )
                if speed == "normal":
                    normal_audio = audio_segment

            # Upload the audio and store metadata
            self.upload_audio(
                audio_segment=audio_segment,
                bucket_name=bucket_name,
                context=context,
                speed=speed,
                voice_provider=voice_model.provider.value,
                voice_model_id=voice_model.voice_id,
            )

    def upload_all_audio(self, bucket_name: str = PRIVATE_BUCKET) -> None:
        """Upload all audio files in this translation to GCS.

        Iterates through all PhraseAudio objects and uploads each to GCS.
        Only uploads audio that has audio_segment set.

        Args:
            bucket_name: GCS bucket name for uploading audio. Defaults to PRIVATE_BUCKET.

        Raises:
            ValueError: If any audio upload fails
        """
        for phrase_audio in self.audio:
            if phrase_audio.audio_segment is not None:
                phrase_audio.upload_to_gcs(bucket_name)

    def download_from_gcs(self, bucket_name: str = PRIVATE_BUCKET) -> None:
        """Download all audio files for this translation from GCS.

        Iterates through all PhraseAudio objects and downloads each from GCS.

        Args:
            bucket_name: GCS bucket name for downloading audio. Defaults to PRIVATE_BUCKET.

        Raises:
            ValueError: If any audio download fails
        """
        for phrase_audio in self.audio:
            phrase_audio.download_from_gcs(bucket_name)

