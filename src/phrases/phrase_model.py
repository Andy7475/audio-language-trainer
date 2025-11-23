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
    get_phrase_image_path,
    get_phrase_audio_path,

    PRIVATE_BUCKET,
)
from src.audio.voices import get_voice_model, VoiceInfo
from src.audio.generation import generate_translation_audio, generate_fast_audio

LowercaseStr = Annotated[str, BeforeValidator(lambda v: v.lower().strip() )]


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
            source=source
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

        if not self._has_translation(language):
            return None 

        for _tranlsation in self.transations:
            if _tranlsation.language == language:
                return _tranlsation

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
            tokens=self.tokens
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
            tokens=tokens)

        # Step 5: Add the translation to the phrase's translations list
        self.translations.append(translation)

        return translation

    def upload(self, database_name: str = "firephrases") -> str:
        """Upload the phrase and its translations to Firestore and GCS.

        This method will:
        1. Upload the phrase document to Firestore
        2. Upload each translation document to the translations subcollection
        3. Upload all audio files for each translation to GCS
        4. Upload images for each translation to GCS

        Args:
            database_name: Name of the Firestore database (default: "firephrases")

        Returns:
            str: The phrase hash (document ID) of the uploaded phrase

        Raises:
            RuntimeError: If upload fails
        """
        firestore_client = get_firestore_client(database_name)

        # Step 1 & 2: Upload phrase and translations to Firestore
        phrase_hash = self._upload_to_firestore(
            firestore_client=firestore_client,
            database_name=database_name
        )

        # Step 3 & 4: Upload audio and images to GCS
        self._upload_to_gcs(bucket_name=PRIVATE_BUCKET)

        return phrase_hash
    def _upload_to_firestore(self, firestore_client: FirestoreClient, database_name: str = "firephrases") -> str:
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
            # Upload phrase document (without translations)
            phrase_data = self.model_dump(mode="json", exclude={"translations"})
            doc_ref = firestore_client.collection("phrases").document(self.phrase_hash)
            doc_ref.set(phrase_data)

            # Upload each translation to subcollection
            for translation in self.translations:
                language_tag = translation.language.to_tag()
                translation_data = translation.model_dump(mode="json")
                translations_ref = doc_ref.collection("translations").document(language_tag)
                translations_ref.set(translation_data)

            return self.phrase_hash

        except Exception as e:
            raise RuntimeError(f"Failed to upload phrase: {e}")

    def _upload_to_gcs(self, bucket_name: str = PRIVATE_BUCKET) -> None:
        """Upload all audio from all translations to GCS.

        Iterates through all translations and calls upload_all_audio() on each.

        Args:
            bucket_name: GCS bucket name for uploading audio. Defaults to PRIVATE_BUCKET.

        Raises:
            ValueError: If any audio upload fails
        """
        for translation in self.translations:
            translation._upload_to_gcs(bucket_name)


class PhraseAudio(BaseModel):
    """Pydantic model representing audio metadata for a specific setting/speed combination.

    This represents a single audio file with its metadata (file path, voice, duration, etc).
    Audio is organized by context (flashcard/story) and speed (slow/normal/fast).

    Note: Stores only the file_path (relative path) in Firestore to save space and allow
    flexible bucket management. The full GCS URI can be reconstructed using gcs_uri_from_file_path()
    with the PRIVATE_BUCKET constant.

    Binary audio_segment data is excluded from Firestore serialization and loaded on demand.
    """
    model_config = {"arbitrary_types_allowed": True}  # allow us to use AudioSegment and VoiceInfo types
    phrase_hash: str = Field(..., description="Hash of the associated English root phrase")
    text: str = Field(..., description="Translated text of the phrase")
    file_path: str = Field(..., description="GCS file path (e.g., 'phrases/fr-FR/audio/flashcard/slow/hello_a3f8d2.mp3')")
    language: BCP47Language = Field(..., description="BCP-47 language tag for the translation")
    context: Literal["flashcard", "story"] = Field(..., description="Context/setting for the audio (flashcard or story)")
    speed: Literal["slow", "normal", "fast"] = Field(..., description="Speaking speed of the audio")
    voice_info: VoiceInfo = Field(..., description="Voice model information (provider and voice_id)")
    duration_seconds: Optional[float] = Field(default=None, description="Duration of the audio in seconds")
    # Binary data excluded from Firestore serialization
    audio_segment: Optional[AudioSegment] = Field(default=None, exclude=True, description="Audio data (excluded from Firestore)")

    @classmethod
    def create_phrase_audio(cls, phrase_hash: str, text: str, language: BCP47Language, context: Literal["flashcard", "story"], speed: Literal["slow", "normal"], gender:str = "FEMALE") -> PhraseAudio:
        """Factory method to create a PhraseAudio object with generated file_path.

        Args:
            phrase_hash: Hash of the associated English root phrase"""

        file_path = get_phrase_audio_path(
            phrase_hash=phrase_hash,
            language=language,
            context=context,
            speed=speed
        )

        voice_info = get_voice_model(
            language_code=language.to_tag(),gender=gender, audio_type=context)

        return cls(
            phrase_hash=phrase_hash,
            text=text,
            file_path=file_path,
            language=language,
            context=context,
            speed=speed,
            voice_info=voice_info)
        



    def _upload_to_gcs(self, bucket_name: str = PRIVATE_BUCKET) -> None:
        """Upload this audio file to GCS.

        Uploads the audio_segment to the GCS path specified by file_path.

        Args:
            bucket_name: GCS bucket name (default: PRIVATE_BUCKET)

        Raises:
            ValueError: If audio_segment is not set or upload fails
        """
        if self.audio_segment is None:
            Warning(f"No audio_segment to upload for {self.model_dump()}")
            return

        try:
            upload_file_to_gcs(
                obj=self.audio_segment,
                bucket_name=bucket_name,
                file_path=self.file_path,
                save_local=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to upload audio to {self.file_path}: {e}")

    def _download_from_gcs(self, bucket_name: str = PRIVATE_BUCKET) -> AudioSegment:
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

    def generate_audio(self)->None:
        """Generates audio for this PhraseAudio using TTS."""

        # Generate normal or slow audio using TTS
        self.audio_segment = generate_translation_audio(
            translated_text=self.text,
            voice_model=self.voice_info,
            speed=self.speed,
        )

        self.duration_seconds = self.audio_segment.duration_seconds


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
    # Binary data excluded from Firestore serialization
    image: Optional[Image.Image] = Field(default=None, exclude=True, description="Image data (excluded from Firestore)")


    def _download_from_gcs(
        self, bucket_name: str = PRIVATE_BUCKET, unique_image_for_language: bool = False
    ) -> None:
        """Downloads multimedia files from GCS

        """

        for phrase_audio in self.audio:
            self.audio.append(phrase_audio._download_from_gcs(bucket_name))
        
        self.image = self._download_image_from_gcs(
            bucket_name=bucket_name,
            unique_image_for_language=unique_image_for_language
        )
        
    def _upload_to_gcs(self, bucket_name: str = PRIVATE_BUCKET, unique_image_for_language:bool = False) -> None:
        """Upload multimedia files to GCS

        Args:
            bucket_name: GCS bucket name (default: PRIVATE_BUCKET)

        Raises:
            ValueError: If any upload fails
        """

        for phrase_audio in self.audio:
            phrase_audio._upload_to_gcs(bucket_name)

        self._upload_image_to_gcs(
            bucket_name=bucket_name,
            unique_image_for_language=False
        )
    def _upload_image_to_gcs(
        self,
        bucket_name: str = PRIVATE_BUCKET,
        unique_image_for_language: bool = False
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

        if not unique_image_for_language:
            # Common image for all languages is stored under en-GB
            language = BCP47Language.get("en-GB")
        else:
            language = self.language    
        try:
            # Generate GCS path
            file_path = get_phrase_image_path(
                phrase_hash=self.phrase_hash,
                language=language,
            )

            # Upload image to GCS
            upload_file_to_gcs(
                obj=self.image,
                bucket_name=bucket_name,
                file_path=file_path,
                save_local=True,
            )

            return file_path

        except Exception as e:
            raise ValueError(f"Failed to upload image for translation ({self.language.to_tag()}): {e}")

    def _download_image_from_gcs(
        self, bucket_name: str = PRIVATE_BUCKET, unique_to_language: bool = False
    ) -> Image.Image:
        """Downloads image file from GCS
        """

        # Common image for all languages is stored under en-GB
        if not unique_to_language:
            language = BCP47Language.get("en-GB")
        else:
            language = self.language
        image_file_path = get_phrase_image_path(
            phrase_hash=self.phrase_hash,
            language=language)

        try:
            # Download from GCS (will use local cache if available)
            image = download_from_gcs(
                bucket_name=bucket_name,
                file_path=image_file_path,
                expected_type="image",
                use_local=True,
            )

            self.image = image
            return image

        except Exception as e:
            raise ValueError(f"Failed to load image from {image_file_path}: {e}")


    def _check_audio_exists(self, context: Literal["flashcard", "story"], speed: Literal["slow", "normal", "fast"]) -> bool:
        """Check if audio exists for the given context and speed.

        Args:
            context: Audio context ("flashcard" or "story")
            speed: Audio speed ("slow", "normal", or "fast")

        Returns:
            bool: True if audio exists, False otherwise
        """
        for phrase_audio in self.audio:
            if phrase_audio.context == context and phrase_audio.speed == speed:
                return True
        return False

    def generate_audio(
        self,
        context: Literal["flashcard", "story"],
        gender: Literal["MALE", "FEMALE"] = "FEMALE",
        overwrite: bool = False
    ) -> None:
        """Generate audio for this translation at appropriate speeds based on context.

        """
        if context == "story":
            speeds = ["normal"]
        elif context == "flashcard":
            speeds = ["normal", "slow"]

        for speed in speeds:
            if self._check_audio_exists(context, speed) and not overwrite:
                print(f"Audio already exists for {self.language.to_tag()} {context} {speed}, skipping generation")
                continue
            phrase_audio = PhraseAudio.create_phrase_audio(
                phrase_hash=self.phrase_hash,
                text=self.text,
                language=self.language,
                context=context,
                speed=speed,
                gender=gender)

            phrase_audio.generate_audio()
            
            self.audio.append(phrase_audio)

    def _upload_audio_to_gcs(self, bucket_name: str = PRIVATE_BUCKET) -> None:
        """Upload all audio files in this translation to GCS.

        Iterates through all PhraseAudio objects and uploads each to GCS.
        Only uploads audio that has audio_segment set.

        Args:
            bucket_name: GCS bucket name for uploading audio. Defaults to PRIVATE_BUCKET.

        Raises:
            ValueError: If any audio upload fails
        """
        for phrase_audio in self.audio:
            phrase_audio._upload_to_gcs(bucket_name)

