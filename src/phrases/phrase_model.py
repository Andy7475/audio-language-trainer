from __future__ import annotations

from typing import List, Optional, Literal, Annotated

from pydantic import BaseModel, Field, field_validator, BeforeValidator, ConfigDict
from pydub import AudioSegment
from PIL import Image
from src.logger import logger
from src.connections.gcloud_auth import get_firestore_client
from src.phrases.utils import generate_phrase_hash
from google.cloud.firestore import Client as FirestoreClient
from src.nlp import (
    extract_lemmas_and_pos,
    get_tokens_from_lemmas_and_pos,
    get_verbs_from_lemmas_and_pos,
    get_vocab_from_lemmas_and_pos,
    get_text_tokens,
)
from src.models import BCP47Language, get_language
from src.translation import (
    translate_with_google_translate,
    refine_translation_with_anthropic,
)
from src.storage import (
    upload_file_to_gcs,
    download_from_gcs,
    get_phrase_image_path,
    get_phrase_audio_path,
    check_blob_exists,
    PRIVATE_BUCKET,
)
from src.audio.voices import get_voice_model, VoiceInfo
from src.audio.generation import generate_translation_audio
from google.cloud.firestore import DocumentReference
from src.llm_tools.image_generation import generate_phrase_image_prompt
from src.images.generator import generate_image as generate_image_with_provider
from src.images.manipulation import resize_image

LowercaseStr = Annotated[str, BeforeValidator(lambda v: v.lower().strip())]


def get_phrase(
    phrase_hash: str, database_name: str = "firephrases"
) -> Optional[Phrase]:
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
    client = get_firestore_client(database_name)
    doc_ref = client.collection("phrases").document(phrase_hash)
    doc = doc_ref.get()

    if not doc.exists:
        raise ValueError(f"Phrase with hash {phrase_hash} not found in Firestore")

    # Get phrase data
    phrase_data = doc.to_dict()

    core_phrase_references = {
        "key": phrase_hash,
        "firestore_document_ref": doc_ref,
    }

    # Fetch all translations from subcollection
    translations_docs = doc_ref.collection("translations").stream()
    translations = {}
    for translated_doc in translations_docs:
        translation_data = translated_doc.to_dict()
        translation = Translation.model_validate(translation_data)
        # Use language tag as the key
        translations[translation.language.to_tag()] = translation

    # Add translations to phrase data
    phrase_data["translations"] = translations
    phrase_data.update(core_phrase_references)
    return Phrase.model_validate(phrase_data)


def get_phrase_by_english(
    english_phrase: str, database_name: str = "firephrases"
) -> Optional[Phrase]:
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


class FirePhraseDataModel(BaseModel):
    """Parent model for common variables and storage details for FireStore"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    firestore_collection: Optional[Literal["phrases", "stories"]] = Field(
        default="phrases", description="Context for the phrase in FireStore"
    )
    bucket_name: str = Field(default=PRIVATE_BUCKET)
    firestore_database: str = Field(
        default="firephrases", description="database name in FireStore"
    )
    firestore_document_ref: DocumentReference | None = Field(
        default=None,
        exclude=True,
        description="The firestore document reference of the parent phrase",
    )
    key: str = Field(
        ..., description="Key in Firestore to get the document reference"
    )

    def _get_firestore_document_reference(self) -> DocumentReference:
        """Get or create a Firestore document reference for this phrase.

        Returns the existing reference if available, otherwise creates one
        from the key and caches it.

        Returns:
            DocumentReference: Firestore document reference
        """
        if self.firestore_document_ref is not None:
            return self.firestore_document_ref

        # Create a new reference and cache it
        client = get_firestore_client(self.firestore_database)
        doc_ref = client.collection(self.firestore_collection).document(self.key)
        self.firestore_document_ref = doc_ref
        return doc_ref


class Phrase(FirePhraseDataModel):
    """Pydantic model representing a phrase in Firestore.

    This model corresponds to the phrases collection schema defined in firestore.md.
    Each phrase contains English text with linguistic analysis including tokens, lemmas,
    verbs, and vocabulary. Translations are stored in a subcollection and fetched separately.
    """

    firestore_collection: str = Field(default="phrases", description="Firestore collection name for phrases")
    english: str = Field(
        ..., description="Original English phrase with original capitalisation"
    )
    english_lower: LowercaseStr = Field(
        ..., description="Lowercase version for consistent lookups"
    )
    tokens: List[str] = Field(..., description="Tokenised words from the phrase")
    verbs: List[str] = Field(
        default_factory=list, description="Lemmatised verb forms only"
    )
    vocab: List[str] = Field(
        default_factory=list, description="Lemmatised non-verb words"
    )
    translations: dict[str, Translation] = Field(
        default_factory=dict,
        description="Dictionary of translations keyed by language tag (e.g., 'fr-FR')",
    )
    collections: List[str] = Field(
        default_factory=list, description="Collections this phrase belongs to"
    )

    def __str__(self) -> str:
        return self.english
    @classmethod
    def create(cls, english_phrase: str) -> Phrase:
        """Factory method to create a Phrase with NLP processing.

        Args:
            english_phrase: The English phrase text

        Returns:
            Phrase: A new Phrase object with NLP analysis and default en-GB translation
        """
        phrase_hash = generate_phrase_hash(english_phrase)
        lemmas_and_pos = extract_lemmas_and_pos([english_phrase], language_code="en")
        tokens = get_tokens_from_lemmas_and_pos(lemmas_and_pos)

        phrase = cls(
            key=phrase_hash,
            english=english_phrase,
            english_lower=english_phrase,
            tokens=tokens,
            verbs=get_verbs_from_lemmas_and_pos(lemmas_and_pos),
            vocab=get_vocab_from_lemmas_and_pos(lemmas_and_pos),
        )

        # Add default en-GB translation
        en_translation = phrase._get_english_translation()
        phrase.translations[en_translation.language.to_tag()] = en_translation

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

    def _get_translation_firestore_document_ref(self, language: BCP47Language) -> DocumentReference:
        """Get the Firestore document reference for a specific translation.

        Args:
            language: BCP47 language tag for the translation
        Returns:
            DocumentReference: Firestore document reference for the translation
        """
        if self.firestore_document_ref is None:
            self._get_firestore_document_reference()
        translation_doc_ref = self.firestore_document_ref.collection("translations").document(language.to_tag())
        return translation_doc_ref
    
    def _has_translation(self, language: BCP47Language) -> bool:
        """Check if a translation exists for the given language.

        Args:
            language: BCP47 language tag to check for

        Returns:
            bool: True if translation exists, False otherwise
        """
        return language.to_tag() in self.translations

    def _get_translation(self, language: BCP47Language) -> Optional[Translation]:
        """Get the translation for a specific language if it exists.

        Args:
            language: BCP47 language tag to retrieve

        Returns:
            Optional[Translation]: The translation if found, None otherwise
        """
        if not self._has_translation(language):
            raise ValueError(f"No translation found for language {language.to_tag()}")

        return self.translations[language.to_tag()]

    def _get_english_translation(self) -> Translation:
        """Create an English (en-GB) translation from the Phrase.

        This creates a Translation object for the default English phrase, copying
        linguistic data (tokens, text) from the Phrase model. No audio or image
        is included at this stage.

        Returns:
            Translation: An English translation with en-GB language tag
        """

        firestore_document_ref = self._get_translation_firestore_document_ref(
            BCP47Language.get("en-GB")
        )
        return Translation(
            key=self.key,
            firestore_document_ref=firestore_document_ref,
            language=BCP47Language.get("en-GB"),
            text=self.english,
            text_lower=self.english_lower,
            tokens=self.tokens,
            image_file_path=get_phrase_image_path(
                phrase_hash=self.key, language=BCP47Language.get("en-GB")
            ),
        )

    def translate(
        self,
        target_language: BCP47Language | str,
        refine: bool = True,
        model: Optional[str] = None,
        overwrite: bool = False,
        translated_text: str | None = None,
    ) -> Translation:
        """Translate the phrase to a target language.

        This method will:
        1. Check if translation already exists and return it if found
        2. Translate using Google Translate API
        3. Optionally refine the translation using Anthropic's Claude
        4. Tokenize the translated text
        5. Create and return a Translation object
        6. Add the translation to the phrase's translations dict

        Args:
            target_language: BCP47 language tag for the target language
            refine: Whether to refine the translation with Claude (default: True)
            model: Anthropic model to use for refinement (defaults to claude-sonnet-4-20250514)

        Returns:
            Translation: The translation object

        Raises:
            RuntimeError: If translation fails

        Example:
            >>> phrase = Phrase.create("Hello, how are you?")
            >>> fr_translation = phrase.translate(BCP47Language.get("fr-FR"))
            >>> logger.info(fr_translation.text)
            'Bonjour, comment allez-vous ?'
        """
        # Check if translation already exists
        target_language = get_language(target_language)

        if self._has_translation(target_language) and not overwrite:
            logger.info(f"Translation for {target_language.to_tag()} already exists")
            return self._get_translation(target_language)

        if translated_text is None:
            # Step 1: Translate with Google Translate
            translated_text = translate_with_google_translate(
                text=self.english, target_language=target_language, source_language="en"
            )

            # Step 2: Refine with Claude if requested
            if refine:
                translated_text = refine_translation_with_anthropic(
                    source_phrase=self.english,
                    initial_translation=translated_text,
                    target_language=target_language,
                    source_language=None,
                    model=model,
                )

        # Step 3: Tokenize the translated text
        # Extract language code for tokenization (e.g., "fr" from "fr-FR")
        language_code = target_language.language
        tokens = get_text_tokens(translated_text, language_code=language_code)

        translation_document_ref = self._get_translation_firestore_document_ref(target_language)
        # Step 4: Create the Translation object
        translation = Translation(
            key=self.key,
            firestore_document_ref=translation_document_ref,
            language=target_language,
            text=translated_text,
            text_lower=translated_text.lower(),
            tokens=tokens,
        )
        translation._set_image_file_path(default=True)

        # Step 5: Add the translation to the phrase's translations dict
        self.translations[translation.language.to_tag()] = translation

        return translation

    def upload(
        self, language: BCP47Language | None = None, overwrite: bool = False
    ) -> DocumentReference:
        """Upload the phrase and its translations to Firestore and GCS. If language specified, then just upload that language.

        This method will:
        1. Upload the phrase document to Firestore
        2. Upload each translation document to the translations subcollection
        3. Upload all audio files for each translation to GCS
        4. Upload images for each translation to GCS

        Args:
            database_name: Name of the Firestore database (default: "firephrases")
            overwrite: If True, overwrite existing multimedia files in GCS (default: False)

        Returns:
            str: The Firestore document reference

        Raises:
            RuntimeError: If upload fails
        """
        if language:
            logger.info(f"Uploading phrase {self.key} with {language.to_tag()} translation to Firestore and GCS")
        else:
            logger.info(f"Uploading phrase {self.key} with all translations to Firestore and GCS")


        doc_ref = self._upload_to_firestore()


        # Step 3 & 4: Upload audio and images to GCS
        self._upload_translations(language=language, overwrite=overwrite)

        return doc_ref

    def download(self, language: BCP47Language | None = None, local:bool = True) -> None:
        """Download multimedia files for all translations from GCS.

        This method downloads audio and image files for all translations (or a specific language).
        By default, all translations share the same image (en-GB)

        Args:
            language: If specified, only download multimedia for this language.
                     If None (default), downloads for all translations.


        Example:
            >>> phrase = get_phrase("hello_a3f8d2")
            >>> # Download all multimedia for all translations
            >>> phrase.download()
            >>> # Download only French translation multimedia
            >>> phrase.download(language=BCP47Language.get("fr-FR"))

        """
        if language:
            logger.info(f"Downloading multimedia for phrase {self.key} - {language.to_tag()} translation only")
        else:
            logger.info(f"Downloading multimedia for phrase {self.key} - all translations")

        download_all_languages = True
        if language is not None:
            download_all_languages = False

        for translation in self.translations.values():
            if download_all_languages or (translation.language == language):
                translation.download(local=local)

    def generate_audio(
        self,
        context: Literal["flashcard", "story"],
        language: BCP47Language | str | None = None,
        gender: Literal["MALE", "FEMALE"] = "FEMALE",
        overwrite: bool = False,
        local: bool = True,
    ) -> None:
        """Generate audio for all translations (or a specific language).

        This method generates audio files for translations using text-to-speech.
        Audio is generated for the specified context and gender.

        Args:
            context: Audio context ("flashcard" or "story"). Flashcard generates both
                    normal and slow speeds, while story only generates normal speed.
            language: If specified, only generate audio for this language.
                     If None (default), generates audio for all translations.
            gender: Voice gender for TTS ("MALE" or "FEMALE", default: "FEMALE")
            overwrite: If True, regenerate audio even if it already exists (default: False)

        Example:
            >>> phrase = Phrase.create("Hello, how are you?")
            >>> phrase.translate("fr-FR")
            >>> # Generate flashcard audio for all translations
            >>> phrase.generate_audio(context="flashcard")
            >>> # Generate story audio only for French
            >>> phrase.generate_audio(context="story", language=BCP47Language.get("fr-FR"))
        """

        if language is not None:
            language = get_language(language)
            self.translations[language.to_tag()].generate_audio(
                context=context, gender=gender, overwrite=overwrite, local=local)

        else:
            for translation in self.translations.values():
                translation.generate_audio(
                    context=context, gender=gender, overwrite=overwrite, local=local
                )

    def generate_image(
        self,
        language: BCP47Language | str | None = None,
        style: str = "default",
        overwrite: bool = False,
        model_order: List[Literal["imagen", "stability", "deepai"]] = None,
    ) -> None:
        """Generate an image for a translation.

        By default, generates a shared en-GB image used by all translations. When a specific
        language is provided, generates a bespoke image for that translation with a
        language-specific file path.

        Args:
            language: Target language for image generation. If None (default), generates
                     shared en-GB image. If specified, generates bespoke image for that language.
            style: Art style to apply (default: "default")
            overwrite: If True, regenerate image even if it already exists (default: False)
            model_order: List of image providers to try in order
                        (default: ["imagen", "stability", "deepai"])

        Raises:
            ValueError: If the specified language translation doesn't exist
            RuntimeError: If all image generation providers fail

        Example:
            >>> phrase = Phrase.create("The cat sleeps on the mat")
            >>> phrase.translate("fr-FR")
            >>> # Generate shared en-GB image (used by all translations)
            >>> phrase.generate_image()
            >>> # Generate bespoke French image
            >>> phrase.generate_image(language="fr-FR")
            >>> # Upload images
            >>> phrase.upload(overwrite=True)
        """
        # Determine target language (default to en-GB for shared image)
        if language is None:
            target_language = BCP47Language.get("en-GB")
        else:
            target_language = get_language(language)

        # Get the translation for this language
        language_tag = target_language.to_tag()
        if language_tag not in self.translations:
            raise ValueError(
                f"No translation found for language {language_tag}. "
                f"Available translations: {list(self.translations.keys())}"
            )
        translation = self.translations[language_tag]

       # Update image file path for bespoke images - it will have a default one created during translation
        if language_tag != "en-GB":
            translation._set_image_file_path(default=False)

        if check_blob_exists(self.bucket_name, translation.image_file_path) and not overwrite:
            logger.info(f"Image already exists at {translation.image_file_path}, skipping generation")
            self.download(language=target_language, local=False)
            return

 

        prompt = generate_phrase_image_prompt(self.english)

        # Generate the image using available providers
        image = generate_image_with_provider(
            prompt=prompt,
            style=style,
            model_order=model_order,
        )

        if image is None:
            raise RuntimeError(
                f"Failed to generate image for {language_tag} with all providers"
            )

        # Resize to standard size
        image = resize_image(image, height=500, width=500)

        # Set the image on the translation
        translation.image = image
        logger.info(f"✅ Generated image for {language_tag}")

    def get_audio(
        self,
        language: BCP47Language | str,
        context: Literal["flashcard", "story"] = "flashcard",
        speed: Literal["slow", "normal", "fast"] = "normal",
        local: bool = True,
    ) -> AudioSegment:
        """Get audio for a specific translation, context, and speed.

        Automatically downloads the audio from GCS if not already loaded locally.

        Args:
            language: BCP47 language tag for the translation
            context: Audio context ("flashcard" or "story")
            speed: Audio speed ("slow", "normal", or "fast")
            local: If True, use local cache for download (default: True)

        Returns:
            AudioSegment: The requested audio segment

        Raises:
            ValueError: If translation doesn't exist, or audio doesn't exist for the
                       specified context/speed combination

        Example:
            >>> phrase = get_phrase_by_english("Hello, how are you?")
            >>> audio = phrase.get_audio("fr-FR", "flashcard", "slow")
            >>> audio.export("output.mp3")
        """
        # Convert string to BCP47Language if needed
        language = get_language(language)
        language_tag = language.to_tag()

        # Check if translation exists
        if language_tag not in self.translations:
            raise ValueError(
                f"No translation found for language {language_tag}. "
                f"Available translations: {list(self.translations.keys())}"
            )

        translation = self.translations[language_tag]

        # Check if audio exists for this context/speed
        if context not in translation.audio or speed not in translation.audio[context]:
            raise ValueError(
                f"No audio found for {language_tag} with context='{context}' and speed='{speed}'. "
                f"Available contexts: {list(translation.audio.keys())}"
            )

        phrase_audio = translation.audio[context][speed]

        # Download if not already loaded
        if phrase_audio.audio_segment is None:
            logger.info(f"Downloading audio for {language_tag} {context}/{speed}...")
            phrase_audio.download(local=local)

        return phrase_audio.audio_segment

    def get_image(
        self,
        language: BCP47Language | str | None = None,
        local: bool = True,
    ) -> Image.Image:
        """Get image for a translation.

        By default, returns the shared en-GB image. Automatically downloads the image
        from GCS if not already loaded locally.

        Args:
            language: BCP47 language tag for the translation. If None (default),
                     returns the shared en-GB image.
            local: If True, use local cache for download (default: True)

        Returns:
            Image.Image: The requested PIL Image

        Raises:
            ValueError: If translation doesn't exist or image path is not set

        Example:
            >>> phrase = get_phrase_by_english("Hello, how are you?")
            >>> # Get shared English image
            >>> image = phrase.get_image()
            >>> image.show()
            >>> # Get bespoke French image (if it exists)
            >>> fr_image = phrase.get_image("fr-FR")
        """
        # Default to en-GB for shared image
        if language is None:
            language = BCP47Language.get("en-GB")
        else:
            language = get_language(language)

        language_tag = language.to_tag()

        # Check if translation exists
        if language_tag not in self.translations:
            raise ValueError(
                f"No translation found for language {language_tag}. "
                f"Available translations: {list(self.translations.keys())}"
            )

        translation = self.translations[language_tag]

        # Check if image file path is set
        if translation.image_file_path is None:
            raise ValueError(
                f"No image_file_path set for {language_tag}. "
                f"Generate an image first using phrase.generate_image()"
            )

        # Download if not already loaded
        if translation.image is None:
            logger.info(f"Downloading image for {language_tag}...")
            translation._download_image_from_gcs(local=local)

        return translation.image

    def _upload_to_firestore(
        self
    ) -> DocumentReference:
        """Upload a phrase and its translations to Firestore.

        Uploads the phrase document to `phrases/{phrase_hash}` and each translation to
        `phrases/{phrase_hash}/translations/{language_code}` subcollection.
        """
        
        if self.firestore_document_ref is None:
            self.firestore_document_ref = self._get_firestore_document_reference()
        phrase_data = self.model_dump(mode="json", exclude={"translations"})
        self.firestore_document_ref.set(phrase_data)

        return self.firestore_document_ref

    def _upload_translations(
        self, language: BCP47Language | None = None, overwrite: bool = False
    ) -> list[DocumentReference]:
        """Upload each translation to firestore and GCS"""

        all_references = []
        if language:
            translation = self.translations.get(language.to_tag())
            if not translation:
                raise ValueError(
                    f"No translation found for language {language.to_tag()}"
                )
            all_references.append(translation.upload(overwrite=overwrite))
        else:
            for language_code, translation in self.translations.items():
                all_references.append(translation.upload(overwrite=overwrite))

        return all_references


class PhraseAudio(FirePhraseDataModel):
    """Pydantic model representing audio metadata for a specific setting/speed combination.

    This represents a single audio file with its metadata (file path, voice, duration, etc).
    Audio is organized by context (flashcard/story) and speed (slow/normal/fast).

    Note: Stores only the file_path (relative path) in Firestore to save space and allow
    flexible bucket management. The full GCS URI can be reconstructed using gcs_uri_from_file_path()
    with the PRIVATE_BUCKET constant.

    Binary audio_segment data is excluded from Firestore serialization and loaded on demand.
    """

    text: str = Field(..., description="Translated text of the phrase")
    file_path: str = Field(
        ...,
        description="GCS file path (e.g., 'phrases/fr-FR/audio/flashcard/slow/hello_a3f8d2.mp3')",
    )
    language: BCP47Language = Field(
        ..., description="BCP-47 language tag for the translation"
    )
    context: Literal["flashcard", "story"] = Field(
        ..., description="Context/setting for the audio (flashcard or story)"
    )
    speed: Literal["slow", "normal", "fast"] = Field(
        ..., description="Speaking speed of the audio"
    )
    voice_info: VoiceInfo = Field(
        ..., description="Voice model information (provider and voice_id)"
    )
    duration_seconds: Optional[float] = Field(
        default=None, description="Duration of the audio in seconds"
    )
    # Binary data excluded from Firestore serialization
    audio_segment: Optional[AudioSegment] = Field(
        default=None, exclude=True, description="Audio data (excluded from Firestore)"
    )
    gender: Optional[Literal["MALE", "FEMALE"]] = Field(
        default="FEMALE", description="Gender of the speaker"
    )

    @classmethod
    def create(
        cls,
        key: str,
        text: str,
        language: BCP47Language,
        context: Literal["flashcard", "story"],
        speed: Literal["slow", "normal"],
        gender: Literal["MALE", "FEMALE"] = "FEMALE",
    ) -> PhraseAudio:
        """Factory method to create a PhraseAudio object with generated file_path.

        Args:
            phrase_hash: Hash of the associated English root phrase"""

        file_path = get_phrase_audio_path(
            phrase_hash=key, language=language, context=context, speed=speed
        )

        voice_info = get_voice_model(
            language_code=language.to_tag(), gender=gender, audio_type=context
        )

        return cls(
            key=key,
            text=text,
            file_path=file_path,
            language=language,
            context=context,
            speed=speed,
            voice_info=voice_info,
            gender=gender
        )

    def _upload_to_gcs(self) -> None:
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

        logger.info(f"Uploading audio: {self.language.to_tag()} {self.context}/{self.speed} to {self.file_path} (local cache enabled)")

        try:
            upload_file_to_gcs(
                obj=self.audio_segment,
                bucket_name=self.bucket_name,
                file_path=self.file_path,
                save_local=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to upload audio to {self.file_path}: {e}")

    def download(self, local:bool=True) -> None:
        """Download audio file from GCS.

        Public method to download the audio segment from GCS and populate the audio_segment field.
        """
        self._download_from_gcs(local=local)

    def _download_from_gcs(self, local:bool) -> AudioSegment:
        """Download this audio file from GCS.

        Downloads the audio from GCS using the stored file_path and stores it in audio_segment.

        Args:
            bucket_name: GCS bucket name (default: PRIVATE_BUCKET)

        Returns:
            AudioSegment: The downloaded audio

        Raises:
            ValueError: If download fails
        """
        local_status = "with local cache" if local else "without local cache"
        logger.info(f"Downloading audio: {self.language.to_tag()} {self.context}/{self.speed} from {self.file_path} ({local_status})")

        try:
            audio_segment = download_from_gcs(
                bucket_name=self.bucket_name,
                file_path=self.file_path,
                expected_type="audio",
                use_local=local,
            )
            self.audio_segment = audio_segment
            return audio_segment
        except Exception as e:
            raise ValueError(f"Failed to download audio from {self.file_path}: {e}")

    def generate_audio(self) -> None:
        """Generates audio for this PhraseAudio using TTS. Uploads it to GCS"""

        # Generate normal or slow audio using TTS
        self.audio_segment = generate_translation_audio(
            translated_text=self.text,
            voice_model=self.voice_info,
            speed=self.speed,
        )

        self.duration_seconds = self.audio_segment.duration_seconds
        #self._upload_to_gcs() # should be a separate call to .upload()

class Translation(FirePhraseDataModel):
    """Pydantic model representing a translation of a phrase in Firestore.

    Audio is stored as a list of PhraseAudio objects, each containing metadata about
    a specific audio file (context, speed, voice, file path, etc.). This flat structure
    is simpler and more flexible than nested dicts.

    Binary data (image) are excluded from Firestore serialization.
    Audio segments are attached to PhraseAudio objects and loaded on demand.
    """

    model_config = {
        "arbitrary_types_allowed": True
    }  # allow us to use Language and binary types
    language: BCP47Language = Field(
        ..., description="BCP-47 language tag for the translation"
    )
    text: str = Field(..., description="Translated text of the phrase")
    text_lower: LowercaseStr = Field(
        ..., description="Lowercase version for consistent lookups"
    )
    tokens: List[str] = Field(
        ..., description="Tokenised words from the translated phrase"
    )
    audio: dict[str, dict[str, PhraseAudio]] = Field(
        default_factory=dict,
        description="Nested dict of audio files: {context: {speed: PhraseAudio}}. Example: {'flashcard': {'normal': PhraseAudio, 'slow': PhraseAudio}, 'story': {'normal': PhraseAudio}}",
    )
    image_file_path: Optional[str] = Field(
        default=None,
        description="GCS file path for the image associated with this translation (if any)",
    )
    # Binary data excluded from Firestore serialization
    image: Optional[Image.Image] = Field(
        default=None, exclude=True, description="Image data (excluded from Firestore)"
    )

    @field_validator("audio", mode="before")
    @classmethod
    def _ensure_audio_dict(cls, value: None | dict | List[PhraseAudio]) -> dict[str, dict[str, PhraseAudio]]:
        """Ensure audio is always a dict, even if None or list is provided.

        Handles migration from old list format to new nested dict format.
        """
        if value is None:
            return {}

        # If it's already a dict, return as-is
        if isinstance(value, dict):
            return value

        # If it's a list (old format), convert to nested dict
        if isinstance(value, list):
            result = {}
            for audio_item in value:
                context = audio_item.context
                speed = audio_item.speed
                if context not in result:
                    result[context] = {}
                result[context][speed] = audio_item
            return result

        return {}

    def get_wiktionary_links(
        self,
        force_refresh: bool = False,
        separator: str = " ",
    ) -> str:
        """Generate HTML links to Wiktionary for each token in this translation.

        Uses Firestore caching to avoid repeated web lookups. Returns links in the
        same order as the tokens, preserving the word order from the original text.

        Args:
            force_refresh: Force fresh lookups even if cached (default: False)
            max_age_days: Refresh cache if older than this many days (default: 90)
            separator: String to join links (default: single space)

        Returns:
            str: HTML string with wiktionary links, tokens in original order

        Example:
            >>> translation = phrase.translations["fr-FR"]
            >>> html = translation.get_wiktionary_links()
            >>> logger.info(html)
            '<a href="...">Bonjour</a> <a href="...">le</a> monde'

        Note:
            - Uses language code only (not territory), e.g., 'fr' not 'fr-FR'
            - Returns plain text for tokens without Wiktionary entries
            - Preserves original token order and casing
        """
        from src.wiktionary import batch_get_or_fetch_wiktionary_entries

        # Extract language code (e.g., 'fr' from 'fr-FR')
        language_code = self.language.language

        # Get or fetch entries for all tokens (batch operation)
        entries = batch_get_or_fetch_wiktionary_entries(
            tokens=self.tokens,
            language_code=language_code,
            force_refresh=force_refresh,
            database_name=self.firestore_database,
        )

        # Generate HTML links preserving token order
        links = []
        for token in self.tokens:
            token_lower = token.lower()
            entry = entries.get(token_lower)

            if entry and entry.exists:
                # Use entry to generate link with original token casing
                links.append(entry.get_html_link(token))
            else:
                # No entry found, return plain text
                links.append(token)

        return separator.join(links)

    def download(self, local:bool = True) -> None:
        """Download all multimedia files from GCS.

        Public method to download all audio files and the image for this translation.

        Args:
            unique_image_for_language: If True, downloads language-specific image.
                                      If False (default), downloads en-GB image.
        """
        self._download_from_gcs(local=local)

    def _download_from_gcs(self, local:bool) -> None:
        """Downloads multimedia files from GCS"""

        logger.info(f"Downloading all multimedia for {self.language.to_tag()} translation")

        for context_dict in self.audio.values():
            for phrase_audio in context_dict.values():
                phrase_audio._download_from_gcs(local=local)

        self.image = self._download_image_from_gcs(local=local)

    def upload(self, overwrite: bool = False) -> DocumentReference:
        """Uploads the translation to Firestore and files to GCS"""

        doc_ref = self._upload_to_firestore()
        self._upload_to_gcs(overwrite=overwrite)
        return doc_ref

    def _upload_to_firestore(self) -> DocumentReference:
        """Uploads the translation text and URLs to firestore"""

        translation_data = self.model_dump(mode="json")
        self.firestore_document_ref.set(translation_data)
        return self.firestore_document_ref

    def _upload_to_gcs(self, overwrite: bool = False) -> None:
        """Upload multimedia files to GCS"""

        logger.info(f"Uploading all multimedia for {self.language.to_tag()} translation")

        if self.audio:
            for context_dict in self.audio.values():
                for phrase_audio in context_dict.values():
                    if check_blob_exists(self.bucket_name, phrase_audio.file_path) and not overwrite:
                        logger.info(f"Audio already exists at {phrase_audio.file_path}, skipping upload")
                    else:
                        phrase_audio._upload_to_gcs()
        if self.image:
            if check_blob_exists(self.bucket_name, self.image_file_path) and not overwrite:
                logger.info(f"Image already exists at {self.image_file_path}, skipping upload")
            else:
                self._upload_image_to_gcs()

    def _set_image_file_path(self, default: bool = True) -> str:
        """Set the GCS file path for the phrase image in the specified language.
        Defaults to english image"""
        if default:
            language = BCP47Language.get("en-GB")
        else:
            language = self.language

        self.image_file_path = get_phrase_image_path(
            phrase_hash=self.key,
            language=language,
        )

    def _upload_image_to_gcs(self) -> str:
        """Upload image for this translation to GCS and return file_path.

        Args:
            bucket_name: GCS bucket name (default: PRIVATE_BUCKET)

        Returns:
            str: GCS file path of the uploaded image (e.g., 'phrases/en-GB/images/hello_a3f8d2.png')

        Raises:
            ValueError: If image is not set or upload fails
        """
        if self.image is None:
            logger.info(f"No image attached to translation ({self.language.to_tag()})")
        if self.image_file_path is None:
            raise ValueError("image_file_path is not set")

        logger.info(f"Uploading image: {self.language.to_tag()} to {self.image_file_path} (local cache enabled)")

        uri = upload_file_to_gcs(
            obj=self.image,
            bucket_name=self.bucket_name,
            file_path=self.image_file_path,
            save_local=True,
        )

        return uri

    def _download_image_from_gcs(self, local:bool) -> Image.Image:
        """Downloads image file from GCS"""

        local_status = "with local cache" if local else "without local cache"
        logger.info(f"Downloading image: {self.language.to_tag()} from {self.image_file_path} ({local_status})")

        # Download from GCS (will use local cache if available)
        image = download_from_gcs(
            bucket_name=self.bucket_name,
            file_path=self.image_file_path,
            expected_type="image",
            use_local=local,
        )

        self.image = image
        return image

    def _phrase_audio_exists(
        self,
        context: Literal["flashcard", "story"],
        speed: Literal["slow", "normal", "fast"],
    ) -> bool:
        """Check if PhraseAudio object exists in the nested dictionary.

        Args:
            context: Audio context ("flashcard" or "story")
            speed: Audio speed ("slow", "normal", or "fast")

        Returns:
            bool: True if PhraseAudio object exists in memory, False otherwise
        """
        return context in self.audio and speed in self.audio[context]

    def _check_audio_file_exists(
        self,
        context: Literal["flashcard", "story"],
        speed: Literal["slow", "normal", "fast"],
    ) -> bool:
        """Check if audio file exists in GCS storage.

        Returns:
            bool: True if audio file exists in GCS, False otherwise
        """
        file_path = get_phrase_audio_path(
            phrase_hash=self.key,
            language=self.language,
            context=context,
            speed=speed,
        )
        return check_blob_exists(self.bucket_name, file_path)

    def generate_audio(
        self,
        context: Literal["flashcard", "story"],
        gender: Literal["MALE", "FEMALE"] = "FEMALE",
        overwrite: bool = False,
        local:bool = True,
    ) -> None:
        """Generate audio for this translation at appropriate speeds based on context."""

        audio_speeds = {
            "story": ["normal"],
            "flashcard": ["normal", "slow"],
        }
        speeds = audio_speeds[context]

        # Ensure context key exists in the nested dict
        if context not in self.audio:
            self.audio[context] = {}

        for speed in speeds:
            # Check if PhraseAudio already exists in memory
            if self._phrase_audio_exists(context, speed) and not overwrite:
                logger.info(
                    f"Audio already exists for {self.language.to_tag()} {context} {speed}, skipping generation"
                )
                continue

            # Determine if we need to generate or download existing audio
            audio_segment = None

            # Check if audio file exists in GCS (for historical audio)
            if not overwrite and self._check_audio_file_exists(context, speed):
                # Download existing audio from GCS
                file_path = get_phrase_audio_path(
                    phrase_hash=self.key,
                    language=self.language,
                    context=context,
                    speed=speed,
                )
                logger.info(f"Downloading existing audio from GCS for {self.language.to_tag()} {context} {speed}")
                audio_segment = download_from_gcs(
                    bucket_name=self.bucket_name,
                    file_path=file_path,
                    expected_type="audio",
                    use_local=local,
                )

            # Create PhraseAudio object (either with downloaded or newly generated audio)
            phrase_audio = PhraseAudio.create(
                key=self.key,
                text=self.text,
                language=self.language,
                context=context,
                speed=speed,
                gender=gender,
            )

            # Use existing audio or generate new
            if audio_segment is not None:
                # Use downloaded audio from GCS
                phrase_audio.audio_segment = audio_segment
                phrase_audio.duration_seconds = audio_segment.duration_seconds
            else:
                # Generate new audio
                phrase_audio.generate_audio()

            # Store in nested dict structure
            self.audio[context][speed] = phrase_audio

    def _upload_audio_to_gcs(self) -> None:
        """Upload all audio files in this translation to GCS.

        Iterates through all PhraseAudio objects and uploads each to GCS.
        Only uploads audio that has audio_segment set.

        Args:
            bucket_name: GCS bucket name for uploading audio. Defaults to PRIVATE_BUCKET.

        Raises:
            ValueError: If any audio upload fails
        """
        for context_dict in self.audio.values():
            for phrase_audio in context_dict.values():
                phrase_audio._upload_to_gcs()
