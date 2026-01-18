from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from pydub import AudioSegment


from src.phrases.search import get_phrase
from src.llm_tools.refine_story_translation import refine_story_translation
from src.storage import (
    PUBLIC_BUCKET,
)


from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from src.models import BCP47Language, get_language

from src.phrases.phrase_model import Phrase, Translation  # noqa: F401 - Translation needed for Pydantic model resolution
from google.cloud.firestore import DocumentReference
from src.connections.gcloud_auth import get_firestore_client
from src.phrases.utils import generate_phrase_hash
from src.phrases.phrase_model import FirePhraseDataModel
from src.logger import logger
from langcodes import Language
from src.audio.constants import INTER_UTTERANCE_GAP, STORY_PART_TRANSITION
from src.storage import upload_to_gcs
from src.utils import render_html_content


def get_story(
    story_title: str | None = None, story_title_hash: str | None = None
) -> Optional["Story"]:
    """Retrieve a Story from Firestore by its title hash."""
    if story_title:
        story_title_hash = generate_phrase_hash(story_title)

    doc = _get_story_doc_ref(story_title_hash).get()
    if doc.exists:
        story_data = doc.to_dict()
        return Story.model_validate(story_data)
    else:
        logger.warning(f"Story with hash {story_title_hash} not found in Firestore.")
        return None


def _story_exists(story_title_hash: str) -> bool:
    doc = _get_story_doc_ref(story_title_hash).get()
    return doc.exists


def _get_story_doc_ref(story_title_hash: str) -> DocumentReference:
    client = get_firestore_client()
    return client.collection("stories").document(story_title_hash)


def get_all_stories() -> List[Story]:
    """Gets all story title hashes from firestore

    Returns:
        List[str]: the story titles
    """

    client = get_firestore_client()
    collection_ref = client.collection("stories")
    docs = collection_ref.stream()

    ALL_STORIES = []
    for story in docs:
        ALL_STORIES.append(Story.model_validate(story.to_dict()))

    return ALL_STORIES


class StoryPhrase(Phrase):
    """Phrase model specifically for story dialogues with context-aware translations."""

    story_title_hash: str = Field(..., description="Hash of the parent story")
    story_part: str = Field(
        ..., description="Part of the story (e.g., introduction, development)"
    )
    sequence: int = Field(
        ..., description="Utterance sequence within the story part", gte=0
    )

    @classmethod
    def create(
        cls, english_phrase: str, story_title_hash: str, story_part: str, sequence: int
    ) -> "StoryPhrase":
        """Factory method to create a StoryPhrase with story-specific hash."""
        # Create base phrase for NLP processing
        base_phrase = Phrase.create(english_phrase=english_phrase)

        # Generate story-specific phrase hash
        story_specific_hash = cls._generate_story_phrase_hash(
            english_phrase, story_title_hash, story_part, sequence
        )

        return cls(
            **base_phrase.model_dump(exclude={"key"}),
            key=story_specific_hash,
            story_title_hash=story_title_hash,
            story_part=story_part,
            sequence=sequence,
        )

    @staticmethod
    def _generate_story_phrase_hash(
        english_phrase: str, story_title_hash: str, story_part: str, sequence: int
    ) -> str:
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sequence: int = Field(..., description="Sequence number of the dialogue line")
    speaker: str = Field(..., description="Name of the speaker")
    text: str = Field(..., description="English Text spoken by the speaker")
    story_phrase: Optional[StoryPhrase] = Field(
        None, exclude=True, description="Phrase data like audio etc"
    )
    phrase_hash: Optional[str] = Field(
        None, description="Phrase hash for this dialogue line"
    )
    story_part: str = Field(
        ...,
        description="Name of the story part this utterance belongs to, 'introduction' etc",
    )
    target_text: Optional[str] = Field(
        None, description="Translated text for this utterance"
    )
    wiktionary_links: Optional[str] = Field(
        None, description="Wiktionary links for target text, space separted"
    )
    source_text: Optional[str] = Field(
        None,
        description="Source text if different from English, e.g., for reverse translations",
    )
    target_audio_normal: Optional[AudioSegment] = Field(
        None, exclude=True, description="Translated audio at normal speed"
    )

    @computed_field
    @property
    def target_audio_filename_normal(self) -> str:
        """Get the expected audio filename for normal speed."""
        return f"{self.story_part}_{self.speaker}_{self.sequence}_normal.mp3".lower()

    @computed_field
    @property
    def story_part_titlecase(self) -> str:
        """Get the part name in title case."""
        return self.story_part.replace("_", " ").title()

    @model_validator(mode="after")
    def assign_phrase_hash(self) -> "Utterance":
        """Assign phrase_hash from story_phrase if available."""
        if self.story_phrase:
            self.phrase_hash = self.story_phrase.key
        return self

    @model_validator(mode="after")
    def get_story_phrase(self) -> "Utterance":
        """Retrieve StoryPhrase object if phrase_hash is present but story_phrase is None."""
        if self.phrase_hash and not self.story_phrase:
            logger.debug(
                f"Getting Phrase data from Firestore for phrase_hash: {self.phrase_hash}"
            )
            story_phrase = get_phrase(self.phrase_hash)
            self.story_phrase = story_phrase
        return self

    def _verify_translation_loaded(self):
        if all(
            [
                self.target_text,
                self.wiktionary_links,
                self.source_text,
                self.target_audio_normal,
            ]
        ):
            return
        else:
            raise ValueError(
                f"Utternace not loaded with audio or text: {self.model_dump()}"
            )

    def _load_target_audio(
        self, language: Language | str, speed: str = "normal"
    ) -> AudioSegment:
        """Get combined audio for the utterance in the specified language. Adds audio to target_audio_normal."""
        language = get_language(language)
        if self.story_phrase:
            logger.debug(
                f"Retrieving audio for utterance seq {self.sequence} in language {language.to_tag()}"
            )
            self.target_audio_normal = self.story_phrase.get_audio(
                language=language, context="story", speed=speed
            )
            return self.target_audio_normal
        else:
            raise ValueError(f"No story phrase found for utterance seq {self.sequence}")

    def _publish_audio_to_gcs(self, base_prefix: str) -> str:
        """Publish the audio for this utterance to GCS and return the public URL.
        base prefix is the story location"""
        if not self.target_audio_normal:
            raise ValueError(f"No audio to upload for utterance seq {self.sequence}")

        return upload_to_gcs(
            obj=self.target_audio_normal,
            base_prefix=base_prefix,
            bucket_name=PUBLIC_BUCKET,
            file_name=self.target_audio_filename_normal,
            save_local=True,
        )

    def _load_story_translation(
        self,
        target_language: Language | str,
        source_language: Language | str = Language.get("en-GB"),
    ) -> None:
        """Get the translated text with Wiktionary links for vocabulary words.
        Adds new fields into the model: target_text and wiktionary_links"""
        target_language = get_language(target_language)
        source_language = get_language(source_language)
        self.get_story_phrase()
        self.story_phrase.translate(
            target_language=target_language, refine=False, overwrite=False
        )

        if self.story_phrase:
            source_translation = self.story_phrase._get_translation(source_language)
            target_translation = self.story_phrase._get_translation(target_language)
            if source_translation:
                self.source_text = source_translation.text
            else:
                logger.debug(
                    f"No source translation found for phrase hash {self.phrase_hash} in language {source_language.to_tag()}"
                )
                raise ValueError(
                    f"Source translation not found for phrase hash {self.phrase_hash} in language {source_language.to_tag()}"
                )
            if target_translation:
                self.wiktionary_links = target_translation.get_wiktionary_links()
                self.target_text = target_translation.text
            else:
                logger.debug(
                    f"No translation found for phrase hash {self.phrase_hash} in language {target_language.to_tag()}"
                )
                raise ValueError(
                    f"Translation not found for phrase hash {self.phrase_hash} in language {target_language.to_tag()}"
                )
        else:
            logger.debug(f"No story phrase found for utterance seq {self.sequence}")
            raise ValueError(f"No story phrase found for utterance seq {self.sequence}")


class PublishedStory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    source_language: BCP47Language = Field(
        ..., description="BCP47 tag of the source language"
    )
    target_language: BCP47Language = Field(
        ..., description="BCP47 tag of the target language"
    )
    gcs_path: str = Field(..., description="GCS path of the published story")
    active: bool = Field(
        default=True, description="Whether this published version is active"
    )
    public_url: str = Field(default="", description="Public URL for story")
    title: str = Field(..., description="The title of the corresponding Story")
    deck: str = Field(..., description="The deck of the corresponding Story")
    collection: str = Field(..., description="Collection story comes from")

    def _is_published(
        self,
        source_language: Language | None = None,
        target_language: Language | None = None,
    ) -> bool:
        """Checks for a match based on source and optional target language"""
        if source_language is None and target_language is None:
            raise ValueError("At least one language must be specified")

        source_match = (source_language is None) or (
            self.source_language == source_language
        )
        target_match = (target_language is None) or (
            self.target_language == target_language
        )
        return source_match and target_match


class Story(FirePhraseDataModel):
    title: str = Field(..., description="Title of the story")
    summary: str = Field("", description="Brief summary of the story")
    story_parts: Dict[str, List[Utterance]] = Field(
        ..., description="Dictionary of story parts (e.g., introduction, development)"
    )
    firestore_collection: str = Field(
        "stories", description="Firestore collection name for stories"
    )
    collection: Optional[str] = Field(
        default=None, description="List of collections this story belongs to"
    )
    deck: Optional[str] = Field(
        default=None,
        description="What decks this story supports. Name made from collection ID and deck name, e.g. LM2000-Pack01",
    )
    published: Dict[str, PublishedStory] = Field(
        default_factory=dict,
        description="Dictionary of published versions of the story, key is source_tag|target_tag",
    )
    target_language_name: Optional[str] = Field(
        None, description="Name of the target language for translation"
    )
    source_language_name: Optional[str] = Field(
        None, description="Name of the source language"
    )
    target_language_tag: Optional[str] = Field(
        None, description="BCP47 tag of the target language for translation"
    )
    source_language_tag: Optional[str] = Field(
        None, description="BCP47 tag of the source language for translation"
    )

    @model_validator(mode="after")
    def sort_story_parts(self) -> "Story":
        """Ensure each story_part is sorted."""
        self.story_parts = dict(sorted(self.story_parts.items()))
        return self

    @computed_field
    @property
    def title_snake_case(self) -> str:
        """Get the story title in snake case."""
        return self.title.lower().replace(" ", "_")

    @classmethod
    def create(
        cls,
        title: str,
        summary: str,
        story_dialogue: Dict[str, List[Utterance]],
        collection: str | None = None,
        deck: str | None = None,
    ) -> "Story":
        """Factory method to create a Story instance."""
        story_title_hash = generate_phrase_hash(title)
        if _story_exists(story_title_hash):
            raise ValueError(f"Story with title {title} already exists in Firestore.")
        story_parts = defaultdict(list)
        # create sequence and story part information for each piece of dialogue
        for story_part, part_dialogue in story_dialogue.items():
            story_parts[story_part] = []  # will be a list of utterances

            for seq, utterance in enumerate(part_dialogue["dialogue"]):
                story_phrase_obj = StoryPhrase.create(
                    english_phrase=utterance["text"],
                    story_title_hash=story_title_hash,
                    story_part=story_part,
                    sequence=seq,
                )

                utterance_obj = Utterance(
                    sequence=seq,
                    speaker=utterance["speaker"],
                    text=utterance["text"],
                    story_title_hash=story_title_hash,
                    story_phrase=story_phrase_obj,
                    story_part=story_part,
                )
                story_parts[story_part].append(utterance_obj)

        story_obj = cls(
            title=title,
            key=story_title_hash,
            summary=summary,
            story_parts=story_parts,
            collection=collection,
            deck=deck,
        )

        return story_obj

    def publish_story(
        self,
        target_language: Language | str,
        source_language: Language | str = "en-GB",
        overwrite: bool = False,
    ) -> str:
        """Publish the story to the public GCS bucket for the specified language pair.
        Overwrite will collect translation data again and re-publish, if overwrite is False and the story is already published
        it will be skipped."""
        target_language = get_language(target_language)
        source_language = get_language(source_language)

        # Check if already published
        if not overwrite:
            if self._is_published(
                target_language=target_language, source_language=source_language
            ):
                logger.info(
                    f"Story '{self.title}' already published for {source_language.to_tag()} -> {target_language.to_tag()}"
                )
                return

        self._load_story_translation(
            target_language=target_language, source_language=source_language
        )
        self._verify_translation_loaded()
        gcs_path = self._get_gcs_path()

        # Upload HTML to GCS
        html_content = self._render_story_html()

        story_page = upload_to_gcs(
            obj=html_content,
            bucket_name=PUBLIC_BUCKET,
            base_prefix=gcs_path,
            file_name="index.html",
            content_type="text/html",
        )
        logger.info(f"Published story at {story_page}")

        self._publish_all_audio()

        publish_tag = self._get_published_tag(target_language, source_language)
        # Add to published dictionary
        public_url = self._get_public_url()
        self.published[publish_tag] = PublishedStory(
            source_language=self.source_language_tag,
            target_language=self.target_language_tag,
            gcs_path=gcs_path,
            active=True,
            public_url=public_url,
            title=self.title,
            deck=self.deck,
            collection=self.collection,
        )

        logger.info(
            f"Finished Publishing story '{self.title}' to GCS at {gcs_path} for {source_language.to_tag()} -> {target_language.to_tag()}"
        )
        self.upload()
        return public_url

    def get_story_text(self) -> str:
        """Simple string of all utterances (english only)
        Used to feed into LLM tools for a story summary"""
        dialogue = ",".join(
            [
                utterance.text
                for story_part, utterances in self.story_parts.items()
                for utterance in utterances
            ]
        )
        return self.title + ": summary: " + self.summary + ": dialogue: " + dialogue

    def _get_published_tag(self, target_language: Language, source_language: Language):
        """Dictionary key for the published dictionary such as en-GB-fr-FR"""

        return f"{source_language.to_tag()}|{target_language.to_tag()}"

    def get_all_published_source_languages(self) -> List[Language]:
        """all published elements matching languages"""

        return [published.source_language for _, published in self.published.items()]

    def get_published_stories(
        self,
        source_language: Language | str | None = "en-GB",
        target_language: Language | None = None,
    ) -> List[PublishedStory]:
        """all published elements matching languages"""

        source_language = get_language(source_language)
        target_language = get_language(target_language)

        matching_published = []
        for _, _published in self.published.items():
            if _published._is_published(source_language, target_language):
                matching_published.append(_published)
        return matching_published

    def _verify_utterances_loaded(self) -> None:
        for story_part, utterances in self.story_parts.items():
            for utterance in utterances:
                utterance._verify_translation_loaded()

    def _verify_translation_loaded(self) -> None:
        if all(
            [
                self.target_language_name,
                self.source_language_name,
                self.target_language_tag,
                self.source_language_tag,
            ]
        ):
            self._verify_utterances_loaded()
            return
        else:
            raise ValueError(
                "We do not have a translation loaded into the story, run self._load_story_translation()"
            )

    def _publish_story_parts_audio(self) -> List[str]:
        """Publish all story parts audio"""

        all_files_published = []
        for story_part in self.story_parts:
            all_files_published.append(self._publish_story_part_audio(story_part))

        return all_files_published

    def _publish_story_part_audio(self, story_part: str) -> str:
        """Publish the audio part of the store, returning the gcs file path"""

        story_part_audio = self._get_story_part_audio(story_part)
        story_part_filename = story_part + ".mp3"
        return upload_to_gcs(
            obj=story_part_audio,
            bucket_name=PUBLIC_BUCKET,
            base_prefix=self._get_gcs_path(),
            file_name=story_part_filename,
        )

    def _publish_full_story_audio(self) -> str:
        all_story_audio = self._get_story_audio()
        all_story_filename = "full_story.mp3"
        return upload_to_gcs(
            obj=all_story_audio,
            file_name=all_story_filename,
            bucket_name=PUBLIC_BUCKET,
            base_prefix=self._get_gcs_path(),
        )

    def _get_story_part_audio(self, story_part: str) -> AudioSegment:
        all_audio = []
        utterances = self.story_parts[story_part]
        for utterance in utterances:
            all_audio.append(utterance.target_audio_normal)
            all_audio.append(INTER_UTTERANCE_GAP)

        return sum(all_audio)

    def _get_story_audio(self) -> AudioSegment:
        """Returns a single audio segment for the entire story"""

        all_audio: List[AudioSegment] = []
        for story_part in self.story_parts:
            all_audio.append(self._get_story_part_audio(story_part))
            all_audio.append(STORY_PART_TRANSITION)

        return sum(all_audio)

    def _publish_all_audio(self) -> None:
        """Publish all audio files to story GCS bucket"""
        full_file = self._publish_full_story_audio()
        logger.info(f"Published full story {full_file}")
        story_part_files = self._publish_story_parts_audio()
        logger.info(f"Published parts {','.join(story_part_files)}")
        utterances = self._publish_audio_utterances()
        logger.info(f"Published utterances {','.join(utterances)}")

    def _publish_audio_utterances(self) -> List[str]:
        """Publish all utterance audio to GCS and return list of public URLs."""

        public_urls = []
        for story_part, utterances in self.story_parts.items():
            for utterance in utterances:
                logger.debug(
                    f"Publishing audio for utterance seq {utterance.text} in part '{story_part}'"
                )
                public_url = utterance._publish_audio_to_gcs(
                    base_prefix=self._get_gcs_path()
                )
                public_urls.append(public_url)
        return public_urls

    def _get_public_url(self):
        gcs_stub = self._get_gcs_path()
        full_path = (
            f"https://storage.googleapis.com/{PUBLIC_BUCKET}/{gcs_stub}index.html"
        )
        return full_path

    def _render_story_html(self) -> str:
        """Render the story as an HTML page for the specified language pair."""

        html_content = render_html_content(self.model_dump(), "story_template.html")
        return html_content

    def _is_published(
        self, target_language: Language | None, source_language: Language
    ) -> bool:
        """Check if the story is already published for the specified language pair."""

        for _, published_story in self.published.items():
            if published_story._is_published(
                source_language=source_language, target_language=target_language
            ):
                return True
        return False

    def _load_story_translation(
        self,
        target_language: Language | str,
        source_language: Language | str = Language.get("en-GB"),
    ) -> None:
        """Get the translated text with Wiktionary links for vocabulary words for all utterances.
        Adds new fields into each Utterance: target_text and wiktionary_links"""
        target_language = get_language(target_language)
        source_language = get_language(source_language)

        self.target_language_name = target_language.language_name()
        self.source_language_name = source_language.language_name()
        self.target_language_tag = target_language.to_tag()
        self.source_language_tag = source_language.to_tag()

        for story_part, utterances in self.story_parts.items():
            for utterance in utterances:
                logger.debug(
                    f"Getting translation for utterance seq {utterance.text} in part '{story_part}'"
                )
                utterance._load_story_translation(
                    target_language=target_language, source_language=source_language
                )

        logger.debug(f"Generating audio with overwrite as False for {target_language}")
        self.generate_audio(language=target_language, overwrite=False)

        self._load_audio(target_language=target_language, speed="normal")

    def _get_gcs_path(self) -> str:
        """Get the public GCS path for the story webpage"""
        if not self.target_language_tag or not self.source_language_tag:
            raise ValueError(
                "Run self._load_story_translation() first to populate self.target_language_tag"
            )

        return (
            f"stories/{self.source_language_tag}/{self.target_language_tag}/{self.key}/"
        )

    def generate_audio(self, language: Language | str, overwrite: bool = False) -> None:
        """Generate audio for all utterances in the story for the specified language."""
        language = get_language(language)
        for story_part, utterances in self.story_parts.items():
            for utterance in utterances:
                gender = (
                    "MALE" if utterance.speaker.lower().strip() == "sam" else "FEMALE"
                )
                if utterance.story_phrase:
                    logger.debug(
                        f"Generating audio for utterance seq {utterance.text} in language {language.to_tag()}"
                    )
                    utterance.story_phrase.generate_audio(
                        context="story",
                        gender=gender,
                        language=language,
                        overwrite=overwrite,
                    )
                else:
                    logger.debug(
                        f"Skipping audio generation for utterance seq {utterance.text} in part '{story_part}' - no story phrase found"
                    )

    def _load_audio(
        self, target_language: Language | str, speed: str = "normal"
    ) -> Dict[str, List[AudioSegment]]:
        """Get combined audio for the entire story in the specified language."""
        target_language = get_language(target_language)
        all_audio = defaultdict(list)
        for story_part, utterances in self.story_parts.items():
            for utterance in utterances:
                phrase_audio = utterance._load_target_audio(
                    language=target_language, speed=speed
                )
                if phrase_audio:
                    all_audio[story_part].append(phrase_audio)
                else:
                    raise ValueError(
                        f"No audio found for utterance seq {utterance.text} in part '{story_part}'"
                    )
        return all_audio

    def upload(
        self, language: Language | str | None = None, overwrite: bool = False
    ) -> DocumentReference:
        """Upload the story to Firestore including all phrases and their audio etc.
        Returns the DocumentReference for the story itself."""
        if language:
            language = get_language(language)
        self._upload_phrase_entries(language=language, overwrite=overwrite)
        return self._upload_to_firestore()

    def _upload_phrase_entries(
        self, language: Language | str | None = None, overwrite: bool = False
    ) -> List[DocumentReference]:
        """Upload all StoryPhrase entries to Firestore."""
        if language:
            language = get_language(language)
        doc_refs = []
        for story_part, utterances in self.story_parts.items():
            for utterance in utterances:
                if utterance.story_phrase:
                    doc_ref = utterance.story_phrase.upload(
                        language=language, overwrite=overwrite
                    )
                    doc_refs.append(doc_ref)
        return doc_refs

    def _upload_to_firestore(self) -> DocumentReference:
        """Upload the story to Firestore."""
        doc_ref = self._get_firestore_document_reference()
        doc_ref.set(
            self.model_dump(
                mode="json",
                exclude={
                    "source_text",
                    "target_text",
                    "wiktionary_links",
                    "target_language_name",
                    "source_language_name",
                    "target_language_tag",
                    "source_language_tag",
                },
            )
        )
        return doc_ref

    def translate(
        self,
        target_language: Language | str,
        refine: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Translate all utterances in the story to the target language."""
        target_language = get_language(target_language)

        for story_part, utterances in self.story_parts.items():
            for utterance in utterances:
                if utterance.story_phrase:
                    logger.debug(
                        f"Translating utterance seq {utterance.text} in part '{story_part}'"
                    )
                    utterance.story_phrase.translate(
                        target_language, refine=False, overwrite=overwrite
                    )
        if refine:
            logger.debug(
                f"Refining translations for story '{self.title}' to language {target_language.to_tag()}'"
            )
            self._refine_translations(target_language)
        else:
            logger.debug(
                f"Skipping refinement of translations for story '{self.title}' to language {target_language.to_tag()}'"
            )

    def _replace_translations(
        self,
        refined_translations: Dict[str, List[Dict[str, str]]],
        target_language: Language,
    ) -> None:
        """Replace translations in the story with refined translations."""

        for story_part, utterances in self.story_parts.items():
            refined_part = refined_translations.get(story_part, [])
            logger.debug(
                f"Refining translations for part '{story_part}' with {len(utterances)} utterances"
            )
            if not refined_part:
                raise ValueError(
                    f"Refined dialogue missing part '{story_part}' for story '{self.title}'"
                )
            for utterance, refined_utterance in zip(utterances, refined_part):
                refined_text = refined_utterance["translation"]
                utterance.story_phrase.translate(
                    target_language=target_language,
                    overwrite=True,
                    translated_text=refined_text,
                )
                logger.debug(
                    f"Replaced translation for utterance with '{refined_text}'"
                )

    def _refine_translations(self, target_language: Language) -> None:
        """Refine translations for all utterances in the story to the target language."""
        translation_dialogue = self._get_dialogue_with_translations(target_language)
        refined_dialogue = refine_story_translation(
            story_parts=translation_dialogue, language=target_language
        )
        self._replace_translations(refined_dialogue, target_language)

    def _get_dialogue_with_translations(
        self, target_language: Language
    ) -> Dict[str, List[Dict]]:
        """Get the story dialogue with translations for the target language."""
        dialogue_with_translations = {}
        for story_part, utterances in self.story_parts.items():
            dialogue_with_translations[story_part] = []
            for utterance in utterances:
                translation_text = ""
                if utterance.story_phrase:
                    translation = utterance.story_phrase._get_translation(
                        target_language
                    )
                    if translation:
                        translation_text = translation.text

                        dialogue_with_translations[story_part].append(
                            {
                                "speaker": utterance.speaker,
                                "text": utterance.text,
                                "translation": translation_text,
                            }
                        )
                    else:
                        logger.exception(
                            f"Translation not found for phrase hash {utterance.phrase_hash} in language {target_language.to_tag()}"
                        )
                        raise ValueError(
                            f"Translation not found for phrase hash {utterance.phrase_hash} in language {target_language.to_tag()}"
                        )

        return dict(sorted(dialogue_with_translations.items()))


Story.model_rebuild()
