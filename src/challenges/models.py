from __future__ import annotations

from typing import Dict, List, Literal, Optional

from google.cloud.firestore import DocumentReference
from langcodes import Language
from pydantic import BaseModel, ConfigDict, Field, computed_field

from llm_tools.base import load_prompt_template
from llm_tools.challenge_generation import generate_challenge_content
from src.connections.gcloud_auth import get_firestore_client
from src.logger import logger
from src.models import BCP47Language, get_language
from src.phrases.phrase_model import FirePhraseDataModel
from src.storage import PUBLIC_BUCKET, upload_to_gcs
from src.story import Story
from src.utils import render_html_content
from src.story import get_story


def get_challenge(story_title_hash: str) -> Optional["ChallengeRecord"]:
    """Retrieve a ChallengeRecord from Firestore by its story title hash."""
    doc = _get_challenge_doc_ref(story_title_hash).get()
    if doc.exists:
        challenge_data = doc.to_dict()
        return ChallengeRecord.model_validate(challenge_data)
    else:
        logger.warning(
            f"Challenge for story {story_title_hash} not found in Firestore."
        )
        return None


def get_challenge_by_story(story: "Story") -> Optional["ChallengeRecord"]:
    """Retrieve a ChallengeRecord from Firestore by Story object."""
    return get_challenge(story.key)


def _challenge_exists(story_title_hash: str) -> bool:
    """Check if a challenge exists for the given story."""
    doc = _get_challenge_doc_ref(story_title_hash).get()
    return doc.exists


def _get_challenge_doc_ref(story_title_hash: str) -> DocumentReference:
    """Get Firestore document reference for a challenge."""
    client = get_firestore_client()
    return client.collection("challenges").document(story_title_hash)


class ChallengeBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class QandA(ChallengeBaseModel):
    question: str
    answer: str


class Scenario(ChallengeBaseModel):
    role_learner: str = Field(
        ..., description="Role for learner to play (e.g., 'coffee shop customer')"
    )
    role_teacher: str = Field(
        ..., description="Role for teacher to play (e.g., 'coffee shop staff')"
    )
    situation: str = Field(
        ..., description="Setting description (e.g., 'A coffee shop')"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty level (e.g., 'easy', 'medium', 'hard')"
    )
    task: str = Field(..., description="Main task to complete (e.g., 'Order a coffee')")
    find_out: List[QandA] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Specific information to discover, together with a proposed answer (e.g., 'What is the price?', 5.00)",
    )


class Challenge(ChallengeBaseModel):
    scenarios: List[Scenario] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="List of 3 roleplay scenarios, one at each difficult",
    )


class PublishedChallenge(BaseModel):
    """Represents a published version of a challenge in a specific language pair."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_language: BCP47Language = Field(
        ..., description="BCP47 tag of the source language"
    )
    target_language: BCP47Language = Field(
        ..., description="BCP47 tag of the target language"
    )
    gcs_path: str = Field(..., description="GCS path of the published challenge")
    public_url: str = Field(..., description="Public URL for the challenge page")
    active: bool = Field(
        default=True, description="Whether this published version is active"
    )

    def _is_published(
        self,
        source_language: Language | None = None,
        target_language: Language | None = None,
    ) -> bool:
        """Checks for a match based on source and optional target language."""
        if source_language is None and target_language is None:
            raise ValueError("At least one language must be specified")

        source_match = (source_language is None) or (
            self.source_language == source_language
        )
        target_match = (target_language is None) or (
            self.target_language == target_language
        )
        return source_match and target_match


def generate_challenges(
    story: Story,
) -> Challenge:
    # Load prompt templates
    system_template = load_prompt_template("challenge_generation", "system")
    user_template = load_prompt_template("challenge_generation", "user")

    # Substitute variables
    system_prompt = system_template.substitute()
    user_prompt = user_template.substitute(
        story_content=story.get_story_text(),
    )

    challenge = generate_challenge_content(system_prompt, user_prompt)
    return challenge


class ChallengeRecord(FirePhraseDataModel):
    """Represents a set of language learning challenges for a story."""

    challenge: "Challenge" = Field(..., description="Challenge data with scenarios")
    published: Dict[str, PublishedChallenge] = Field(
        default_factory=dict,
        description="Dictionary of published versions, key is source_tag|target_tag",
    )
    firestore_collection: str = Field(
        "challenges", description="Firestore collection name for challenges"
    )

    @computed_field
    @property
    def story_title_hash(self) -> str:
        """Alias for key to maintain semantic clarity that this is linked to a story."""
        return self.key

    @classmethod
    def create(cls, story: Story) -> "ChallengeRecord":
        """Factory method to create a ChallengeRecord from a Story.

        Args:
            story: The Story object to generate challenges from

        Returns:
            ChallengeRecord: New instance with generated challenges (not yet uploaded)

        Raises:
            ValueError: If a challenge already exists for this story
        """

        story_title_hash = story.key

        if _challenge_exists(story_title_hash):
            raise ValueError(
                f"Challenge for story '{story.title}' already exists in Firestore."
            )

        logger.info(f"Generating challenges for story '{story.title}'")
        challenge = generate_challenges(story)

        challenge_record = cls(
            key=story_title_hash,
            challenge=challenge,
            firestore_collection="challenges",
        )

        logger.info(
            f"Created ChallengeRecord with {len(challenge.scenarios)} scenarios "
            f"for story '{story.title}'"
        )
        return challenge_record

    def publish(
        self,
        target_language: Language | str,
        source_language: Language | str = "en-GB",
        overwrite: bool = False,
    ) -> str:
        """Publish the challenge to the public GCS bucket for the specified language pair.

        Args:
            target_language: Target language for the challenge
            source_language: Source language (defaults to British English)
            overwrite: If True, republish even if already published for this language pair

        Returns:
            str: Public URL of the published challenge page
        """
        target_language = get_language(target_language)
        source_language = get_language(source_language)

        # Check if already published
        if not overwrite:
            if self._is_published(
                target_language=target_language, source_language=source_language
            ):
                logger.info(
                    f"Challenge for story hash '{self.story_title_hash}' already published "
                    f"for {source_language.to_tag()} -> {target_language.to_tag()}"
                )
                publish_tag = self._get_published_tag(target_language, source_language)
                return self.published[publish_tag].public_url

        gcs_path = self._get_gcs_path(source_language, target_language)

        # Render and upload HTML
        html_content = self._render_challenge_html(source_language, target_language)

        challenge_page = upload_to_gcs(
            obj=html_content,
            bucket_name=PUBLIC_BUCKET,
            base_prefix=gcs_path,
            file_name="index.html",
            content_type="text/html",
        )
        logger.info(f"Published challenge at {challenge_page}")

        # Create published record
        publish_tag = self._get_published_tag(target_language, source_language)
        public_url = self._get_public_url(source_language, target_language)

        self.published[publish_tag] = PublishedChallenge(
            source_language=source_language.to_tag(),
            target_language=target_language.to_tag(),
            gcs_path=gcs_path,
            public_url=public_url,
            active=True,
        )

        logger.info(
            f"Finished publishing challenge to GCS at {gcs_path} for "
            f"{source_language.to_tag()} -> {target_language.to_tag()}"
        )

        # Update Firestore with new published version
        self.upload()
        return public_url

    def upload(self) -> DocumentReference:
        """Upload the challenge record to Firestore.

        Returns:
            DocumentReference: Reference to the uploaded Firestore document
        """
        return self._upload_to_firestore()

    def _upload_to_firestore(self) -> DocumentReference:
        """Upload the challenge to Firestore."""
        doc_ref = self._get_firestore_document_reference()
        doc_ref.set(self.model_dump(mode="json"))
        logger.info(
            f"Uploaded ChallengeRecord to Firestore: challenges/{self.story_title_hash}"
        )
        return doc_ref

    def _render_challenge_html(
        self, source_language: Language, target_language: Language
    ) -> str:
        """Render the challenge as an HTML page for the specified language pair.

        Args:
            source_language: Source language
            target_language: Target language

        Returns:
            str: Rendered HTML content
        """
        # Get story title for the page - we'll need to fetch this

        story = get_story(story_title_hash=self.story_title_hash)
        title = story.title if story else "Language Challenge"

        context = {
            "title": title,
            "target_language": target_language.display_name(),
            "target_language_code": target_language.to_tag(),
            "source_language": source_language.display_name(),
            "source_language_code": source_language.to_tag(),
            "scenarios": [
                scenario.model_dump() for scenario in self.challenge.scenarios
            ],
        }

        html_content = render_html_content(context, "challenge_template.html")
        return html_content

    def _get_gcs_path(
        self, source_language: Language, target_language: Language
    ) -> str:
        """Get the GCS path for the challenge webpage.

        Args:
            source_language: Source language
            target_language: Target language

        Returns:
            str: GCS path like challenges/en-GB/sv-SE/story_hash/
        """
        return (
            f"challenges/{source_language.to_tag()}/"
            f"{target_language.to_tag()}/{self.story_title_hash}/"
        )

    def _get_public_url(
        self, source_language: Language, target_language: Language
    ) -> str:
        """Get the public URL for the challenge page.

        Args:
            source_language: Source language
            target_language: Target language

        Returns:
            str: Full public URL
        """
        gcs_stub = self._get_gcs_path(source_language, target_language)
        return f"https://storage.googleapis.com/{PUBLIC_BUCKET}/{gcs_stub}index.html"

    def _get_published_tag(
        self, target_language: Language, source_language: Language
    ) -> str:
        """Get the dictionary key for the published dictionary.

        Args:
            target_language: Target language
            source_language: Source language

        Returns:
            str: Tag like "en-GB|sv-SE"
        """
        return f"{source_language.to_tag()}|{target_language.to_tag()}"

    def _is_published(
        self,
        target_language: Language | None = None,
        source_language: Language | None = None,
    ) -> bool:
        """Check if the challenge is already published for the specified language pair.

        Args:
            target_language: Target language (optional)
            source_language: Source language (optional)

        Returns:
            bool: True if published for this language pair
        """
        for _, published_challenge in self.published.items():
            if published_challenge._is_published(
                source_language=source_language, target_language=target_language
            ):
                return True
        return False

    def get_published_challenges(
        self,
        source_language: Language | str | None = "en-GB",
        target_language: Language | str | None = None,
    ) -> List[PublishedChallenge]:
        """Get all published challenges matching the specified languages.

        Args:
            source_language: Source language filter (defaults to British English)
            target_language: Target language filter (optional)

        Returns:
            List[PublishedChallenge]: Matching published challenges
        """
        if source_language:
            source_language = get_language(source_language)
        if target_language:
            target_language = get_language(target_language)

        matching_published = []
        for _, published in self.published.items():
            if published._is_published(source_language, target_language):
                matching_published.append(published)
        return matching_published


# Rebuild model to resolve forward references
ChallengeRecord.model_rebuild()
