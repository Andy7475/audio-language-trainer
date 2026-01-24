from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer
from typing import List, Dict, Optional
from src.story import Story, PublishedStory
from src.challenges.models import PublishedChallenge
from src.models import get_language, BCP47Language
from langcodes import Language
from src.logger import logger
from collections import defaultdict
from src.utils import render_html_content
from src.storage import PUBLIC_BUCKET, get_public_url_from_gcs_stub, upload_to_gcs
from src.story import get_all_stories
from src.challenges.models import get_challenge
from urllib.parse import quote_plus


class IndexPage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    template_name: str

    @property
    def base_prefix(self) -> str:
        return "stories"

    def _render_index_html(self) -> str:
        return render_html_content(self.model_dump(), self.template_name)

    def upload_html(self) -> str:
        html_content = self._render_index_html()
        gcs_path = upload_to_gcs(
            obj=html_content,
            bucket_name=PUBLIC_BUCKET,
            base_prefix=self.base_prefix,
            file_name="index.html",
            content_type="text/html",
        )
        public_url = get_public_url_from_gcs_stub(gcs_path)
        logger.info(f"Published index at {public_url}")
        return public_url


class LanguageIndex(IndexPage):
    source_languages: List[BCP47Language] = Field(...)
    template_name: str = "language_index.html"
    all_stories: List[Story] = Field(default_factory=list)

    @property
    def base_prefix(self) -> str:
        return "stories"

    @field_serializer("source_languages")
    def serialize_source_languages(
        self, source_languages: List[BCP47Language]
    ) -> List[tuple[str, str]]:
        return [(lang.to_tag(), lang.display_name()) for lang in source_languages]


def create_language_index() -> LanguageIndex:
    all_stories = get_all_stories()
    ALL_SOURCE_LANGUAGES = set(
        tag
        for story in all_stories
        for tag in story.get_all_published_source_languages()
    )
    return LanguageIndex(source_languages=ALL_SOURCE_LANGUAGES, all_stories=all_stories)


class StoryWithChallenge(BaseModel):
    """Container for a published story and its optional challenge."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    story: PublishedStory = Field(..., description="The published story")
    challenge: Optional[PublishedChallenge] = Field(
        None, description="The published challenge if it exists"
    )

    @computed_field
    @property
    def story_deck(self) -> str:
        """Get the deck name from the story."""
        return self.story.deck

    @computed_field
    @property
    def story_title(self) -> str:
        """Get the title from the story."""
        return self.story.title

    @computed_field
    @property
    def story_public_url(self) -> str:
        """Get the public URL from the story."""
        return self.story.public_url

    @computed_field
    @property
    def has_challenge(self) -> bool:
        """Check if a challenge exists."""
        return self.challenge is not None

    @computed_field
    @property
    def challenge_public_url(self) -> str:
        """Get the challenge public URL if it exists."""
        return self.challenge.public_url if self.challenge else ""

    @computed_field
    @property
    def deck_url(self) -> str:
        """Generate the shop URL for this deck.

        Returns URL like:
        https://firephrase.co.uk/collections/all?filter.p.m.custom.target_language=Italian%20%28Italy%29&...
        """
        # URL-encode the display names to handle special characters and spaces
        target_lang_encoded = quote_plus(self.story.target_language.display_name())
        source_lang_encoded = quote_plus(self.story.source_language.display_name())
        collection_encoded = (
            quote_plus(self.story.collection) if self.story.collection else ""
        )

        base_url = "https://firephrase.co.uk/collections/all"
        params = [
            "filter.v.price.gte=",
            "filter.v.price.lte=",
            f"filter.p.m.custom.target_language={target_lang_encoded}",
            f"filter.p.m.custom.source_language={source_lang_encoded}",
        ]

        if collection_encoded:
            params.append(f"filter.p.m.custom.collection={collection_encoded}")

        params.append("sort_by=title-ascending")

        return f"{base_url}?{'&'.join(params)}"


class TargetLanguageIndex(IndexPage):
    source_language: BCP47Language = Field(...)
    target_language: BCP47Language = Field(...)
    stories_with_challenges: List[StoryWithChallenge] = Field(
        default_factory=list, description="Stories with their associated challenges"
    )
    template_name: str = "target_language_index.html"

    @property
    def base_prefix(self) -> str:
        return (
            f"stories/{self.source_language.to_tag()}/{self.target_language.to_tag()}"
        )

    @computed_field
    @property
    def source_language_name(self) -> str:
        return self.source_language.display_name()

    @computed_field
    @property
    def target_language_name(self) -> str:
        return self.target_language.display_name()

    @computed_field
    @property
    def all_collections(self) -> List[str]:
        """A list of all collections for this target language"""
        all_collections = set()
        for item in self.stories_with_challenges:
            all_collections.add(item.story.collection)
        return sorted(list(all_collections))

    @computed_field
    @property
    def stories_by_collection(self) -> Dict[str, List[StoryWithChallenge]]:
        """Group stories with challenges by collection."""
        stories_by_collection = defaultdict(list)
        for collection in self.all_collections:
            stories_by_collection[collection] = []
            for item in self.stories_with_challenges:
                if collection == item.story.collection:
                    stories_by_collection[collection].append(item)

        return dict(
            sorted(
                stories_by_collection.items(),
                key=lambda x: (x[0], sorted(x[1], key=lambda y: y.story.deck)),
            )
        )


TargetLanguageIndex.model_rebuild()


def create_target_language_index(
    stories: List[Story],
    source_language: Language | str,
    target_language: Language | str,
) -> TargetLanguageIndex:
    """Create the target language index with stories and challenges."""
    source_language = get_language(source_language)
    target_language = get_language(target_language)

    stories_with_challenges = []

    for story in stories:
        # Get published stories for this language pair
        published_stories = story.get_published_stories(
            source_language, target_language
        )

        for published_story in published_stories:
            # Try to get the challenge for this story
            challenge_record = get_challenge(story.key)
            published_challenge = None

            if challenge_record:
                # Check if challenge is published for this language pair
                matching_challenges = challenge_record.get_published_challenges(
                    source_language=source_language, target_language=target_language
                )
                published_challenge = (
                    matching_challenges[0] if matching_challenges else None
                )

            stories_with_challenges.append(
                StoryWithChallenge(story=published_story, challenge=published_challenge)
            )

    logger.info(
        f"Found {len(stories_with_challenges)} stories (with {sum(1 for s in stories_with_challenges if s.challenge)} challenges) "
        f"for {source_language} -> {target_language}"
    )

    return TargetLanguageIndex(
        source_language=source_language,
        target_language=target_language,
        stories_with_challenges=stories_with_challenges,
    )


class SourceLanguageIndex(IndexPage):
    """From a single source language to multiple target languages"""

    template_name: str = "source_language_index.html"
    published_stories: List[PublishedStory] = Field(default_factory=list)
    source_language: BCP47Language = Field(...)

    @property
    def base_prefix(self) -> str:
        return f"stories/{self.source_language.to_tag()}"

    @computed_field
    @property
    def source_language_name(self) -> str:
        return self.source_language.display_name()

    @computed_field
    @property
    def target_languages(self) -> List[tuple[str, str]]:
        return list(
            {
                (
                    published_story.target_language.to_tag(),
                    published_story.target_language.display_name(),
                )
                for published_story in self.published_stories
            }
        )


def create_source_language_index(
    stories: List[Story], source_language: Language | str
) -> SourceLanguageIndex:
    """
    Gets a dictionary of all data needed to populate the source_language_index_template
    This lists target languages available from the source language

    Returns:
        Dict: with all bits needed for template

    """
    source_language = get_language(source_language)

    ALL_PUBLISHED = []
    for story in stories:
        ALL_PUBLISHED.extend(story.get_published_stories(source_language))
    logger.info(f"Adding {len(ALL_PUBLISHED)} stories for {source_language}")
    logger.debug(f"Stories to add {len(ALL_PUBLISHED)} and they are {ALL_PUBLISHED}")
    return SourceLanguageIndex(
        source_language=source_language, published_stories=ALL_PUBLISHED
    )


def update_indexes() -> None:
    """Update all indexes for all published stories and challenges"""

    lang_index = create_language_index()
    lang_index.upload_html()

    for source_lang in lang_index.source_languages:
        logger.info(
            f"Updating index for {source_lang} with {len(lang_index.all_stories)} stories"
        )
        source_lang_index = create_source_language_index(
            lang_index.all_stories, source_lang
        )
        source_lang_index.upload_html()

        for target_lang in source_lang_index.target_languages:
            logger.info(f"Updating index for {source_lang} -> {target_lang}")
            target_lang_index = create_target_language_index(
                lang_index.all_stories, source_lang, target_lang[0]
            )
            target_lang_index.upload_html()


if __name__ == "__main__":
    update_indexes()
