from pydantic import BaseModel, ConfigDict, Field, computed_field
from typing import List, Dict
from src.story import Story, PublishedStory
from src.models import get_language, BCP47Language
from langcodes import Language
from src.logger import logger
from collections import defaultdict
from src.utils import render_html_content
from src.storage import PUBLIC_BUCKET, get_public_url_from_gcs_stub, upload_to_gcs

def get_source_language_index_prefix(source_language: Language | str)->str:
    """the bucket location"""
    source_language = get_language(source_language)
    return f"stories/{source_language.to_tag()}"

def get_target_language_index_prefix(source_language: Language, target_langauge: Language)->str:

    source_prefix = get_source_language_index_prefix(source_language)
    return f"{source_prefix}/{target_langauge.to_tag()}"

class TargetLanguageIndex(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    source_language: BCP47Language = Field(...)
    target_language: BCP47Language = Field(...)
    published_stories: List[PublishedStory] = Field(default_factory=list) # collection -> story -> List[PublishedStory]

    @property
    def base_prefix(self)->str:
        return f"stories/{self.source_language.to_tag()}/{self.target_language.to_tag()}"
    @computed_field
    @property
    def source_language_name(self)->str:
        return self.source_language.language_name()
    
    @computed_field
    @property
    def target_language_name(self)->str:
        return self.target_language.language_name()

    @computed_field
    @property
    def all_collections(self)->List[str]:
        """A list of all collections for this target language"""
        all_collections = set()
        for published_story in self.published_stories:
            all_collections.add(published_story.collection)
        return sorted(list(all_collections))
    @computed_field
    @property
    def stories_by_collection(self)->Dict[str, List[PublishedStory]]:
        stories_by_collection = defaultdict(list)
        for collection in self.all_collections:
            stories_by_collection[collection] = []
            for published_story in self.published_stories:
                if collection == published_story.collection:
                    stories_by_collection[collection].append(published_story)

        return dict(sorted(stories_by_collection.items(), key=lambda x: (x[0], sorted(x[1], key= lambda y: y.deck))))
    
    def _render_index_html(self)->str:
        return render_html_content(self.model_dump(), "target_language_index.html")

    def upload_html(self)->str:
        html_content = self._render_index_html()
        gcs_path = upload_to_gcs(obj = html_content,
        bucket_name=PUBLIC_BUCKET,
        base_prefix = self.base_prefix,
        file_name = "index.html",
        content_type="text/html")

        return get_public_url_from_gcs_stub(gcs_path)
    
TargetLanguageIndex.model_rebuild()
def create_target_language_index(stories:List[Story], source_langauge:Language | str, target_language:Language | str)->TargetLanguageIndex:
    """Data we need to create the target language index"""
    source_langauge = get_language(source_langauge)
    target_language = get_language(target_language)

    ALL_PUBLISHED = []
    for story in stories:
        ALL_PUBLISHED.extend(story.get_published_stories(source_langauge, target_language))

    logger.debug(f"Stories to add {len(ALL_PUBLISHED)} and they are {ALL_PUBLISHED}")
    return TargetLanguageIndex(
        source_language = source_langauge,
        target_language = target_language,
        published_stories = ALL_PUBLISHED
    )

def upload_source_language_index(html_content:str, source_language: Language | str)->str:
    """The index.html page for the source language"""
    base_prefix = get_source_language_index_prefix(source_language)
    gcs_path = upload_to_gcs(obj = html_content,
    bucket_name=PUBLIC_BUCKET,
    base_prefix = base_prefix,
    file_name = "index.html",
    content_type="text/html")

    return get_public_url_from_gcs_stub(gcs_path)

def render_source_language_index_html(data:dict)->str:
    """The HTML source of the source language index"""
    return render_html_content(data, "source_language_index.html")


def get_source_language_index_dict(stories: List[Story], source_language: Language | str)->dict:
    """
    Gets a dictionary of all data needed to populate the source_language_index_template
    This lists target languages available from the source language

    Returns:
        Dict: with all bits needed for template

        {source_language_tag : en-GB,
        source_language_name : english,
        target_languages : [(fr-FR, French)
                            ]}
    """
    source_language = get_language(source_language)
    source_language_tag = source_language.to_tag()
    data = defaultdict()
    data["source_language_tag"] = source_language_tag
    data["source_language_name"] = source_language.language_name()
    data["target_languages"] = []

    ALL_TARGETS = set()
    for story in stories:
        matching_published = story.get_published_stories(source_language)

        for _published in matching_published:
            if _published._is_published(source_language):
                ALL_TARGETS.add(_published.target_language_tag)
    
    for target_tag in ALL_TARGETS:
        target_name = get_language(target_tag).language_name()
        data["target_languages"].append((target_tag , target_name))

    return data