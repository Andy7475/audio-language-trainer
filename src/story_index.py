from pydantic import BaseModel, Field
from typing import List, Dict
from src.story import Story
from src.models import get_language
from langcodes import Language
from src.logger import logger
from collections import defaultdict
from src.utils import render_html_content
from src.storage import PUBLIC_BUCKET, get_public_url_from_gcs_stub, upload_to_gcs

def get_source_language_index_prefix(source_language: Language | str)->str:
    """the bucket location"""
    source_language = get_language(source_language)
    return f"stories/{source_language.to_tag()}"

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
        if not story.published:
            logger.debug(f"No published story for {story.title} with source language {source_language}")
            continue
        for published_key in list(story.published.keys()):
            source_tag, target_tag = published_key.split("|")
            if source_tag == source_language_tag:
                ALL_TARGETS.add(target_tag)
    
    for target_tag in ALL_TARGETS:
        target_name = get_language(target_tag).language_name()
        data["target_languages"].append((target_tag , target_name))

    return data