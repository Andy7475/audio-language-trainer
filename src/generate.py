from tqdm import tqdm

from src.audio_generation import (  # async_process_phrases,
    generate_audio_from_dialogue,
    generate_normal_and_fast_audio,
    generate_translated_phrase_audio,
)
from src.translation import translate_dialogue, translate_phrases


def add_translations(story_data_dict):
    """Translates all the phrases and dialogue to the target language"""

    for story_part in tqdm(story_data_dict, desc="adding translations"):
        print(f"Beginning translation for {story_part}")
        if "dialogue" in story_data_dict[story_part]:
            dialogue = story_data_dict[story_part]["dialogue"]
            translated_dialogue = translate_dialogue(dialogue)
            print("Translated dialogue")
            story_data_dict[story_part]["translated_dialogue"] = translated_dialogue

        corrected_phrase_list = story_data_dict[story_part].get("corrected_phrase_list")
        if corrected_phrase_list:
            translated_phrase_list = translate_phrases(corrected_phrase_list)

            story_data_dict[story_part][
                "translated_phrase_list"
            ] = translated_phrase_list
            print("Translated phrases\n")

    return story_data_dict


def add_audio(story_data_dict, source_language_audio: bool = False):
    """Adds text-to-speech for english and target language for all dialogue and
    practice phrases"""
    for story_part in tqdm(story_data_dict, desc="adding audio"):
        if "translated_dialogue" in story_data_dict[story_part]:
            print(f"Beginning text-to-speech for {story_part}")
            translated_dialogue_audio_segments = generate_audio_from_dialogue(
                story_data_dict[story_part]["translated_dialogue"],
                config_language="target",
            )
            story_data_dict[story_part][
                "translated_dialogue_audio"
            ] = translated_dialogue_audio_segments
            normal_translated_clip, fast_translated_clips = (
                generate_normal_and_fast_audio(translated_dialogue_audio_segments)
            )
            story_data_dict[story_part][
                "translated_dialogue_audio_fast"
            ] = fast_translated_clips

            print("Text-to-speech for dialogue done")
        # now do phrases asynchronoulsy (still unsure if Google API allows this, not getting huge speed up)
        translated_phrases = story_data_dict[story_part].get("translated_phrase_list")
        if translated_phrases:
            translated_phrases_audio = generate_translated_phrase_audio(
                translated_phrases, source_language_audio
            )
            story_data_dict[story_part][
                "translated_phrase_list_audio"
            ] = translated_phrases_audio
            print("Text-to-speech for phrases done\n")

    return story_data_dict
