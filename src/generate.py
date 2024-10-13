from collections import defaultdict
from typing import Dict
from src.audio_generation import async_process_phrases, generate_audio_from_dialogue
from pydub import AudioSegment
from src.config_loader import config
from src.dialogue_generation import get_vocab_from_dialogue, update_vocab_usage
from src.dialogue_generation import (
    generate_story_plan,
    generate_dialogue_prompt,
    generate_dialogue,
    generate_recap,
)
from src.audio_generation import (
    generate_audio_from_dialogue,
    generate_normal_and_fast_audio,
    async_process_phrases,
)
from src.phrase import generate_practice_phrases_from_dialogue

from src.translation import translate_dialogue, translate_phrases
from src.audio_generation import create_m4a_with_timed_lyrics
from src.dialogue_generation import get_least_used_words, add_usage_to_words


def create_story_plan_and_dialogue(
    story_name: str,
    n_verbs: int = 10,
    n_vocab: int = 30,
    output_dir: str = "../outputs",
) -> Dict:
    """Given a brief story name and a verb and vocab count, will return a dictionary
    containing the plan for the story and dialogue"""
    verbs_for_story = get_least_used_words("verbs", n_verbs)
    vocab_for_story = get_least_used_words("vocab", n_vocab)

    verbs_for_story_usage = add_usage_to_words(verbs_for_story, "verbs")
    vocab_for_story_usage = add_usage_to_words(vocab_for_story, "vocab")

    story_plan = generate_story_plan(
        story_guide=story_name,
        verb_list=verbs_for_story,
        vocab_list=vocab_for_story,
        output_dir=output_dir,
        test=False,
        story_name=story_name,
    )
    if story_plan:
        print(f"Story plan created {str(story_plan)[:20]}")
    else:
        raise Exception("Creation of story plan failed")

    story_data_dict = defaultdict(lambda: defaultdict(str))
    recap = "This is the beginning of the story."
    for story_part in list(story_plan.keys()):
        prompt = generate_dialogue_prompt(
            story_part=story_part,
            story_part_outline=story_plan[story_part],
            last_recap=recap,
            verb_usage_str=verbs_for_story_usage,
            vocab_usage_str=vocab_for_story_usage,
            verb_use_count=5,
            vocab_use_count=10,
            grammar_concept_count=5,
            grammar_use_count=3,
        )
        dialogue = generate_dialogue(prompt)
        vocab_used = get_vocab_from_dialogue(dialogue)
        update_vocab_usage(vocab_used)
        verbs_for_story_usage = add_usage_to_words(verbs_for_story, "verbs")
        vocab_for_story_usage = add_usage_to_words(vocab_for_story, "vocab")
        recap = generate_recap(dialogue, test=False)
        story_data_dict[story_part]["dialogue_generation_prompt"] = prompt
        story_data_dict[story_part]["dialogue"] = dialogue
        story_data_dict[story_part]["recap"] = recap
    return story_data_dict


def add_practice_phrases(story_data_dict):
    """Takes the longer dialgoue and breaks it up into smaller practice phrases.
    Modifies the input dictionary and returns it"""
    for story_part in story_data_dict:
        dialogue = story_data_dict[story_part]["dialogue"]
        story_data_dict[story_part]["corrected_phrase_list"] = (
            generate_practice_phrases_from_dialogue(dialogue)
        )

        print(f"Added practice phrases for {story_part}\n")

    return story_data_dict


def add_translations(story_data_dict):
    """Translates all the phrases and dialogue to the target language"""

    for story_part in story_data_dict:
        print(f"Beginning translation for {story_part}")
        if "dialogue" in story_data_dict[story_part]:
            dialogue = story_data_dict[story_part]["dialogue"]
            translated_dialogue = translate_dialogue(dialogue)
            print(f"Translated dialogue")
            story_data_dict[story_part]["translated_dialogue"] = translated_dialogue
        corrected_phrase_list = story_data_dict[story_part]["corrected_phrase_list"]
        translated_phrase_list = translate_phrases(corrected_phrase_list)

        story_data_dict[story_part]["translated_phrase_list"] = translated_phrase_list

        print(f"Translated phrases\n")

    return story_data_dict


async def add_audio(story_data_dict):
    """Adds text-to-speech for english and target language for all dialogue and
    practice phrases"""
    for story_part in story_data_dict:
        if "translated_dialogue" in story_data_dict[story_part]:
            print(f"Beginning text-to-speech for {story_part}")
            translated_dialogue_audio_segments = generate_audio_from_dialogue(
                story_data_dict[story_part]["translated_dialogue"]
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

            print(f"Text-to-speech for dialogue done")
        # now do phrases asynchronoulsy (still unsure if Google API allows this, not getting huge speed up)
        translated_phrases = story_data_dict[story_part]["translated_phrase_list"]
        tranlsated_phrases_audio = await async_process_phrases(translated_phrases)
        story_data_dict[story_part][
            "translated_phrase_list_audio"
        ] = tranlsated_phrases_audio
        print(f"Text-to-speech for phrases done\n")

    return story_data_dict


def create_album_files(story_data_dict, image_data, output_dir, story_name_clean):
    """Creates and saves M4A files for the story, with album artwork"""

    # get lists and audio clips synced together
    full_audio_list = []
    full_captions_list = []

    # fast dialogue (no text)
    PAUSE_TEXT = "---------"
    THINKING_GAP = AudioSegment.silent(duration=config.THINKING_GAP_MS)
    GAP_BETWEEN_PHRASES = AudioSegment.silent(duration=500)
    # translated dialogue

    TOTAL_TRACKS = (
        len(story_data_dict) + 1
    )  # to account for the full dialogue as a separate track
    ALBUM_NAME = story_name_clean.replace("_", " ")
    TRACK_NUMBER = 0

    for story_part in story_data_dict:
        TRACK_NUMBER += 1  # so we don't start at 0

        audio_list = []
        captions_list = []
        dialogue_list = [
            utterence["text"]
            for utterence in story_data_dict[story_part]["translated_dialogue"]
        ]
        dialogue_audio_list = story_data_dict[story_part]["translated_dialogue_audio"]

        audio_list.append(GAP_BETWEEN_PHRASES)
        captions_list.append(f"{story_part} - First dialogue")

        audio_list.extend(dialogue_audio_list)
        captions_list.extend(dialogue_list)
        # print(f"audio {len(audio_list)} - captions {len(captions_list)}")

        audio_list.append(GAP_BETWEEN_PHRASES)
        captions_list.append(f"{story_part} - Practice phrases")

        for step, phrase in enumerate(
            story_data_dict[story_part]["translated_phrase_list"]
        ):
            english_text = phrase[0]
            target_text = phrase[1]

            english_audio = story_data_dict[story_part]["translated_phrase_list_audio"][
                step
            ][0]
            target_audio_slow = story_data_dict[story_part][
                "translated_phrase_list_audio"
            ][step][1]
            target_audio_normal = story_data_dict[story_part][
                "translated_phrase_list_audio"
            ][step][2]

            audio_list.append(english_audio)
            captions_list.append(english_text)

            audio_list.append(THINKING_GAP)
            captions_list.append(PAUSE_TEXT)

            audio_list.append(target_audio_normal)
            captions_list.append(target_text)

            audio_list.append(GAP_BETWEEN_PHRASES)
            captions_list.append(PAUSE_TEXT)

            audio_list.append(target_audio_slow)
            captions_list.append(target_text)

            audio_list.append(GAP_BETWEEN_PHRASES)
            captions_list.append(PAUSE_TEXT)

        audio_list.append(story_data_dict[story_part]["translated_dialogue_audio_fast"])
        captions_list.append(f"{story_part} - Repeated Fast Dialogue")

        audio_list.append(GAP_BETWEEN_PHRASES)
        captions_list.append(f"{story_part} - Final Dialogue")

        audio_list.extend(dialogue_audio_list)
        captions_list.extend(dialogue_list)

        create_m4a_with_timed_lyrics(
            audio_segments=audio_list,
            phrases=captions_list,
            output_file=f"{output_dir}/{story_name_clean}_{story_part}.m4a",
            album_name=ALBUM_NAME,
            track_title=story_part,
            track_number=TRACK_NUMBER,
            total_tracks=TOTAL_TRACKS,
            image_data=image_data,
        )
        print(f"Saving M4A file track number {TRACK_NUMBER}")
        full_audio_list.extend(audio_list)
        full_captions_list.extend(captions_list)

    all_dialogue_audio = []
    all_dialogue_captions = []

    for story_part in story_data_dict:
        dialogue_list = [
            utterence["text"]
            for utterence in story_data_dict[story_part]["translated_dialogue"]
        ]
        dialogue_audio_list = story_data_dict[story_part]["translated_dialogue_audio"]
        all_dialogue_audio.extend(dialogue_audio_list)
        all_dialogue_captions.extend(dialogue_list)

        all_dialogue_audio.append(GAP_BETWEEN_PHRASES)
        all_dialogue_captions.append(PAUSE_TEXT)

    full_audio_list.extend(all_dialogue_audio)
    full_captions_list.extend(all_dialogue_captions)

    TRACK_NUMBER += 1
    create_m4a_with_timed_lyrics(
        audio_segments=all_dialogue_audio,
        phrases=all_dialogue_captions,
        output_file=f"{output_dir}/{story_name_clean}_full_dialogue.m4a",
        album_name=ALBUM_NAME,
        track_title="Full Dialogue - All episodes",
        track_number=TRACK_NUMBER,
        total_tracks=TOTAL_TRACKS,
        image_data=image_data,
    )
    print(f"Saving M4A file track number {TRACK_NUMBER}")
