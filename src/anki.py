from random import shuffle
import genanki
import os
import uuid
from typing import Dict, List


def export_to_anki(story_data_dict: Dict[str, Dict], output_dir: str, story_name: str):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create a model for each card type
    listening_model = genanki.Model(
        1607392319,
        "Listening Practice",
        fields=[
            {"name": "TargetAudio"},
            {"name": "TargetText"},
            {"name": "EnglishText"},
        ],
        templates=[
            {
                "name": "Listening Card",
                "qfmt": "{{TargetAudio}}",
                "afmt": '{{FrontSide}}<hr id="answer">{{TargetText}}<br>{{EnglishText}}',
            },
        ],
    )

    reading_model = genanki.Model(
        1607392320,
        "Reading Practice",
        fields=[
            {"name": "TargetText"},
            {"name": "EnglishText"},
            {"name": "TargetAudio"},
        ],
        templates=[
            {
                "name": "Reading Card",
                "qfmt": "{{TargetText}}",
                "afmt": '{{FrontSide}}<hr id="answer">{{EnglishText}}<br>{{TargetAudio}}',
            },
        ],
    )

    speaking_model = genanki.Model(
        1607392321,
        "Speaking Practice",
        fields=[
            {"name": "EnglishText"},
            {"name": "TargetText"},
            {"name": "TargetAudio"},
        ],
        templates=[
            {
                "name": "Speaking Card",
                "qfmt": "{{EnglishText}}",
                "afmt": '{{FrontSide}}<hr id="answer">{{TargetText}}<br>{{TargetAudio}}',
            },
        ],
    )

    # Create a deck for each story part
    # decks = {}
    media_files = []
    notes = []
    deck_id = uuid.uuid4().int & (1 << 31) - 1
    deck = genanki.Deck(deck_id, f"{story_name} - phrases")

    for _, data in story_data_dict.items():

        for (english, target), audio_segments in zip(
            data["translated_phrase_list"], data["translated_phrase_list_audio"]
        ):
            # Generate unique filenames for audio
            target_audio_slow = f"{uuid.uuid4()}.mp3"
            target_audio_normal = f"{uuid.uuid4()}.mp3"

            # Export audio segments
            audio_segments[1].export(
                os.path.join(output_dir, target_audio_slow), format="mp3"
            )
            audio_segments[2].export(
                os.path.join(output_dir, target_audio_normal), format="mp3"
            )

            # Add to media files list
            media_files.extend([target_audio_slow, target_audio_normal])

            # Create notes for each card type
            listening_note = genanki.Note(
                model=listening_model,
                fields=[f"[sound:{target_audio_normal}]", target, english],
            )
            reading_note = genanki.Note(
                model=reading_model,
                fields=[target, english, f"[sound:{target_audio_normal}]"],
            )
            speaking_note = genanki.Note(
                model=speaking_model,
                fields=[english, target, f"[sound:{target_audio_normal}]"],
            )

            notes.extend([listening_note, reading_note, speaking_note])
            # Add notes to the deck

    # shuffle the notes
    shuffle(notes)
    for note in notes:
        deck.add_note(note)
    # Create a package with all decks
    # package = genanki.Package(list(decks.values()))
    package = genanki.Package(deck)
    package.media_files = [os.path.join(output_dir, file) for file in media_files]

    # Write the package to a file
    output_filename = os.path.join(output_dir, f"{story_name}_anki_deck.apkg")
    package.write_to_file(output_filename)

    print(f"Anki deck exported to {output_filename}")
