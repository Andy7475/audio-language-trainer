"""Sync Firestore-tagged phrases into the live local Anki collection.

For a given Firestore tag and source/target language pair: ensures every
matching phrase exists as a note in the live Anki collection (creating it
if missing) and overwrites that note's fs::-prefixed tags to match
Firestore - see src/anki_sync.py for the full mechanics.

Set dry_run=False to actually perform the changes (a real Anki backup is
taken first). Requires Anki Desktop to be closed - it holds an exclusive
lock on the collection file.
"""

import os
from pathlib import Path

from anki_sync import sync_tag_to_anki
from connections.anki_collection import close_anki_collection, get_anki_collection


def sync_tag(
    tag: str,
    source_language: str,
    target_language: str,
    dry_run: bool = True,
) -> None:
    col = get_anki_collection()
    try:
        if not dry_run:
            backup_folder = str(
                Path(os.environ["ANKI_COLLECTION_PATH"]).parent / "backups"
            )
            os.makedirs(backup_folder, exist_ok=True)
            col.create_backup(
                backup_folder=backup_folder, force=True, wait_for_completion=True
            )

        report = sync_tag_to_anki(
            col, tag, source_language, target_language, dry_run=dry_run
        )
        print(report.summary())
    finally:
        close_anki_collection()


if __name__ == "__main__":
    # Set dry_run=False to actually perform the Anki updates
    sync_tag(
        tag="media::film::bron",
        source_language="en-GB",
        target_language="sv-SE",
        dry_run=True,
    )
