"""Sync tagged Firestore phrases into a live local Anki collection.

Complements anki_tools.py (which only ever produces standalone .apkg files
via genanki): this module reads/writes the live collection.anki2 directly
via the real `anki` package, so Firestore's tags can be pushed onto notes
that already exist in the user's Anki collection, and missing notes can be
added directly - matching the workflow in phrases/search.py
(add_tags_from_text -> find_phrases_by_tag -> this).

Note identity: every note has two distinct identifiers, and this module
only ever generates one of them.
- note.id (NoteId) is Anki's own permanent internal identity, assigned
  automatically the instant a note is created. This is what cards.nid
  actually references, so it's what review history is anchored to - never
  set here.
- note.guid is a plain text bookkeeping field this module computes
  (generate_note_guid), purely so a script run days later can relocate
  "the note for phrase X in language pair Y" without persisting anything
  itself - guid is fully reconstructable at any time from just
  phrase.key + source_tag + target_tag.
Every write to an existing note here is: col.get_note(note_id) (load by
Anki's real id) -> mutate only .tags and/or .guid -> col.update_note(note).
note.id itself never changes across this, so cards/revlog are untouched.
"""

import os
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Dict, List, Literal, Optional, Tuple

from anki.collection import Collection
from anki.decks import DeckId
from anki.models import NotetypeDict, NotetypeId
from anki.notes import NoteId
from anki.utils import split_fields
from langcodes import Language
from tqdm import tqdm

from anki_tools import prepare_phrase_note_content
from logger import logger
from models import get_language
from phrases.phrase_model import Phrase
from phrases.search import find_phrases_by_tag
from phrases.utils import generate_note_guid, to_anki_tags, ANKI_MANAGED_TAG_PREFIX


def build_content_index(
    col: Collection, notetype_id: NotetypeId
) -> Dict[Tuple[str, str], NoteId]:
    """Build a (SourceText, TargetText) -> note_id index for one notetype.

    Built once per sync run (not per phrase), so the content-fallback match
    in find_existing_note is an O(1) lookup per phrase instead of an O(n)
    scan.

    Args:
        col: Open Anki collection.
        notetype_id: Notetype to index notes for.

    Returns:
        Dict mapping (SourceText, TargetText) to note id. If more than one
        note shares the same pair (e.g. a leftover duplicate), the last one
        seen wins and it's logged.
    """
    notetype = col.models.get(notetype_id)
    field_map = col.models.field_map(notetype)
    source_ord = field_map["SourceText"][0]
    target_ord = field_map["TargetText"][0]

    index: Dict[Tuple[str, str], NoteId] = {}
    rows = col.db.all("select id, flds from notes where mid = ?", notetype_id)
    for note_id, flds in rows:
        fields = split_fields(flds)
        key = (fields[source_ord], fields[target_ord])
        if key in index:
            logger.info(
                f"Duplicate (SourceText, TargetText) while indexing notes: "
                f"{key} -> note ids {index[key]} and {note_id}"
            )
        index[key] = NoteId(note_id)
    return index


@dataclass
class FoundNote:
    note_id: NoteId
    matched_by: Literal["guid", "content"]
    guid_needs_healing: bool


def find_existing_note(
    col: Collection,
    phrase: Phrase,
    source_language: Language | str,
    target_language: Language | str,
    notetype_id: NotetypeId,
    content_index: Optional[Dict[Tuple[str, str], NoteId]] = None,
) -> Optional[FoundNote]:
    """Find the Anki note (if any) for a phrase/language pair.

    Purely read-only - no writes happen here, so dry_run reporting stays
    accurate; the caller decides whether/how to act on guid_needs_healing.

    Lookup order:
        1. Exact guid match (fast path; matches notes this sync already
           touched, or that were created via anki_tools with the same
           deterministic generate_note_guid).
        2. Exact (SourceText, TargetText) match via content_index (fallback,
           for notes that predate generate_note_guid and so carry an old,
           unrecoverable random guid) -> guid_needs_healing=True.

    Args:
        col: Open Anki collection.
        phrase: Phrase to look up.
        source_language: BCP47 source language.
        target_language: BCP47 target language.
        notetype_id: Restrict matching to notes of this notetype.
        content_index: Pre-built index from build_content_index. If None,
            content-fallback matching is skipped (guid-only lookup).

    Returns:
        FoundNote if a match was found, else None.
    """
    source_tag = get_language(source_language).to_tag()
    target_tag = get_language(target_language).to_tag()

    guid = str(generate_note_guid(phrase.key, source_tag, target_tag))
    note_id = col.db.scalar(
        "select id from notes where guid = ? and mid = ?", guid, notetype_id
    )
    if note_id is not None:
        return FoundNote(
            note_id=NoteId(note_id), matched_by="guid", guid_needs_healing=False
        )

    if not content_index:
        return None
    if source_tag not in phrase.translations or target_tag not in phrase.translations:
        return None

    key = (phrase.translations[source_tag].text, phrase.translations[target_tag].text)
    note_id = content_index.get(key)
    if note_id is None:
        return None

    return FoundNote(note_id=note_id, matched_by="content", guid_needs_healing=True)


def create_note_in_collection(
    col: Collection,
    phrase: Phrase,
    source_language: Language | str,
    target_language: Language | str,
    deck_id: DeckId,
    notetype: NotetypeDict,
    index: int = 0,
) -> NoteId:
    """Create a new Anki note directly in the live collection for a phrase.

    Reuses anki_tools.prepare_phrase_note_content for translation backfill
    and media handling - the same logic the genanki .apkg export path uses -
    so the two paths can't drift apart. Tags are deliberately left empty
    here; sync_note_tags is the single place tag policy is decided, run
    uniformly for every phrase whether new or pre-existing.

    Args:
        col: Open Anki collection.
        phrase: Phrase to create a note for.
        source_language: BCP47 source language.
        target_language: BCP47 target language.
        deck_id: Deck to file the new note under.
        notetype: The notetype to use (e.g. from col.models.by_name(...)).
        index: Position hint used for the SortOrder field.

    Returns:
        NoteId: The id Anki assigned to the newly created note.
    """
    source_tag = get_language(source_language).to_tag()
    target_tag = get_language(target_language).to_tag()

    with TemporaryDirectory() as temp_dir:
        fields, media_files = prepare_phrase_note_content(
            phrase, source_language, target_language, index, temp_dir
        )

        # Copy media into the collection's media folder. Anki renames on a
        # filename collision, so substitute the returned name into any
        # field value that referenced the original temp filename.
        for media_path in media_files:
            original_name = os.path.basename(media_path)
            final_name = col.media.add_file(media_path)
            if final_name != original_name:
                for field_name, value in fields.items():
                    if original_name in value:
                        fields[field_name] = value.replace(original_name, final_name)

        note = col.new_note(notetype)
        for field_name, value in fields.items():
            note[field_name] = value
        note.tags = []
        note.guid = str(generate_note_guid(phrase.key, source_tag, target_tag))

        col.add_note(note, deck_id)
        return note.id


def sync_note_tags(note, firestore_tags: List[str]) -> bool:
    """Overwrite exactly the fs::-prefixed tags on a note to match Firestore.

    Tags not starting with ANKI_MANAGED_TAG_PREFIX (Anki-native tags like
    'marked'/'leech', or anything hand-added) are left completely untouched
    - this is what makes "Firestore overwrites Anki tags" safe to run
    repeatedly against a live personal collection.

    Args:
        note: An Anki note-like object exposing a mutable `.tags` list
            (works with a real anki.notes.Note or a simple test stub, since
            this never calls col.update_note() itself - persistence stays
            the caller's job so dry_run reporting stays accurate).
        firestore_tags: The phrase translation's tags as stored in
            Firestore (unprefixed).

    Returns:
        bool: True if note.tags would change (compared as a set - tag order
            is not meaningful).
    """
    kept = [tag for tag in note.tags if not tag.startswith(ANKI_MANAGED_TAG_PREFIX)]
    managed = to_anki_tags(firestore_tags)
    new_tags = kept + [tag for tag in managed if tag not in kept]

    changed = set(new_tags) != set(note.tags)
    note.tags = new_tags
    return changed


@dataclass
class SyncResult:
    phrase_key: str
    created: bool = False
    guid_healed: bool = False
    tags_changed: bool = False
    error: Optional[str] = None


@dataclass
class SyncReport:
    results: List[SyncResult] = field(default_factory=list)
    dry_run: bool = True

    def summary(self) -> str:
        would = "would be " if self.dry_run else ""
        created = sum(1 for r in self.results if r.created)
        healed = sum(1 for r in self.results if r.guid_healed)
        tagged = sum(1 for r in self.results if r.tags_changed)
        errors = [r for r in self.results if r.error]

        lines = [
            f"{'[DRY RUN] ' if self.dry_run else ''}"
            f"Synced {len(self.results)} phrase(s):",
            f"  {created} note(s) {would}created",
            f"  {healed} guid(s) {would}healed",
            f"  {tagged} note(s) {would}have tags updated",
        ]
        if errors:
            lines.append(f"  {len(errors)} error(s):")
            for r in errors:
                lines.append(f"    {r.phrase_key}: {r.error}")
        return "\n".join(lines)


def sync_phrases_to_anki(
    col: Collection,
    phrases: List[Phrase],
    source_language: Language | str,
    target_language: Language | str,
    notetype_name: str = "FirePhrase2",
    deck_name: str = "FirePhrase",
    dry_run: bool = True,
) -> SyncReport:
    """Ensure each phrase exists as an Anki note with tags matching Firestore.

    For each phrase: finds an existing note (by guid, falling back to exact
    content match and healing the guid in place if found that way), creates
    one if missing, then overwrites its fs::-prefixed tags via
    sync_note_tags. One phrase erroring doesn't stop the run - it's recorded
    on that phrase's SyncResult and the loop continues, matching
    anki_tools.create_anki_deck's existing per-phrase error handling.

    Args:
        col: Open Anki collection. Lifecycle (open/backup/close) is the
            caller's responsibility, not this function's, so a caller
            syncing several tags in one run can share one open collection.
        phrases: Phrases to sync. For content-fallback matching and tag
            sync to work, these should already have the target (and
            ideally source) translation populated - e.g. via
            find_phrases_by_tag.
        source_language: BCP47 source language (what the user knows).
        target_language: BCP47 target language (what the user is learning).
        notetype_name: Anki notetype to use; must already exist in the
            collection (see Raises).
        deck_name: Deck newly created notes are filed under. Anki's native
            Filtered Decks are the intended way to build tag-based study
            sets on top of this - this project doesn't compute deck names
            from tags.
        dry_run: If True (default), makes no writes anywhere - not to Anki,
            not to Firestore. Phrases that would be newly created are only
            reported as "would create"; prepare_phrase_note_content's
            translation-backfill-and-upload side effect never runs for
            them, so a dry run can't preview the exact fields such a note
            would get.

    Returns:
        SyncReport summarizing what happened (or would happen).

    Raises:
        RuntimeError: If notetype_name doesn't already exist in the
            collection. Fails fast rather than constructing a matching
            notetype from scratch, which risks drifting from the
            genanki-side definition in anki_tools.get_anki_model() - import
            one .apkg via the existing export path first to establish it.
    """
    notetype = col.models.by_name(notetype_name)
    if notetype is None:
        raise RuntimeError(
            f"Notetype '{notetype_name}' not found in the Anki collection. "
            "Create/import a deck via anki_tools.create_anki_deck() at "
            "least once first to establish it."
        )

    deck_id = col.decks.id_for_name(deck_name)
    if deck_id is None and not dry_run:
        deck_id = col.decks.add_normal_deck_with_name(deck_name).id

    content_index = build_content_index(col, notetype["id"])
    target_tag = get_language(target_language).to_tag()

    results: List[SyncResult] = []

    for index, phrase in enumerate(tqdm(phrases, desc="Syncing phrases to Anki")):
        result = SyncResult(phrase_key=phrase.key)
        try:
            found = find_existing_note(
                col,
                phrase,
                source_language,
                target_language,
                notetype["id"],
                content_index,
            )

            if found is None:
                result.created = True
                if not dry_run:
                    new_note_id = create_note_in_collection(
                        col,
                        phrase,
                        source_language,
                        target_language,
                        deck_id,
                        notetype,
                        index,
                    )
                    found = FoundNote(
                        note_id=new_note_id, matched_by="guid", guid_needs_healing=False
                    )

            if found is not None:
                if found.guid_needs_healing:
                    result.guid_healed = True
                    if not dry_run:
                        note = col.get_note(found.note_id)
                        note.guid = str(
                            generate_note_guid(
                                phrase.key,
                                get_language(source_language).to_tag(),
                                target_tag,
                            )
                        )
                        col.update_note(note)

                if target_tag in phrase.translations:
                    note = col.get_note(found.note_id)
                    changed = sync_note_tags(note, phrase.translations[target_tag].tags)
                    if changed:
                        result.tags_changed = True
                        if not dry_run:
                            col.update_note(note)

        except Exception as e:
            result.error = str(e)
            logger.info(f"Error syncing phrase {phrase.key}: {e}")

        results.append(result)

    return SyncReport(results=results, dry_run=dry_run)


def sync_tag_to_anki(
    col: Collection,
    tag: str,
    source_language: Language | str,
    target_language: Language | str,
    collection: Optional[str] = None,
    deck: Optional[str] = None,
    database_name: str = "firephrases",
    notetype_name: str = "FirePhrase2",
    deck_name: str = "FirePhrase",
    dry_run: bool = True,
) -> SyncReport:
    """Sync every Firestore phrase tagged `tag` into the live Anki collection.

    Thin wrapper: find_phrases_by_tag(...) then sync_phrases_to_anki(...) -
    the "given a Firestore tag" entry point matching the tag-driven workflow
    in phrases/search.py (add_tags_from_text -> find_phrases_by_tag ->
    this).

    Args:
        col: Open Anki collection.
        tag: The Firestore tag to sync (e.g. 'media::film::bron').
        source_language: BCP47 source language (what the user knows).
        target_language: BCP47 target language (what the user is learning).
        collection: Optional Firestore collection filter.
        deck: Optional Firestore deck filter.
        database_name: Firestore database name.
        notetype_name: Anki notetype to use; must already exist.
        deck_name: Anki deck new notes are filed under.
        dry_run: If True (default), makes no writes anywhere.

    Returns:
        SyncReport summarizing what happened (or would happen).
    """
    phrases = find_phrases_by_tag(
        tag,
        target_language,
        collection=collection,
        deck=deck,
        database_name=database_name,
    )
    return sync_phrases_to_anki(
        col,
        phrases,
        source_language,
        target_language,
        notetype_name=notetype_name,
        deck_name=deck_name,
        dry_run=dry_run,
    )
