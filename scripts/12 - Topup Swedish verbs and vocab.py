# %% [markdown]
# # Top-up Swedish verbs & vocab on existing Translations
#
# Iterates every phrase in Firestore, finds any Swedish translation
# (tag starting with `sv`), runs spaCy NLP to extract `verbs` and `vocab`,
# then does a **partial `.update()`** — leaving audio, tokens, image_file_path
# and everything else completely untouched.
#
# Safe to re-run: documents that already have non-empty `verbs` are skipped
# unless you set `OVERWRITE = True`.

# %% --- config -----------------------------------------------------------
SV_TAG = "sv-SE"  # the exact Firestore document key for Swedish translations
OVERWRITE = False  # set True to recompute even if verbs already exist
DRY_RUN = False  # set True to print what would change without writing

# %% --- imports ----------------------------------------------------------
from connections.gcloud_auth import get_firestore_client
from nlp import get_verbs_and_vocab

db = get_firestore_client("firephrases")

# %% --- helpers ----------------------------------------------------------


def _needs_update(translation_data: dict) -> bool:
    """Return True if verbs or vocab are absent / empty."""
    verbs = translation_data.get("verbs", [])
    vocab = translation_data.get("vocab", [])
    return not verbs and not vocab


# %% --- main loop --------------------------------------------------------

phrase_docs = list(db.collection("phrases").stream())
total_phrases = len(phrase_docs)

updated = 0
skipped = 0
not_found = 0

print(f"Found {total_phrases} phrases to inspect.\n")

for i, phrase_doc in enumerate(phrase_docs, start=1):
    phrase_hash = phrase_doc.id

    # Direct lookup — we know the key is always "sv-SE"
    t_ref = phrase_doc.reference.collection("translations").document(SV_TAG)
    t_doc = t_ref.get()

    if not t_doc.exists:
        not_found += 1
        continue

    t_data = t_doc.to_dict()
    text = t_data.get("text", "")

    if not text:
        print(
            f"  [{i}/{total_phrases}] {phrase_hash[:12]}… | {SV_TAG} — ⚠️  no text, skipping"
        )
        skipped += 1
        continue

    if not OVERWRITE and not _needs_update(t_data):
        existing_v = t_data.get("verbs", [])
        existing_w = t_data.get("vocab", [])
        print(
            f"  [{i}/{total_phrases}] {phrase_hash[:12]}… | {SV_TAG} — ✓ already has "
            f"{len(existing_v)} verbs, {len(existing_w)} vocab — skipping"
        )
        skipped += 1
        continue

    # Run spaCy NLP (sv model loaded/cached on first call)
    result = get_verbs_and_vocab([text], "sv")
    verbs = result["verbs"]
    vocab = result["vocab"]

    print(
        f'  [{i}/{total_phrases}] {phrase_hash[:12]}… | {SV_TAG} — "{text[:50]}"\n'
        f"    verbs={verbs}\n"
        f"    vocab={vocab}"
    )

    if not DRY_RUN:
        t_ref.update({"verbs": verbs, "vocab": vocab})

    updated += 1

# %% --- summary ----------------------------------------------------------
print("\n" + "=" * 60)
print("Done.")
print(f"  Phrases inspected : {total_phrases}")
print(f"  Swedish not found : {not_found}")
print(
    f"  Updated           : {updated}{'  (DRY RUN — nothing written)' if DRY_RUN else ''}"
)
print(f"  Skipped (ok)      : {skipped}")

# %% [markdown]
# # Demo: tag a phrase from text, then sync the tag to Anki
#
# Walks through the real end-to-end workflow with one concrete example:
#
# 1. `add_tags_from_text` — from a piece of Swedish text, find the minimum
#    set of *existing* phrases whose Swedish translation already covers its
#    vocab, and tag those phrases in Firestore (tag stored unprefixed).
# 2. Confirm the tag landed on the phrase's `sv-SE` translation in Firestore.
# 3. `sync_tag_to_anki` — sync that tag into the live Anki collection
#    (`dry_run=True` first). It should show up on the note as
#    `fs::food_and_drink` — the `fs::` prefix is added only at this step,
#    Firestore itself keeps the bare `food_and_drink`.
# 4. Only once you're happy with the dry-run report: flip to a real write.

# %% --- demo config --------------------------------------------------------
DEMO_TEXT = "en flaska vin"  # Swedish for "a bottle of wine"
DEMO_LANGUAGE = "sv-SE"
DEMO_TAG = "food_and_drink"
DEMO_SOURCE_LANGUAGE = "en-GB"

# %% --- demo imports --------------------------------------------------------
from phrases.search import add_tags_from_text
from connections.anki_collection import get_anki_collection, close_anki_collection
from anki_sync import sync_tag_to_anki

# %% --- 1. tag the covering phrase(s) in Firestore --------------------------
tagged_phrases, demo_missing = add_tags_from_text(DEMO_TEXT, DEMO_LANGUAGE, DEMO_TAG)

print(f"\nTagged {len(tagged_phrases)} phrase(s):")
for p in tagged_phrases:
    sv_text = p.translations[DEMO_LANGUAGE].text
    print(f"  {p.key} | en: {p.english!r} | sv-SE: {sv_text!r}")
print(f"Missing vocab (no covering phrase found): {demo_missing}")

# %% --- 2. confirm the tag in Firestore, unprefixed --------------------------
for p in tagged_phrases:
    print(p.key, "->", p.translations[DEMO_LANGUAGE].tags)

# %% --- 3. sync the tag into the live Anki collection (dry run) -------------
demo_col = get_anki_collection()
try:
    demo_report = sync_tag_to_anki(
        demo_col, DEMO_TAG, DEMO_SOURCE_LANGUAGE, DEMO_LANGUAGE, dry_run=True
    )
    print(demo_report.summary())
    for r in demo_report.results:
        print(" ", r)
finally:
    close_anki_collection()

# %% [markdown]
# ### 4. Run for real — only after reviewing the dry-run report above
#
# Close Anki Desktop first (it holds an exclusive lock on the collection
# file). Takes a real Anki backup before writing, same as
# `scripts/sync_anki_tags.py`.

# %% --- 4. real write (run manually when ready) -----------------------------
import os
from pathlib import Path

demo_col = get_anki_collection()
try:
    backup_folder = str(Path(os.environ["ANKI_COLLECTION_PATH"]).parent / "backups")
    os.makedirs(backup_folder, exist_ok=True)
    demo_col.create_backup(
        backup_folder=backup_folder, force=True, wait_for_completion=True
    )

    demo_report = sync_tag_to_anki(
        demo_col, DEMO_TAG, DEMO_SOURCE_LANGUAGE, DEMO_LANGUAGE, dry_run=False
    )
    print(demo_report.summary())
    for r in demo_report.results:
        print(" ", r)
finally:
    close_anki_collection()
