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
SV_TAG   = "sv-SE"         # the exact Firestore document key for Swedish translations
OVERWRITE = False          # set True to recompute even if verbs already exist
DRY_RUN   = False          # set True to print what would change without writing

# %% --- imports ----------------------------------------------------------
from src.connections.gcloud_auth import get_firestore_client
from src.nlp import get_verbs_and_vocab

db = get_firestore_client("firephrases")

# %% --- helpers ----------------------------------------------------------

def _needs_update(translation_data: dict) -> bool:
    """Return True if verbs or vocab are absent / empty."""
    verbs = translation_data.get("verbs", [])
    vocab = translation_data.get("vocab", [])
    return not verbs and not vocab


# %% --- main loop --------------------------------------------------------

phrase_docs    = list(db.collection("phrases").stream())
total_phrases  = len(phrase_docs)

updated   = 0
skipped   = 0
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
    text   = t_data.get("text", "")

    if not text:
        print(f"  [{i}/{total_phrases}] {phrase_hash[:12]}… | {SV_TAG} — ⚠️  no text, skipping")
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
    verbs  = result["verbs"]
    vocab  = result["vocab"]

    print(
        f"  [{i}/{total_phrases}] {phrase_hash[:12]}… | {SV_TAG} — \"{text[:50]}\"\n"
        f"    verbs={verbs}\n"
        f"    vocab={vocab}"
    )

    if not DRY_RUN:
        t_ref.update({"verbs": verbs, "vocab": vocab})

    updated += 1

# %% --- summary ----------------------------------------------------------
print("\n" + "=" * 60)
print(f"Done.")
print(f"  Phrases inspected : {total_phrases}")
print(f"  Swedish not found : {not_found}")
print(f"  Updated           : {updated}{'  (DRY RUN — nothing written)' if DRY_RUN else ''}")
print(f"  Skipped (ok)      : {skipped}")
