"""
Snippet to migrate existing Firestore translation documents by appending
their parent phrase's collection and deck as tags.
"""

from google.cloud import firestore
from connections.gcloud_auth import get_firestore_client


def migrate_tags(database_name: str = "firephrases", dry_run: bool = True):
    db = get_firestore_client(database_name)
    phrases_ref = db.collection("phrases")

    print(f"Starting tag migration (Dry Run: {dry_run})...")

    count = 0
    updated_translations = 0

    # Stream all phrases
    for phrase_doc in phrases_ref.stream():
        phrase_data = phrase_doc.to_dict()

        # Get collection and deck, format them safely as tags
        collection = phrase_data.get("collection")
        deck = phrase_data.get("deck")

        tags_to_add = []
        if collection:
            tags_to_add.append(collection.replace(" ", "_"))
        if deck:
            tags_to_add.append(deck.replace(" ", "_"))

        if not tags_to_add:
            continue

        count += 1

        # Stream all translation documents for this phrase
        for t_doc in phrase_doc.reference.collection("translations").stream():
            if not dry_run:
                # Use ArrayUnion to safely append without overwriting existing
                # tags or creating duplicates
                t_doc.reference.update({"tags": firestore.ArrayUnion(tags_to_add)})
            updated_translations += 1

        if count % 100 == 0:
            print(f"Processed {count} phrases...")

    print(
        f"Done! Processed {count} phrases, updating {updated_translations} translation docs."
    )


if __name__ == "__main__":
    # Set dry_run=False to actually perform the Firestore updates
    migrate_tags(dry_run=False)
