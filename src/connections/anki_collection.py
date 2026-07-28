"""Local Anki collection connection utilities.

Connects directly to a live Anki Desktop collection (collection.anki2) via
the real `anki` package, as opposed to anki_tools.py, which only ever
produces standalone .apkg files via genanki. Follows the error-handling
rigor of connections/gcloud_auth.py (try/except -> RuntimeError) combined
with the local-file simplicity of connections/wiktionary.py (no cloud auth
needed), per CLAUDE.md's client-connection pattern.
"""

import os
from typing import Optional

from anki.collection import Collection
from dotenv import load_dotenv


# Singleton collection, plus the path it was opened with (unlike a cloud
# client, this connection is parameterized by a path that can legitimately
# change between calls - e.g. tests pointing at a throwaway file - so the
# cache must track which path is actually open, not just whether one is).
_collection: Optional[Collection] = None
_collection_path: Optional[str] = None


def get_anki_collection(path: Optional[str] = None) -> Collection:
    """Get a live Anki Collection instance (singleton per path).

    Opens the real Anki collection.anki2 file directly via the `anki`
    package. Anki Desktop holds an exclusive OS file lock on this file
    while open, so it must be closed before calling this.

    Args:
        path: Path to collection.anki2. Defaults to the ANKI_COLLECTION_PATH
            environment variable if not given.

    Returns:
        Collection: The open Anki collection.

    Raises:
        RuntimeError: If no path is given/configured, the file doesn't
            exist, or opening it fails (e.g. Anki Desktop is holding it open).
    """
    global _collection, _collection_path

    load_dotenv()
    resolved_path = path or os.environ.get("ANKI_COLLECTION_PATH")
    if not resolved_path:
        raise RuntimeError(
            "ANKI_COLLECTION_PATH not found in environment variables and no "
            "path was given"
        )

    if _collection is not None and _collection_path == resolved_path:
        return _collection

    if _collection is not None:
        # A different path was requested - close the stale connection first
        # rather than leaking its file lock.
        close_anki_collection()

    if not os.path.exists(resolved_path):
        raise RuntimeError(f"Anki collection not found at: {resolved_path}")

    try:
        collection = Collection(resolved_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to open Anki collection at {resolved_path}: {e}. "
            "If Anki Desktop is open, close it first - it holds an "
            "exclusive lock on the collection file."
        )

    _collection = collection
    _collection_path = resolved_path
    return _collection


def close_anki_collection() -> None:
    """Close the cached Anki collection and release its file lock.

    Unlike the test-only reset_*()/reset_client() helpers in
    gcloud_auth.py/anthropic_auth.py (safe to skip, since a Firestore/
    Anthropic client holds no OS resource), this is a first-class production
    function: an open Collection holds an exclusive OS file lock that Anki
    Desktop needs back, so every real run must call this (e.g. in a finally
    block) rather than relying on garbage collection.
    """
    global _collection, _collection_path

    if _collection is not None:
        try:
            _collection.close()
        except Exception:
            pass

    _collection = None
    _collection_path = None
