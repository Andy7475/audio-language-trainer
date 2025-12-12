"""Migration script to transfer old JSON wiktionary cache to Firestore.

This script:
1. Reads the old JSON-based wiktionary cache files from GCS
2. Parses the HTML links to extract token/language/URL information
3. Uploads to the new Firestore wiktionary structure

Usage:
    python scripts/migrate_wiktionary_cache_to_firestore.py --dry-run
    python scripts/migrate_wiktionary_cache_to_firestore.py
"""

import argparse
import re
from datetime import datetime
from typing import Dict, List

from tqdm import tqdm

from src.wiktionary.models import WiktionaryEntry
from src.wiktionary.cache import batch_save_wiktionary_entries
from src.storage import PRIVATE_BUCKET, download_from_gcs


def parse_html_link(html_link: str) -> tuple[str, str | None, str | None]:
    """Parse an HTML wiktionary link to extract URL and anchor.

    Args:
        html_link: HTML like '<a href="https://...#French">word</a>' or just 'word'

    Returns:
        tuple: (token, url, section_anchor) or (token, None, None) if no link
    """
    # Check if it's an HTML link or plain text
    link_match = re.match(r'<a href="([^"]+)"[^>]*>([^<]+)</a>', html_link)

    if not link_match:
        # Plain text token (no wiktionary entry found)
        return html_link.strip(), None, None

    url = link_match.group(1)
    token = link_match.group(2)

    # Split URL and anchor
    if "#" in url:
        base_url, anchor = url.split("#", 1)
        section_anchor = f"#{anchor}"
    else:
        base_url = url
        section_anchor = None

    return token, base_url, section_anchor


def convert_old_cache_to_entries(
    old_cache: Dict[str, str],
    language_code: str,
) -> List[WiktionaryEntry]:
    """Convert old cache format to WiktionaryEntry objects.

    Args:
        old_cache: Dictionary mapping tokens to HTML link strings
        language_code: ISO 639-1 language code (e.g., 'fr', 'de')

    Returns:
        List of WiktionaryEntry objects
    """
    entries = []

    for token_key, html_link in tqdm(old_cache.items(), desc=f"Converting {language_code}"):
        # Clean the token (lowercase for storage)
        token_lower = token_key.lower().strip()

        # Parse the HTML link
        display_token, url, section_anchor = parse_html_link(html_link)

        # Create entry
        entry = WiktionaryEntry(
            token=token_lower,
            language_code=language_code,
            exists=(url is not None),
            url=url,
            section_anchor=section_anchor,
            last_checked=datetime.utcnow(),  # Mark as checked during migration
            lookup_variant=None,  # Unknown from old format
        )

        entries.append(entry)

    return entries


def get_old_cache_paths() -> Dict[str, str]:
    """Get GCS paths for old JSON cache files.

    Returns:
        Dictionary mapping language codes to GCS file paths
    """
    # audio-language-trainer-private-content/resources/french/french_wiktionary_cache.json
    # Common languages - add more as needed
    languages = {
        "en": "collections/LM1000/translations/english_wiktionary_cache.json",
        "fr": "collections/LM1000/translations/french_wiktionary_cache.json",
        "es": "collections/LM1000/translations/spanish_wiktionary_cache.json",
        "de": "collections/LM1000/translations/german_wiktionary_cache.json",
        "it": "collections/LM1000/translations/italian_wiktionary_cache.json",
        "pt": "collections/LM1000/translations/portuguese_wiktionary_cache.json",
        "ja": "collections/LM1000/translations/japanese_wiktionary_cache.json",
        "ko": "collections/LM1000/translations/korean_wiktionary_cache.json",
        "zh": "collections/LM1000/translations/mandarin_chinese_wiktionary_cache.json",
    }

    return languages


def migrate_language_cache(
    language_code: str,
    gcs_path: str,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Migrate a single language's cache from GCS JSON to Firestore.

    Args:
        language_code: ISO 639-1 code (e.g., 'fr')
        gcs_path: GCS path to the old JSON cache file
        dry_run: If True, don't actually save to Firestore

    Returns:
        tuple: (successful_count, total_count)
    """
    print(f"\n{'='*60}")
    print(f"Migrating {language_code.upper()} cache")
    print(f"Source: {gcs_path}")
    print(f"{'='*60}")

    try:
        # Download old cache
        print("Downloading old cache from GCS...")
        old_cache = download_from_gcs(
            bucket_name=PRIVATE_BUCKET,
            file_path=gcs_path,
            expected_type="json",
        )

        if not old_cache:
            print(f"  ⚠️  Empty cache for {language_code}")
            return 0, 0

        print(f"  Found {len(old_cache)} entries")

        # Convert to new format
        print("Converting to WiktionaryEntry format...")
        entries = convert_old_cache_to_entries(old_cache, language_code)

        # Count entries with valid links
        valid_entries = sum(1 for e in entries if e.exists)
        print(f"  Valid entries: {valid_entries}")
        print(f"  Invalid entries: {len(entries) - valid_entries}")

        if dry_run:
            print("  🔵 DRY RUN - Not saving to Firestore")
            # Show a few examples
            print("\n  Example entries:")
            for entry in entries[:3]:
                print(f"    - {entry.token} ({language_code}): exists={entry.exists}")
                if entry.url:
                    print(f"      URL: {entry.url}{entry.section_anchor or ''}")
        else:
            # Save to Firestore
            print("Saving to Firestore...")
            batch_save_wiktionary_entries(entries, database_name="firephrases")
            print(f"  ✅ Saved {len(entries)} entries to Firestore")

        return len(entries), len(entries)

    except FileNotFoundError:
        print(f"  ⚠️  Cache file not found: {gcs_path}")
        return 0, 0
    except Exception as e:
        print(f"  ❌ Error migrating {language_code}: {e}")
        return 0, 0


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate old JSON wiktionary cache to Firestore"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without saving to Firestore",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Specific language codes to migrate (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("WIKTIONARY CACHE MIGRATION")
    print("Old format: JSON files in GCS")
    print("New format: Firestore database 'firephrases'")
    if args.dry_run:
        print("🔵 DRY RUN MODE - No changes will be made")
    print("=" * 60)

    # Get cache paths
    cache_paths = get_old_cache_paths()

    # Filter by requested languages if specified
    if args.languages:
        cache_paths = {
            lang: path
            for lang, path in cache_paths.items()
            if lang in args.languages
        }

    if not cache_paths:
        print("No cache files to migrate!")
        return

    print(f"\nFound {len(cache_paths)} language cache(s) to migrate:")
    for lang in cache_paths.keys():
        print(f"  - {lang}")

    # Migrate each language
    total_migrated = 0
    total_entries = 0

    for language_code, gcs_path in cache_paths.items():
        migrated, total = migrate_language_cache(
            language_code=language_code,
            gcs_path=gcs_path,
            dry_run=args.dry_run,
        )
        total_migrated += migrated
        total_entries += total

    # Summary
    print(f"\n{'='*60}")
    print("MIGRATION SUMMARY")
    print(f"{'='*60}")
    print(f"Languages processed: {len(cache_paths)}")
    print(f"Total entries: {total_entries}")
    if args.dry_run:
        print("🔵 DRY RUN - No changes were made to Firestore")
        print("\nTo perform the actual migration, run without --dry-run")
    else:
        print(f"✅ Successfully migrated {total_migrated} entries to Firestore")
        print("\nNext steps:")
        print("1. Verify entries in Firestore console")
        print("2. Test with: phrase.translations['fr-FR'].get_wiktionary_links()")
        print("3. Update any code that was using old cache files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
