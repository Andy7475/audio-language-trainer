"""
Migration script to move phrase assets from old storage structure to new phrase-hash-based structure.

This script:
1. Loads phrases from collections/{collection}/phrases.json
2. Maps old hashes (clean_filename) to new hashes (generate_phrase_hash)
3. Generates migration manifest
4. Uses gcloud storage to move files
5. Verifies migration success
6. Generates detailed report

Usage:
    python scripts/migrate_phrase_assets.py --collection WarmUp150 --dry-run
    python scripts/migrate_phrase_assets.py --collection WarmUp150 --execute
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib

from google.cloud import storage
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

PRIVATE_BUCKET = "audio-language-trainer-private-content"
COLLECTION = "WarmUp150"

# Language mapping: old language names → BCP47 tags
LANGUAGE_MAPPING = {
    "english": "en-GB",
    "french": "fr-FR",
    "spanish": "es-ES",
    "german": "de-DE",
    "italian": "it-IT",
    "portuguese": "pt-PT",
    "japanese": "ja-JP",
    "chinese": "zh-CN",
    "korean": "ko-KR",
    "russian": "ru-RU",
    "arabic": "ar-SA",
    "turkish": "tr-TR",
    "greek": "el-GR",
    "polish": "pl-PL",
    "dutch": "nl-NL",
    "swedish": "sv-SE",
    "norwegian": "no-NO",
    "danish": "da-DK",
    "finnish": "fi-FI",
    "czech": "cs-CZ",
    "ukrainian": "uk-UA",
}

# Audio contexts and speeds to migrate
AUDIO_SPECS = [
    ("flashcard", "slow"),
    ("flashcard", "normal"),
    ("story", "normal"),
]


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class MigrationFile:
    """Record of a single file to migrate."""

    phrase: str
    language: str
    file_type: str  # "image", "audio"
    context: Optional[str]  # "flashcard", "story", or None for images
    speed: Optional[str]  # "slow", "normal", "fast", or None for images
    old_hash: str
    new_hash: str
    old_path: str
    new_path: str
    exists_in_old: bool = False
    migrated: bool = False
    error: Optional[str] = None


@dataclass
class MigrationStats:
    """Statistics about migration."""

    total_files: int = 0
    files_found: int = 0
    files_migrated: int = 0
    files_failed: int = 0
    size_bytes_migrated: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# ============================================================================
# HASH FUNCTIONS
# ============================================================================


def old_hash(phrase: str) -> str:
    """
    Reconstruct the old clean_filename hash method.

    This is the method used in ARCHIVE/convert.py:
    - Lowercase
    - Remove special characters
    - Replace spaces with underscores
    - Remove double underscores
    - Strip leading/trailing underscores

    Args:
        phrase: The English phrase to hash

    Returns:
        str: The old-style hash (e.g., "she_runs_to_the_store_daily")
    """
    s = phrase.lower()
    # Keep alphanumeric, spaces, and hyphens only
    s = "".join(c if c.isalnum() or c in " -" else "" for c in s)
    # Replace spaces with underscores
    s = s.replace(" ", "_")
    # Remove double underscores
    s = re.sub(r"_+", "_", s)
    # Strip leading/trailing underscores
    return s.strip("_")


def new_hash(phrase: str) -> str:
    """
    Generate the new phrase hash using the current method.

    This matches generate_phrase_hash from src/phrases/utils.py:
    - Create URL-safe slug (first 50 chars)
    - Hash original phrase with SHA-256 (first 6 hex chars)
    - Combine: {slug}_{hash_suffix}

    Args:
        phrase: The English phrase to hash

    Returns:
        str: The new-style hash (e.g., "she_runs_to_the_store_daily_a3f8d2")
    """
    # Hash the ORIGINAL phrase to capture punctuation differences
    hash_suffix = hashlib.sha256(phrase.encode()).hexdigest()[:6]

    # Create URL-safe slug from lowercase version
    normalized = phrase.lower().strip()
    # Keep only alphanumeric and convert spaces to underscores
    slug = "".join(c if c.isalnum() or c == " " else "" for c in normalized)
    slug = slug.replace(" ", "_")[:50]

    return f"{slug}_{hash_suffix}"


# ============================================================================
# PATH GENERATORS
# ============================================================================


def old_audio_path(collection: str, language: str, speed: str, phrase_hash: str) -> str:
    """Generate old-style audio path.

    Old structure: phrases/{language}/audio/{speed}/{hash}.mp3
    (Collections are not in the old path)
    """
    return f"phrases/{language}/audio/{speed}/{phrase_hash}.mp3"


def old_image_path(collection: str, language: str, phrase_hash: str) -> str:
    """Generate old-style image path.

    Old structure: phrases/{language}/images/{hash}.png or phrases/common/images/{hash}.png
    (Collections are not in the old path)
    """
    return f"phrases/{language}/images/{phrase_hash}.png"


def common_image_path(phrase_hash: str) -> str:
    """Generate common images path (fallback location).

    Common location: phrases/common/images/{hash}.png
    """
    return f"phrases/common/images/{phrase_hash}.png"


def new_audio_path(
    language_tag: str, context: str, speed: str, phrase_hash: str
) -> str:
    """Generate new-style audio path."""
    return f"phrases/{language_tag}/audio/{context}/{speed}/{phrase_hash}.mp3"


def new_image_path(phrase_hash: str) -> str:
    """Generate new-style image path. Images always stored with default English tag (en-GB)."""
    return f"phrases/en-GB/images/{phrase_hash}.png"


# ============================================================================
# GCS OPERATIONS
# ============================================================================


class GCSOperations:
    """Handle Google Cloud Storage operations."""

    def __init__(self, bucket_name: str = PRIVATE_BUCKET):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.bucket_name = bucket_name

    def blob_exists(self, blob_path: str) -> bool:
        """Check if a blob exists."""
        blob = self.bucket.blob(blob_path)
        return blob.exists()

    def get_blob_size(self, blob_path: str) -> Optional[int]:
        """Get blob size in bytes, or None if not found."""
        blob = self.bucket.blob(blob_path)
        if blob.exists():
            blob.reload()
            return blob.size
        return None

    def copy_blob(self, source_path: str, dest_path: str) -> bool:
        """Copy blob from source to destination."""
        try:
            source_blob = self.bucket.blob(source_path)
            if not source_blob.exists():
                logger.warning(f"Source blob not found: {source_path}")
                return False

            # Copy the blob
            self.bucket.copy_blob(source_blob, self.bucket, dest_path)
            logger.info(f"Copied: {source_path} → {dest_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy {source_path} to {dest_path}: {e}")
            return False

    def delete_blob(self, blob_path: str) -> bool:
        """Delete a blob."""
        try:
            blob = self.bucket.blob(blob_path)
            blob.delete()
            logger.info(f"Deleted: {blob_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {blob_path}: {e}")
            return False

    def move_blob(self, source_path: str, dest_path: str) -> bool:
        """Move blob from source to destination (copy then delete)."""
        if self.copy_blob(source_path, dest_path):
            return self.delete_blob(source_path)
        return False


# ============================================================================
# MIGRATION LOGIC
# ============================================================================


def load_phrases(collection: str) -> Dict[str, dict]:
    """
    Load phrases from GCS JSON file.

    Handles both list and dict formats:
    - List: [{"phrase": "...", "language": "..."}, ...]
    - Dict: {"phrase_key": {"phrase": "...", "language": "..."}, ...}

    Args:
        collection: Collection name (e.g., "WarmUp150")

    Returns:
        Dict mapping phrase_id to phrase data
    """
    gcs = GCSOperations()

    # Read phrases.json from GCS
    phrases_path = f"collections/{collection}/phrases.json"

    try:
        blob = gcs.bucket.blob(phrases_path)
        if not blob.exists():
            logger.error(f"Phrases file not found: {phrases_path}")
            return {}

        content = blob.download_as_text()
        data = json.loads(content)

        # Handle both list and dict formats
        if isinstance(data, list):
            # Convert list to dict with index as key
            phrases = {str(i): phrase for i, phrase in enumerate(data)}
            logger.info(
                f"Loaded {len(phrases)} phrases (list format) from {phrases_path}"
            )
        elif isinstance(data, dict):
            phrases = data
            logger.info(
                f"Loaded {len(phrases)} phrases (dict format) from {phrases_path}"
            )
        else:
            logger.error(f"Unexpected format for phrases: {type(data)}")
            return {}

        return phrases
    except Exception as e:
        logger.error(f"Failed to load phrases: {e}")
        return {}


def generate_migration_manifest(
    collection: str, phrases: Dict[str, dict], dry_run: bool = True
) -> Tuple[List[MigrationFile], MigrationStats]:
    """
    Generate list of files to migrate.

    Key insight: Audio hashes are based on the ENGLISH phrase, so translations
    of the same English phrase share the same hash but live in different language folders.

    Args:
        collection: Collection name
        phrases: Dictionary of phrases (all English in WarmUp150)
        dry_run: If True, only check existence, don't migrate

    Returns:
        Tuple of (migration_files, stats)
    """
    gcs = GCSOperations()
    migration_files = []
    stats = MigrationStats()

    # Extract English phrases and their hashes
    english_phrases = {}  # phrase_text -> hash
    for phrase_key, phrase_data in phrases.items():
        # Handle both string and dict formats
        if isinstance(phrase_data, str):
            phrase_text = phrase_data
        elif isinstance(phrase_data, dict):
            phrase_text = phrase_data.get("phrase", phrase_key)
        else:
            continue

        old_hash_value = old_hash(phrase_text)
        new_hash_value = new_hash(phrase_text)
        english_phrases[phrase_text] = {
            "old_hash": old_hash_value,
            "new_hash": new_hash_value,
        }

    logger.info(f"Indexed {len(english_phrases)} English phrases")

    # ====== MIGRATE IMAGES ======
    # Images are only in en-GB (from common folder)
    for phrase_text, hashes in english_phrases.items():
        old_hash_value = hashes["old_hash"]
        new_hash_value = hashes["new_hash"]

        # Check both language-specific and common image locations
        old_img_path = old_image_path(collection, "english", old_hash_value)
        common_img_path = common_image_path(old_hash_value)

        # Use whichever path exists (common location is fallback)
        image_exists_old = gcs.blob_exists(old_img_path)
        image_exists_common = gcs.blob_exists(common_img_path)
        actual_old_path = old_img_path if image_exists_old else common_img_path

        new_img_path = new_image_path(new_hash_value)

        migration_file = MigrationFile(
            phrase=phrase_text,
            language="english",
            file_type="image",
            context=None,
            speed=None,
            old_hash=old_hash_value,
            new_hash=new_hash_value,
            old_path=actual_old_path,
            new_path=new_img_path,
            exists_in_old=image_exists_old or image_exists_common,
        )

        if migration_file.exists_in_old:
            stats.files_found += 1
        migration_files.append(migration_file)
        stats.total_files += 1

    # ====== MIGRATE AUDIO FOR ALL LANGUAGES ======
    # Audio uses the SAME ENGLISH HASH for all translations
    # Search in all language folders for audio files with this hash
    for phrase_text, hashes in english_phrases.items():
        old_hash_value = hashes["old_hash"]
        new_hash_value = hashes["new_hash"]

        # Search in all language folders
        for old_language, language_tag in LANGUAGE_MAPPING.items():
            for context, speed in AUDIO_SPECS:
                old_audio_path_value = old_audio_path(
                    collection, old_language, speed, old_hash_value
                )
                new_audio_path_value = new_audio_path(
                    language_tag, context, speed, new_hash_value
                )

                migration_file = MigrationFile(
                    phrase=phrase_text,
                    language=old_language,
                    file_type="audio",
                    context=context,
                    speed=speed,
                    old_hash=old_hash_value,
                    new_hash=new_hash_value,
                    old_path=old_audio_path_value,
                    new_path=new_audio_path_value,
                    exists_in_old=gcs.blob_exists(old_audio_path_value),
                )

                if migration_file.exists_in_old:
                    stats.files_found += 1
                migration_files.append(migration_file)
                stats.total_files += 1

    logger.info(
        f"Generated manifest: {stats.total_files} total files, "
        f"{stats.files_found} found in old locations"
    )
    return migration_files, stats


def execute_migration(
    migration_files: List[MigrationFile], execute: bool = False
) -> MigrationStats:
    """
    Execute migration of files.

    Args:
        migration_files: List of files to migrate
        execute: If True, actually move files. If False, dry-run only.

    Returns:
        Updated migration stats
    """
    gcs = GCSOperations()
    stats = MigrationStats(
        total_files=len(migration_files),
        files_found=sum(1 for f in migration_files if f.exists_in_old),
    )

    for i, mig_file in enumerate(migration_files, 1):
        if not mig_file.exists_in_old:
            logger.debug(
                f"[{i}/{len(migration_files)}] Skipping (not found): {mig_file.old_path}"
            )
            continue

        logger.info(
            f"[{i}/{len(migration_files)}] Migrating {mig_file.file_type} "
            f"({mig_file.context or 'N/A'}/{mig_file.speed or 'N/A'}): "
            f"{mig_file.old_path} → {mig_file.new_path}"
        )

        if execute:
            try:
                # Move file (copy then delete)
                if gcs.move_blob(mig_file.old_path, mig_file.new_path):
                    # Verify
                    if gcs.blob_exists(mig_file.new_path):
                        size = gcs.get_blob_size(mig_file.new_path)
                        mig_file.migrated = True
                        stats.files_migrated += 1
                        stats.size_bytes_migrated += size or 0
                        logger.info(f"✓ Migrated successfully ({size} bytes)")
                    else:
                        mig_file.error = "New file not found after copy"
                        stats.files_failed += 1
                        logger.error("✗ Verification failed: new file not found")
                else:
                    mig_file.error = "Copy failed"
                    stats.files_failed += 1
                    logger.error("✗ Copy operation failed")

            except Exception as e:
                mig_file.error = str(e)
                stats.files_failed += 1
                logger.error(f"✗ Exception: {e}")
        else:
            logger.info("(DRY RUN - no action taken)")

    return stats


# ============================================================================
# REPORTING
# ============================================================================


def generate_report(
    migration_files: List[MigrationFile],
    stats: MigrationStats,
    output_file: Optional[str] = None,
) -> str:
    """
    Generate detailed migration report.

    Args:
        migration_files: List of migration records
        stats: Migration statistics
        output_file: Optional file to save report to

    Returns:
        str: Formatted report
    """
    timestamp = datetime.now().isoformat()

    report = f"""
{'='*80}
PHRASE ASSET MIGRATION REPORT
{'='*80}
Generated: {timestamp}

SUMMARY
-------
Total files processed:     {stats.total_files}
Files found in old paths:  {stats.files_found}
Files migrated:            {stats.files_migrated}
Files failed:              {stats.files_failed}
Total size migrated:       {stats.size_bytes_migrated / (1024**2):.2f} MB

DETAILS
-------
"""

    # Group by phrase
    by_phrase = {}
    for mf in migration_files:
        if mf.phrase not in by_phrase:
            by_phrase[mf.phrase] = []
        by_phrase[mf.phrase].append(mf)

    for phrase, files in sorted(by_phrase.items()):
        report += f"\nPhrase: {phrase}\n"
        report += f"  Old hash: {files[0].old_hash}\n"
        report += f"  New hash: {files[0].new_hash}\n"
        for mf in files:
            status = "✓" if mf.migrated else "✗" if mf.error else "-"
            file_type_str = (
                f"{mf.file_type}({mf.context}/{mf.speed})"
                if mf.context
                else mf.file_type
            )
            report += f"  {status} {file_type_str}: {mf.old_path}\n"
            if mf.error:
                report += f"     Error: {mf.error}\n"

    if stats.errors:
        report += "\n\nERRORS\n------\n"
        for error in stats.errors:
            report += f"- {error}\n"

    report += f"\n{'='*80}\n"

    if output_file:
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(report)
            logger.info(f"Report saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    return report


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Migrate phrase assets from old to new storage structure"
    )
    parser.add_argument(
        "--collection", default=COLLECTION, help="Collection name (default: WarmUp150)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode (check files but don't move)",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute migration (actually move files)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit migration to N phrases (for testing)",
    )
    parser.add_argument(
        "--report",
        default="migration_report.txt",
        help="Output file for migration report",
    )

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        logger.info("No operation mode specified. Use --dry-run or --execute")
        parser.print_help()
        return

    logger.info(f"Starting migration for collection: {args.collection}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")

    # Load phrases
    phrases = load_phrases(args.collection)
    if not phrases:
        logger.error("No phrases loaded")
        return

    # Limit for testing
    if args.limit:
        phrases = dict(list(phrases.items())[: args.limit])
        logger.info(f"Limited to {args.limit} phrases for testing")

    # Generate manifest
    migration_files, manifest_stats = generate_migration_manifest(
        args.collection, phrases, dry_run=True
    )

    logger.info(f"Manifest generated: {manifest_stats.files_found} files found")

    # Execute migration
    if args.execute:
        logger.info("Executing migration...")
        final_stats = execute_migration(migration_files, execute=True)
    else:
        logger.info("DRY RUN: Not executing migration")
        final_stats = manifest_stats

    # Generate report
    report = generate_report(migration_files, final_stats, args.report)
    print(report)

    logger.info("Migration complete")


if __name__ == "__main__":
    main()
