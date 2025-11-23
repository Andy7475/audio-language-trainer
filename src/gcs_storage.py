"""DEPRECATED: This module is being phased out in favor of src.storage.

This file provides backwards compatibility for existing code.
All functionality has been migrated to src.storage module.

New code should use:
  from src.storage import upload_file_to_gcs, download_from_gcs, PRIVATE_BUCKET, PUBLIC_BUCKET

Gradual migration:
- Old: upload_to_gcs(obj, bucket, file_name, base_prefix) -> New: upload_file_to_gcs(obj, bucket, file_path)
- Old: read_from_gcs(bucket, file_path) -> New: download_from_gcs(bucket, file_path)
"""

import warnings
import re
from typing import Any, Optional

# Import from new location
from src.storage import (
    upload_file_to_gcs,
    download_from_gcs,
    PRIVATE_BUCKET,
    PUBLIC_BUCKET,
    get_phrase_audio_path,
    get_phrase_image_path,
)

# Expose new bucket constants for backwards compatibility
GCS_PRIVATE_BUCKET = PRIVATE_BUCKET
GCS_PUBLIC_BUCKET = PUBLIC_BUCKET


def sanitize_path_component(s: str) -> str:
    """
    Sanitize a string for use in GCS paths.
    Replaces spaces with underscores, removes special characters, and converts to lowercase.
    """
    # Replace one or more spaces with a single underscore
    s = re.sub(r"\s+", "_", s)
    # Remove any characters that aren't alphanumeric, underscore, or hyphen
    s = "".join(c for c in s if c.isalnum() or c in "_-")
    # Convert to lowercase for consistency
    return s.lower()


def upload_to_gcs(
    obj: Any,
    bucket_name: str,
    file_name: str,
    base_prefix: str = "",
    content_type: Optional[str] = None,
    save_local: bool = True,
    local_base_dir: str = "../outputs/gcs",
) -> str:
    """
    DEPRECATED: Use upload_file_to_gcs from src.storage instead.

    Upload various file types directly to Google Cloud Storage without writing to local disk.

    Args:
        obj: The object to upload (bytes, dict, str, PIL Image, AudioSegment, List[str], etc.)
        bucket_name: Name of the GCS bucket
        file_name: Name of the file to upload
        base_prefix: Prefix/folder path in the bucket. Defaults to ''.
        content_type: MIME content type. If None, will be inferred.
        save_local: Whether to save a local copy of the file (default: True)
        local_base_dir: Base directory for local GCS mirror (default: "../outputs/gcs")

    Returns:
        GCS URI of the uploaded file

    Raises:
        ValueError: If unsupported object type is provided
    """
    warnings.warn(
        "upload_to_gcs() is deprecated. Use upload_file_to_gcs() from src.storage instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Construct full blob path from base_prefix and file_name (old API)
    file_path = f"{base_prefix.rstrip('/')}/{file_name}".lstrip("/")

    # Call new API
    return upload_file_to_gcs(
        obj=obj,
        bucket_name=bucket_name,
        file_path=file_path,
        content_type=content_type,
        save_local=save_local,
        local_base_dir=local_base_dir,
    )


def read_from_gcs(
    bucket_name: str,
    file_path: str,
    expected_type: Optional[str] = None,
    use_local: bool = True,
    local_base_dir: str = "../outputs/gcs",
) -> Any:
    """
    DEPRECATED: Use download_from_gcs from src.storage instead.

    Download a file from GCS or read from local cache if available.

    Args:
        bucket_name: Name of the GCS bucket
        file_path: Path to the file within the bucket
        expected_type: Optional type hint ('audio', 'image', 'json', 'bytes', 'text', 'list')
                      If None, will be inferred from file extension
        use_local: Whether to check for a local copy first (default: True)
        local_base_dir: Base directory for local GCS mirror (default: "../outputs/gcs")

    Returns:
        The file content as an appropriate Python object
    """
    warnings.warn(
        "read_from_gcs() is deprecated. Use download_from_gcs() from src.storage instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Call new API
    return download_from_gcs(
        bucket_name=bucket_name,
        file_path=file_path,
        expected_type=expected_type,
        use_local=use_local,
        local_base_dir=local_base_dir,
    )


def check_blob_exists(bucket_name: str, blob_path: str) -> bool:
    """
    Check if a blob exists in a GCS bucket.

    Args:
        bucket_name: Name of the GCS bucket
        blob_path: Path to the blob within the bucket

    Returns:
        bool: True if the blob exists, False otherwise
    """
    from src.storage import check_blob_exists as _check_blob_exists
    return _check_blob_exists(bucket_name, blob_path)
