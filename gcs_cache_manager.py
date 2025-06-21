#!/usr/bin/env python3
"""
GCS Cache Management Utility

Helps manage caching issues with Google Cloud Storage public files.
"""

import argparse
import sys
from typing import Optional
from src.gcs_storage import set_gcs_cache_control, clear_gcs_file_cache
from src.config_loader import config


def set_no_cache_for_html_files(bucket_name: str, prefix: str = "") -> None:
    """Set no-cache headers for all HTML files in a bucket/prefix."""
    try:
        from google.cloud import storage

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # List all HTML files
        blobs = bucket.list_blobs(prefix=prefix)
        html_files = [
            blob.name for blob in blobs if blob.name.lower().endswith((".html", ".htm"))
        ]

        print(f"Found {len(html_files)} HTML files in gs://{bucket_name}/{prefix}")

        for file_path in html_files:
            set_gcs_cache_control(bucket_name, file_path, "no-cache, max-age=0")

    except Exception as e:
        print(f"❌ Error processing HTML files: {str(e)}")


def set_long_cache_for_assets(bucket_name: str, prefix: str = "") -> None:
    """Set long cache headers for static assets (CSS, JS, images, etc.)."""
    try:
        from google.cloud import storage

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # List all asset files
        blobs = bucket.list_blobs(prefix=prefix)
        asset_extensions = [
            ".css",
            ".js",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".svg",
            ".ico",
            ".woff",
            ".woff2",
        ]
        asset_files = [
            blob.name
            for blob in blobs
            if any(blob.name.lower().endswith(ext) for ext in asset_extensions)
        ]

        print(f"Found {len(asset_files)} asset files in gs://{bucket_name}/{prefix}")

        for file_path in asset_files:
            set_gcs_cache_control(
                bucket_name, file_path, "public, max-age=31536000"
            )  # 1 year

    except Exception as e:
        print(f"❌ Error processing asset files: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Manage GCS file caching")
    parser.add_argument(
        "command",
        choices=[
            "no-cache",
            "long-cache",
            "clear-file",
            "set-html-no-cache",
            "set-assets-cache",
        ],
        help="Command to run",
    )

    parser.add_argument("--bucket", "-b", help="GCS bucket name")
    parser.add_argument("--file", "-f", help="File path within bucket")
    parser.add_argument("--prefix", "-p", default="", help="Prefix/folder to process")
    parser.add_argument("--cache-control", "-c", help="Custom cache control value")

    args = parser.parse_args()

    # Use config bucket if not specified
    if not args.bucket:
        args.bucket = config.GCS_PUBLIC_BUCKET
        print(f"Using default bucket: {args.bucket}")

    if args.command == "no-cache":
        if not args.file:
            print("❌ --file required for no-cache command")
            sys.exit(1)
        clear_gcs_file_cache(args.bucket, args.file, "cache_control")

    elif args.command == "clear-file":
        if not args.file:
            print("❌ --file required for clear-file command")
            sys.exit(1)
        clear_gcs_file_cache(args.bucket, args.file, "reupload")

    elif args.command == "set-html-no-cache":
        set_no_cache_for_html_files(args.bucket, args.prefix)

    elif args.command == "set-assets-cache":
        set_long_cache_for_assets(args.bucket, args.prefix)

    elif args.command == "long-cache":
        if not args.file:
            print("❌ --file required for long-cache command")
            sys.exit(1)
        cache_control = args.cache_control or "public, max-age=31536000"
        set_gcs_cache_control(args.bucket, args.file, cache_control)


if __name__ == "__main__":
    main()
