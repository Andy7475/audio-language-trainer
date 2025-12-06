#!/usr/bin/env python3
"""
Script to upload styles.css to the public GCS bucket.
Run this whenever you make changes to the CSS file.
"""

from src.config_loader import config
from src.gcs_storage import upload_to_gcs
from src.utils import load_template


def upload_styles_to_gcs():
    """Upload the styles.css file to the public GCS bucket."""

    # Load the CSS content
    styles_content = load_template("styles.css")

    # Upload to GCS
    public_url = upload_to_gcs(
        obj=styles_content,
        bucket_name=config.GCS_PUBLIC_BUCKET,
        file_name="styles.css",
        content_type="text/css",
    )

    print("✅ Styles uploaded successfully!")
    print(f"🌐 Public URL: {public_url}")
    print("📁 File: styles.css")
    print(f"🪣 Bucket: {config.GCS_PUBLIC_BUCKET}")

    return public_url


if __name__ == "__main__":
    upload_styles_to_gcs()
