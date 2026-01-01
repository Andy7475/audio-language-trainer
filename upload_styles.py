#!/usr/bin/env python3
"""
Script to upload styles.css to the public GCS bucket.
Run this whenever you make changes to the CSS file.
"""

from src.utils import load_template
from src.storage import PUBLIC_BUCKET, upload_to_gcs


def upload_styles_to_gcs():
    """Upload the styles.css file to the public GCS bucket."""
    style_sheets = ["styles.css", "challenges.css"]
    # Load the CSS content

    for style_sheet in style_sheets:
        styles_content = load_template(filename=style_sheet, parent_path=f"src/templates")

        # Upload to GCS
        public_url = upload_to_gcs(
            obj=styles_content,
            bucket_name=PUBLIC_BUCKET,
            base_prefix="styles/",
            file_name=style_sheet,
            content_type="text/css",
        )

        print("✅ Styles uploaded successfully!")
        print(f"🌐 Public URL: {public_url}")

if __name__ == "__main__":
    upload_styles_to_gcs()
