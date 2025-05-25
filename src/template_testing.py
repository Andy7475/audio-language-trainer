import os
from typing import Dict, Any

# Import the necessary functions from your modules
from src.phrase import build_phrase_dict_from_gcs
from src.config_loader import config
from src.utils import load_template
from src.convert import convert_PIL_image_to_base64, convert_audio_to_base64
from src.images import create_png_of_html


def batch_convert_anki_cards(html_folder, output_folder=None, device_presets=None):
    """
    Converts a folder of HTML Anki cards to PNG images with various device dimensions.

    Args:
        html_folder: Folder containing HTML files
        output_folder: Where to save the PNG files (defaults to html_folder/renders)
        device_presets: Dictionary of device names and their dimensions, or None for defaults

    Returns:
        Dictionary mapping HTML files to their rendered PNG files
    """
    import os
    import glob

    # Default device presets if none provided
    if device_presets is None:
        device_presets = {
            "iphone": (375, 812),  # iPhone X/11/12
            "android": (360, 800),  # Common Android size
        }

    # Create output folder if it doesn't exist
    if output_folder is None:
        output_folder = os.path.join(html_folder, "renders")

    os.makedirs(output_folder, exist_ok=True)

    # Find all HTML files
    html_files = glob.glob(os.path.join(html_folder, "*.html"))

    results = {}
    for html_file in html_files:
        base_name = os.path.basename(html_file)
        name_without_ext = os.path.splitext(base_name)[0]

        file_results = {}

        # Render for each device preset
        for device, dimensions in device_presets.items():
            width, height = dimensions
            output_name = f"{name_without_ext}_{device}.png"
            output_path = os.path.join(output_folder, output_name)

            try:
                create_png_of_html(
                    html_file, output_path=output_path, width=width, height=height
                )
                file_results[device] = output_path
            except Exception as e:
                print(f"Error rendering {html_file} for {device}: {str(e)}")

        results[html_file] = file_results

    return results


def image_to_base64_html(image) -> str:
    """Convert a PIL Image to base64 for embedding in HTML."""
    image_base64 = convert_PIL_image_to_base64(image)
    return f'<img src="data:image/png;base64,{image_base64}">'


def audio_to_base64_html(audio_segment, audio_type: str = "normal") -> str:
    """Convert an AudioSegment to base64 for embedding in HTML."""
    audio_base64 = convert_audio_to_base64(audio_segment)

    # Create replay button similar to Anki's
    button_color = "#4CAF50" if audio_type == "normal" else "#2196F3"

    return f"""
    <div class="replay-button" style="display:inline-block;">
        <audio id="audio_{audio_type}" style="display:none;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        <button onclick="document.getElementById('audio_{audio_type}').play()" 
                style="background-color:{button_color}; border-radius:50%; width:50px; height:50px; border:none; cursor:pointer;">
            <svg viewBox="0 0 24 24" width="30" height="30" style="margin:auto;">
                <circle cx="12" cy="12" r="12" fill="currentColor"/>
                <path d="M8,6 L18,12 L8,18 Z" fill="white"/>
            </svg>
        </button>
    </div>
    """


def generate_test_html(
    phrase_key: str = "lets_call_for_help_the_car_broke_down",
    output_dir: str = "test_templates",
    collection: str = "LM1000",
    bucket_name: str = None,
) -> None:
    """Generate test HTML files for Anki templates using a specific phrase."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get phrase data from GCS
    try:
        phrase_dict = build_phrase_dict_from_gcs(
            collection=collection, bucket_name=bucket_name, phrase_keys=[phrase_key]
        )
    except Exception as e:
        print(f"Error retrieving phrase data: {str(e)}")
        return

    if not phrase_dict or phrase_key not in phrase_dict:
        print(f"Error: Could not find phrase with key '{phrase_key}'")
        return

    # Extract data for the phrase
    phrase_data = phrase_dict[phrase_key]
    english_text = phrase_data["english_text"]
    target_text = phrase_data["target_text"]
    wiktionary_links = phrase_data.get("wiktionary_links", "")

    # Convert image and audio to base64 for embedding
    picture_html = (
        image_to_base64_html(phrase_data["image"]) if phrase_data["image"] else ""
    )
    target_audio_html = (
        audio_to_base64_html(phrase_data["audio_normal"], "normal")
        if phrase_data["audio_normal"]
        else ""
    )
    target_audio_slow_html = (
        audio_to_base64_html(phrase_data["audio_slow"], "slow")
        if phrase_data["audio_slow"]
        else ""
    )

    # Load templates
    css = load_template("card_styles.css")
    back_template = load_template("card_back_template.html")
    listening_front_template = load_template("listening_card_front_template.html")
    reading_front_template = load_template("reading_card_front_template.html")
    speaking_front_template = load_template("speaking_card_front_template.html")

    # Save the audio files separately for better testing
    if phrase_data["audio_normal"]:
        audio_path = os.path.join(output_dir, f"normal_{phrase_key}.mp3")
        phrase_data["audio_normal"].export(audio_path, format="mp3")
        print(f"Exported normal audio to {audio_path}")

    if phrase_data["audio_slow"]:
        slow_audio_path = os.path.join(output_dir, f"slow_{phrase_key}.mp3")
        phrase_data["audio_slow"].export(slow_audio_path, format="mp3")
        print(f"Exported slow audio to {slow_audio_path}")

    # Save the image separately
    if phrase_data["image"]:
        image_path = os.path.join(output_dir, f"{phrase_key}.png")
        phrase_data["image"].save(image_path, "PNG")
        print(f"Exported image to {image_path}")

    # Process each card type
    for template_name, front_template in [
        ("listening", listening_front_template),
        ("reading", reading_front_template),
        ("speaking", speaking_front_template),
    ]:
        # Create front template HTML
        front_content = front_template
        for placeholder, value in [
            ("{{Picture}}", picture_html),
            ("{{TargetText}}", target_text),
            ("{{EnglishText}}", english_text),
            ("{{TargetAudio}}", target_audio_html),
            ("{{TargetAudioSlow}}", target_audio_slow_html),
        ]:
            front_content = front_content.replace(placeholder, value)

        # Create back template HTML
        back_content = back_template
        for placeholder, value in [
            ("{{Picture}}", picture_html),
            ("{{TargetText}}", target_text),
            ("{{EnglishText}}", english_text),
            ("{{TargetAudio}}", target_audio_html),
            ("{{TargetAudioSlow}}", target_audio_slow_html),
            ("{{WiktionaryLinks}}", wiktionary_links),
            ("{{TargetLanguageName}}", config.TARGET_LANGUAGE_NAME),
            ("{{Tags}}", "story_test_story"),
        ]:
            back_content = back_content.replace(placeholder, value)

        # Write front HTML file
        with open(
            os.path.join(output_dir, f"{template_name}_front.html"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{template_name.capitalize()} Card Front</title>
                <style>
                    {css}
                    body {{
                        background-color: #303030;
                        color: white;
                    }}
                </style>
            </head>
            <body>
                {front_content}
            </body>
            </html>
            """
            )

        # Write back HTML file
        with open(
            os.path.join(output_dir, f"{template_name}_back.html"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{template_name.capitalize()} Card Back</title>
                <style>
                    {css}
                    body {{
                        background-color: #303030;
                        color: white;
                    }}
                </style>
            </head>
            <body>
                {back_content}
            </body>
            </html>
            """
            )

    # Create an index file for easy navigation
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(
            f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Anki Template Tester</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    max-width: 800px; 
                    margin: 0 auto; 
                    padding: 20px;
                    background-color: #303030;
                    color: white;
                }}
                h1, h2 {{ color: #64ffda; }}
                .card-links {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 40px; }}
                .card-link {{ 
                    display: block; padding: 15px; 
                    background-color: #112240; border-radius: 8px;
                    text-decoration: none; color: white; width: 200px;
                    text-align: center; transition: background-color 0.3s;
                }}
                .card-link:hover {{ background-color: #1d3461; }}
                .info {{ background-color: #112240; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                .media {{ display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }}
                .media-item {{ width: 200px; }}
                .media-item audio {{ width: 100%; }}
                .media-item img {{ max-width: 100%; }}
                .phrase-info {{ background-color: #112240; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Anki Template Tester</h1>
            
            <div class="info">
                <h2>About this Tool</h2>
                <p>This tool allows you to test your Anki card templates with real data. Click on the links below to view the different card types.</p>
            </div>
            
            <div class="phrase-info">
                <h2>Current Phrase Data</h2>
                <p><strong>Phrase Key:</strong> {phrase_key}</p>
                <p><strong>English Text:</strong> {english_text}</p>
                <p><strong>Target Text:</strong> {target_text}</p>
                <p><strong>Target Language:</strong> {config.TARGET_LANGUAGE_NAME}</p>
            </div>
            
            <h2>Media Files</h2>
            <div class="media">
                <div class="media-item">
                    <h3>Normal Audio</h3>
                    <audio controls src="normal_{phrase_key}.mp3"></audio>
                </div>
                <div class="media-item">
                    <h3>Slow Audio</h3>
                    <audio controls src="slow_{phrase_key}.mp3"></audio>
                </div>
                <div class="media-item">
                    <h3>Image</h3>
                    <img src="{phrase_key}.png" alt="Phrase Image">
                </div>
            </div>
            
            <h2>Card Templates</h2>
            <div class="card-links">
                <a href="listening_front.html" class="card-link">Listening Card Front</a>
                <a href="listening_back.html" class="card-link">Listening Card Back</a>
                <a href="reading_front.html" class="card-link">Reading Card Front</a>
                <a href="reading_back.html" class="card-link">Reading Card Back</a>
                <a href="speaking_front.html" class="card-link">Speaking Card Front</a>
                <a href="speaking_back.html" class="card-link">Speaking Card Back</a>
            </div>
        </body>
        </html>
        """
        )

    print(f"Generated test HTML files in '{output_dir}' directory")
    print(
        f"Open '{os.path.join(output_dir, 'index.html')}' to navigate between the templates"
    )
