"""
Simple demo script focused on just testing character image generation.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

try:
    from PIL import Image, ImageDraw, ImageFont
    print("✓ PIL/Pillow is available")
except ImportError:
    print("✗ PIL/Pillow is not installed. Please install with: pip install Pillow")
    sys.exit(1)

try:
    import requests
    print("✓ Requests available for downloading fonts")
    HAS_REQUESTS = True
except ImportError:
    print("✗ Requests not available - cannot download fonts")
    HAS_REQUESTS = False



def simple_character_image(character, output_path, font_path=None):
    """Generate a simple character image."""
    
    # Create image
    size = (300, 300)
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    
    # Load font
    font_size = 150
    if font_path and os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
            print(f"✓ Loaded font: {os.path.basename(font_path)}")
        except Exception as e:
            print(f"✗ Error loading font {font_path}: {e}")
            font = ImageFont.load_default()
            print("Using default font")
    else:
        font = ImageFont.load_default()
        print("Using default font (may not support Chinese)")
    
    # Get character info
    import unicodedata
    try:
        char_name = unicodedata.name(character)
        char_code = f"U+{ord(character):04X}"
        print(f"Character info: '{character}' = {char_name} ({char_code})")
    except:
        print(f"Character info: '{character}' = U+{ord(character):04X}")
    
    # Calculate position to center the character
    bbox = draw.textbbox((0, 0), character, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    print(f"Text bounding box: {bbox}")
    print(f"Character dimensions: {text_width}x{text_height}")
    
    # Check if character is being rendered properly
    if text_width <= 2 or text_height <= 2:
        print(f"⚠️  WARNING: Very small dimensions - font likely doesn't support this character!")
        print(f"⚠️  You'll see a box with X instead of the character.")
        
        # Add a warning text to the image
        warning_font = ImageFont.load_default()
        draw.text((10, 10), f"Font doesn't support:", fill="red", font=warning_font)
        draw.text((10, 30), f"'{character}' ({char_code})", fill="red", font=warning_font)
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    print(f"Drawing at position: ({x}, {y})")
    
    # Draw the character
    draw.text((x, y), character, fill="black", font=font)
    
    # Add a border and label for reference
    draw.rectangle([(0, 0), (size[0]-1, size[1]-1)], outline="lightgray")
    label_font = ImageFont.load_default()
    draw.text((5, size[1]-20), f"'{character}'", fill="gray", font=label_font)
    
    # Save image
    image.save(output_path)
    print(f"✓ Saved image: {output_path}")
    
    return output_path


def main():
    """Run simple character image generation test."""
    print("Simple Character Image Generation Test")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("simple_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    
    # Test characters
    test_characters = ["人", "一", "大", "我", "好"]
    font_path = Path("C:/Users/andyl/Python/audio-language-trainer/fonts/Noto_Serif_SC/static/NotoSerifSC-Medium.ttf")
    for char in test_characters:
        print(f"\nGenerating image for '{char}':")
        output_path = output_dir / f"simple_{char}.png"
        
        try:
            simple_character_image(
                character=char,
                output_path=str(output_path),
                font_path=font_path
            )
        except Exception as e:
            print(f"✗ Error generating image for '{char}': {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTest completed! Check the '{output_dir}' directory for generated images.")


if __name__ == "__main__":
    main() 