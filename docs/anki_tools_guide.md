# Anki Tools Guide

A modular system for creating Anki flashcard decks from Phrase objects with support for multiple source and target languages.

## Overview

The `src.anki_tools` module provides functions to:
- Create Anki notes from individual Phrase objects
- Build complete decks from lists of phrases
- Support any source/target language combination using BCP-47 language tags
- Download only target language audio and images
- Separate deck generation from file saving for flexibility

## Key Features

### 1. **Modular Design**
Functions are separated by responsibility:
- `create_anki_note_from_phrase()` - Convert a single phrase to a note
- `create_anki_deck()` - Build a deck from multiple phrases
- `save_anki_deck()` - Save a deck to an .apkg file
- `create_and_save_anki_deck()` - Convenience function combining create and save

### 2. **Multi-Language Support**
- Source and target can be **any language combination**
- Uses BCP-47 language tags (`fr-FR`, `ja-JP`, `en-GB`, etc.)
- Accepts both string tags and `BCP47Language` objects
- Downloads only target language multimedia

### 3. **Flexible Deck Structure**
- Custom deck names with hierarchy support (e.g., `"French::A1::Verbs"`)
- Automatic name generation from language tags
- Consistent sorting with unique identifiers

## Quick Start

### Basic Example

```python
from src.phrases.phrase_model import get_phrase_by_english
from src.anki_tools import create_and_save_anki_deck

# Get phrases
phrases = [
    get_phrase_by_english("Hello"),
    get_phrase_by_english("Good morning"),
]

# Ensure French translations exist
for phrase in phrases:
    phrase.translate("fr-FR")
    phrase.generate_audio("flashcard", "fr-FR")

# Create and save deck
create_and_save_anki_deck(
    phrases=phrases,
    source_language="en-GB",  # What user knows
    target_language="fr-FR",  # What user is learning
    output_path="outputs/french_greetings.apkg",
    deck_name="French::Beginner::Greetings"
)
```

## Function Reference

### `create_anki_deck()`

Creates an Anki deck package from a list of Phrase objects.

**Parameters:**
- `phrases: List[Phrase]` - Phrases to include in the deck
- `source_language: str | BCP47Language` - Source language (what user knows)
- `target_language: str | BCP47Language` - Target language (what user is learning)
- `deck_name: Optional[str]` - Custom deck name (auto-generated if None)
- `model: Optional[genanki.Model]` - Custom card template
- `wiktionary_links_func: Optional[callable]` - Function to generate wiktionary links

**Returns:** `genanki.Package` ready to be saved

**Example:**
```python
package = create_anki_deck(
    phrases=my_phrases,
    source_language="en-GB",
    target_language="ja-JP",
    deck_name="Japanese::N5::Vocabulary"
)
```

### `save_anki_deck()`

Saves an Anki deck package to a .apkg file.

**Parameters:**
- `package: genanki.Package` - Package to save
- `output_path: str` - Path for the .apkg file
- `create_dirs: bool` - Create parent directories if needed (default: True)

**Returns:** `str` - Absolute path to saved file

**Example:**
```python
saved_path = save_anki_deck(package, "outputs/my_deck.apkg")
print(f"Saved to: {saved_path}")
```

### `create_and_save_anki_deck()`

Convenience function combining `create_anki_deck()` and `save_anki_deck()`.

**Parameters:** Same as `create_anki_deck()` plus:
- `output_path: str` - Where to save the .apkg file

**Returns:** `str` - Absolute path to saved file

### `create_anki_note_from_phrase()`

Creates an individual Anki note from a Phrase object. Useful for custom deck building.

**Parameters:**
- `phrase: Phrase` - The phrase to convert
- `source_language: str | BCP47Language` - Source language
- `target_language: str | BCP47Language` - Target language
- `index: int` - Position in deck (for sorting)
- `model: genanki.Model` - Card template
- `temp_dir: str` - Temporary directory for media files
- `wiktionary_links: Optional[str]` - HTML with wiktionary links

**Returns:** `tuple[genanki.Note, list[str]]` - Note and list of media file paths

### `get_anki_model()`

Gets the default Anki model (card template) for language learning.

**Parameters:**
- `model_id: int` - Unique model ID (default: 1607392313)
- `model_name: str` - Model name (default: "FirePhrase")

**Returns:** `genanki.Model` with three card types:
1. **Listening Card** - Hear audio, guess meaning
2. **Reading Card** - Read text, understand meaning
3. **Speaking Card** - See source, speak target

## Card Fields

Each note has the following fields:

| Field | Description | Example |
|-------|-------------|---------|
| SortOrder | Sorting identifier | `0001-3f4a9` |
| SourceText | Text in source language | `Hello, how are you?` |
| TargetText | Text in target language | `Bonjour, comment allez-vous ?` |
| TargetAudio | Normal speed audio | `[sound:abc123.mp3]` |
| TargetAudioSlow | Slow speed audio | `[sound:def456.mp3]` |
| WiktionaryLinks | HTML links to wiktionary | `<a href="...">bonjour</a>` |
| Picture | Image for the phrase | `<img src="xyz789.png">` |
| SourceLanguageName | Source language code | `EN` |
| TargetLanguageName | Target language code | `FR` |

## Advanced Usage

### Multiple Language Decks

Create decks for multiple target languages from the same phrases:

```python
phrases = [get_phrase_by_english("I like coffee")]
target_languages = ["fr-FR", "es-ES", "de-DE", "ja-JP"]

for lang in target_languages:
    for phrase in phrases:
        phrase.translate(lang)
        phrase.generate_audio("flashcard", lang)

    create_and_save_anki_deck(
        phrases=phrases,
        source_language="en-GB",
        target_language=lang,
        output_path=f"outputs/{lang}_deck.apkg"
    )
```

### Reverse Direction Learning

Create decks where you learn English from another language:

```python
# Learn English from French
create_and_save_anki_deck(
    phrases=phrases,
    source_language="fr-FR",  # What you know
    target_language="en-GB",  # What you're learning
    output_path="outputs/english_from_french.apkg"
)
```

### Hierarchical Deck Names

Use `::` to create nested deck structure in Anki:

```python
create_and_save_anki_deck(
    phrases=verb_phrases,
    source_language="en-GB",
    target_language="es-ES",
    deck_name="Spanish::A1 Level::Grammar::Verbs::Present Tense"
)
```

This creates: Spanish → A1 Level → Grammar → Verbs → Present Tense

### Custom Card Templates

Create a custom model with different styling or fields:

```python
from src.anki_tools import get_anki_model

# Get the default model as a starting point
custom_model = get_anki_model(
    model_id=9876543210,  # Must be unique
    model_name="MyCustomModel"
)

# Modify templates, CSS, or fields as needed
# custom_model.templates[0]["qfmt"] = "..."

# Use it when creating decks
package = create_anki_deck(
    phrases=phrases,
    source_language="en-GB",
    target_language="fr-FR",
    model=custom_model
)
```

### Adding Wiktionary Links

Provide a function to generate wiktionary links for vocabulary:

```python
def generate_wiktionary_links(phrase: Phrase) -> str:
    """Generate HTML links to wiktionary for tokens."""
    target_translation = phrase.translations["fr-FR"]
    links = []

    for token in target_translation.tokens:
        url = f"https://en.wiktionary.org/wiki/{token.lower()}"
        links.append(f'<a href="{url}">{token}</a>')

    return " • ".join(links)

# Use it when creating deck
package = create_anki_deck(
    phrases=phrases,
    source_language="en-GB",
    target_language="fr-FR",
    wiktionary_links_func=generate_wiktionary_links
)
```

## Best Practices

### 1. Prepare Phrases First

Always ensure phrases have the necessary translations and audio before creating decks:

```python
# Good: Prepare first
for phrase in phrases:
    if target_lang not in phrase.translations:
        phrase.translate(target_lang)
        phrase.generate_audio("flashcard", target_lang)

package = create_anki_deck(phrases, "en-GB", target_lang)
```

### 2. Use Descriptive Deck Names

```python
# Good: Clear hierarchy and topic
deck_name = "French::A2::Travel::At the Restaurant"

# Less good: Flat, unclear
deck_name = "French deck 1"
```

### 3. Organize Output Files

```python
# Organize by language and topic
output_path = "outputs/french/a2_level/restaurant.apkg"

# Or by date
from datetime import datetime
date_str = datetime.now().strftime("%Y%m%d")
output_path = f"outputs/french_deck_{date_str}.apkg"
```

### 4. Handle Errors Gracefully

```python
for phrase in phrases:
    try:
        phrase.translate(target_lang)
        phrase.generate_audio("flashcard", target_lang)
    except Exception as e:
        print(f"Error processing {phrase.english}: {e}")
        continue  # Skip this phrase

# Only create deck with successfully processed phrases
valid_phrases = [
    p for p in phrases
    if target_lang in p.translations
]
create_anki_deck(valid_phrases, "en-GB", target_lang)
```

## Migration from Old Code

If you're migrating from the old `src/ARCHIVE/anki_tools.py`:

### Old Way
```python
# Old: Hardcoded English source, complex GCS paths
create_anki_deck_from_gcs(
    story_name="my_story",
    collection="LM1000",
    deck_name="French::Story",
    output_dir="../outputs/gcs"
)
```

### New Way
```python
# New: Flexible languages, direct Phrase objects
from src.phrases.phrase_model import Phrase

phrases = [...]  # Get your phrases however you want

create_and_save_anki_deck(
    phrases=phrases,
    source_language="en-GB",  # Can be any language!
    target_language="fr-FR",
    output_path="outputs/french_story.apkg",
    deck_name="French::Story"
)
```

## Troubleshooting

### "Missing target translation" error

**Problem:** Phrase doesn't have the target language translation.

**Solution:** Always translate before creating deck:
```python
for phrase in phrases:
    if target_lang not in phrase.translations:
        phrase.translate(target_lang)
```

### No audio in cards

**Problem:** Audio files not generated or not downloaded.

**Solution:** Generate audio before creating deck:
```python
phrase.generate_audio("flashcard", target_lang)
```

### Media files missing after deck creation

**Problem:** Temporary directory cleaned up before saving.

**Solution:** Use `create_and_save_anki_deck()` or save immediately:
```python
package = create_anki_deck(...)
save_anki_deck(package, output_path)  # Save immediately
```

## Testing

Run the test suite:
```bash
python -m pytest tests/test_anki_tools.py -v
```

See example usage in:
- `tests/test_anki_tools.py` - Comprehensive test examples
- `examples/create_anki_deck_example.py` - Real-world usage patterns

## See Also

- [Phrase Model Documentation](phrase_model.md)
- [Storage Documentation](storage.md)
- [Language Tag Guide](language_tags.md)
- [Anki Manual](https://docs.ankiweb.net/)
- [genanki Documentation](https://github.com/kerrickstaley/genanki)
