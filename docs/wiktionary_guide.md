��# Wiktionary Integration Guide

Modern Firestore-based system for caching and generating dictionary links for language learning materials.

## Overview

The new wiktionary system provides:
- **Firestore caching** - No more JSON files, uses Firestore database
- **Language code storage** - Keys on language code only (e.g., 'fr' not 'fr-FR')
- **Token ordering preservation** - Links maintain original word order
- **Batch operations** - Efficient bulk lookups
- **Automatic integration** - Works seamlessly with Translation class

## Architecture

### Firestore Structure

```
wiktionary/
├── en_hello                    # English "hello"
├── fr_bonjour                  # French "bonjour"
├── fr_hello                    # French "hello"
├── de_haus                     # German "Haus"
└── ja_世界                     # Japanese "世界"
```

**Key Design Decisions:**
1. **Flat structure** - Simple collection with composite document IDs
2. **Composite ID** - `{language_code}_{lowercase_token}` format
3. **Language code only** - Uses 'fr' not 'fr-FR' (Wiktionary doesn't distinguish by territory)
4. **Lowercase keys** - All tokens stored lowercase for consistent lookups

### WiktionaryEntry Model

```python
class WiktionaryEntry:
    token: str                          # Lowercase token
    language_code: str                  # ISO 639-1 code ('en', 'fr', 'ja')
    exists: bool                        # Whether entry exists
    url: Optional[str]                  # Wiktionary URL
    section_anchor: Optional[str]       # HTML anchor (e.g., '#French')
    last_checked: datetime              # Cache timestamp
    lookup_variant: Optional[str]       # 'lowercase', 'capitalized', or 'original'
```

## Quick Start

### Get Wiktionary Links for a Translation

The easiest way is through the `Translation` class:

```python
from src.phrases.phrase_model import get_phrase_by_english

# Get a phrase with French translation
phrase = get_phrase_by_english("Good morning")
phrase.translate("fr-FR")

# Get wiktionary links (automatically cached in Firestore)
translation = phrase.translations["fr-FR"]
html_links = translation.get_wiktionary_links()

print(html_links)
# Output: '<a href="...#French">Bonjour</a> <a href="...#French">le</a> <a href="...#French">matin</a>'
```

### Manual Lookup

For direct lookups outside the Phrase model:

```python
from src.wiktionary import get_or_fetch_wiktionary_entry

# Single token lookup
entry = get_or_fetch_wiktionary_entry("bonjour", "fr")
if entry.exists:
    print(entry.url)  # https://en.wiktionary.org/wiki/bonjour
    print(entry.section_anchor)  # #French
    html_link = entry.get_html_link("Bonjour!")  # Preserves casing/punctuation
```

### Batch Lookups

For multiple tokens (more efficient):

```python
from src.wiktionary import batch_get_or_fetch_wiktionary_entries

tokens = ["bonjour", "merci", "au revoir"]
entries = batch_get_or_fetch_wiktionary_entries(tokens, "fr")

for token, entry in entries.items():
    if entry.exists:
        print(f"{token}: {entry.url}")
```

## Integration with Translation Class

### get_wiktionary_links() Method

The `Translation` class has a built-in method for generating wiktionary links:

```python
translation = phrase.translations["fr-FR"]

# Basic usage (caches in Firestore)
links = translation.get_wiktionary_links()

# Force refresh from web
links = translation.get_wiktionary_links(force_refresh=True)

# Custom cache expiry
links = translation.get_wiktionary_links(max_age_days=30)

# Custom separator
links = translation.get_wiktionary_links(separator=" • ")
```

**Key Features:**
- Uses language code from BCP47 tag automatically ('fr' from 'fr-FR')
- Preserves token order from `translation.tokens`
- Preserves original casing and punctuation
- Batch fetches all tokens efficiently
- Caches results in Firestore

## Token Ordering

### Problem Solved

The old system had unordered tokens. The new system preserves order:

```python
# Translation text: "Bonjour le monde!"
translation.tokens  # ['Bonjour', 'le', 'monde']  ← Now ordered!

# Links maintain this order
links = translation.get_wiktionary_links()
# '<a href="...">Bonjour</a> <a href="...">le</a> <a href="...">monde</a>'
```

### How Tokens Are Ordered

Tokens are generated from the text using `get_text_tokens()` which uses Google Cloud Natural Language API to tokenize while preserving order:

```python
from src.nlp import get_text_tokens

text = "Bonjour le monde!"
tokens = get_text_tokens(text, language_code="fr")
# Returns: ['Bonjour', 'le', 'monde', '!']
```

## Anki Flashcard Integration

The new `anki_tools` automatically generates wiktionary links:

```python
from src.anki_tools import create_and_save_anki_deck

phrases = [...]  # Your phrases
for phrase in phrases:
    phrase.translate("fr-FR")

# Wiktionary links generated automatically!
create_and_save_anki_deck(
    phrases=phrases,
    source_language="en-GB",
    target_language="fr-FR",
    output_path="outputs/french.apkg"
)
```

To disable automatic generation:

```python
from src.anki_tools import create_anki_note_from_phrase

note, media = create_anki_note_from_phrase(
    phrase=phrase,
    source_language="en-GB",
    target_language="fr-FR",
    index=0,
    model=model,
    temp_dir="/tmp",
    auto_generate_wiktionary=False  # Disable auto-generation
)
```

## Caching Strategy

### Cache Lifecycle

1. **First lookup** - Fetches from Wiktionary web, saves to Firestore
2. **Subsequent lookups** - Returns from Firestore cache
3. **Stale cache** - Auto-refreshes if older than `max_age_days`

### Cache Expiry

Default: 90 days. Customize per lookup:

```python
# Use cache if less than 30 days old
entry = get_or_fetch_wiktionary_entry(
    "bonjour",
    "fr",
    max_age_days=30
)

# Always refresh
entry = get_or_fetch_wiktionary_entry(
    "bonjour",
    "fr",
    force_refresh=True
)
```

### Cache Storage

Firestore document ID example:
```
wiktionary/fr_bonjour
```

Document contents:
```json
{
  "token": "bonjour",
  "language_code": "fr",
  "exists": true,
  "url": "https://en.wiktionary.org/wiki/bonjour",
  "section_anchor": "#French",
  "last_checked": "2025-01-15T10:30:00.000Z",
  "lookup_variant": "lowercase"
}
```

## Language Code Handling

### Why Language Codes, Not Tags?

Wiktionary doesn't distinguish between regional variants:
- French from France ('fr-FR') and French from Canada ('fr-CA') both use the same Wiktionary entries
- We extract just the language code: 'fr'

### Automatic Extraction

```python
from src.models import BCP47Language

language = BCP47Language.get("fr-FR")
code = language.language  # Returns 'fr'

language = BCP47Language.get("ja-JP")
code = language.language  # Returns 'ja'
```

This is handled automatically in `Translation.get_wiktionary_links()` using the `language.language` property.

## Migration from Old System

### Running the Migration Script

```bash
# Preview migration (dry run)
python scripts/migrate_wiktionary_cache_to_firestore.py --dry-run

# Migrate all languages
python scripts/migrate_wiktionary_cache_to_firestore.py

# Migrate specific languages only
python scripts/migrate_wiktionary_cache_to_firestore.py --languages fr es de
```

### What Gets Migrated

The script converts:
- **Old format**: JSON files in GCS with HTML strings
- **New format**: Firestore documents with structured data

Example transformation:
```python
# Old JSON format
{
  "bonjour": '<a href="https://en.wiktionary.org/wiki/bonjour#French">bonjour</a>',
  "merci": '<a href="https://en.wiktionary.org/wiki/merci#French">merci</a>'
}

# New Firestore structure
wiktionary/bonjour/languages/fr: {
  token: "bonjour",
  language_code: "fr",
  exists: true,
  url: "https://en.wiktionary.org/wiki/bonjour",
  section_anchor: "#French",
  ...
}
```

### Verification After Migration

```python
from src.wiktionary import get_wiktionary_entry

# Check a migrated entry
entry = get_wiktionary_entry("bonjour", "fr")
print(f"Exists: {entry.exists}")
print(f"URL: {entry.url}")
print(f"Checked: {entry.last_checked}")
```

## API Reference

### Translation.get_wiktionary_links()

Generate HTML wiktionary links for all tokens in a translation.

```python
def get_wiktionary_links(
    self,
    force_refresh: bool = False,
    max_age_days: int = 90,
    separator: str = " ",
) -> str
```

**Parameters:**
- `force_refresh` - Force web lookup even if cached
- `max_age_days` - Refresh cache if older than this
- `separator` - String to join links (default: space)

**Returns:** HTML string with links

### get_or_fetch_wiktionary_entry()

Get a single entry from cache or fetch from web.

```python
def get_or_fetch_wiktionary_entry(
    token: str,
    language_code: str,
    force_refresh: bool = False,
    max_age_days: int = 90,
    database_name: str = "firephrases",
) -> WiktionaryEntry
```

### batch_get_or_fetch_wiktionary_entries()

Efficiently get multiple entries.

```python
def batch_get_or_fetch_wiktionary_entries(
    tokens: List[str],
    language_code: str,
    force_refresh: bool = False,
    max_age_days: int = 90,
    database_name: str = "firephrases",
) -> Dict[str, WiktionaryEntry]
```

**Returns:** Dictionary mapping lowercase tokens to entries

### fetch_wiktionary_entry()

Fetch directly from Wiktionary (bypasses cache).

```python
def fetch_wiktionary_entry(
    token: str,
    language_code: str,
    timeout: int = 10,
) -> WiktionaryEntry
```

Tries multiple lookup strategies:
1. Lowercase version
2. Capitalized version (for German nouns)
3. Original case

## Advanced Usage

### Custom Link Generation

```python
entry = get_or_fetch_wiktionary_entry("Haus", "de")  # German noun

# Generate link with custom text
link = entry.get_html_link("das Haus")
# '<a href="...">das Haus</a>'

# Or access URL directly
if entry.exists:
    full_url = f"{entry.url}{entry.section_anchor}"
    print(f"See: {full_url}")
```

### Handling Missing Entries

```python
tokens = ["bonjour", "xyzabc123"]  # xyzabc123 doesn't exist
entries = batch_get_or_fetch_wiktionary_entries(tokens, "fr")

for token, entry in entries.items():
    if entry.exists:
        print(f"✅ {token}: {entry.url}")
    else:
        print(f"❌ {token}: No entry found")
        # entry.get_html_link(token) returns plain text
```

### Multi-Language Support

```python
# Same token, different languages
entry_fr = get_or_fetch_wiktionary_entry("pain", "fr")  # bread
entry_en = get_or_fetch_wiktionary_entry("pain", "en")  # hurt

print(entry_fr.url)  # .../wiki/pain#French
print(entry_en.url)  # .../wiki/pain#English
```

### Cache Management

```python
from src.wiktionary.cache import (
    get_wiktionary_entry,
    save_wiktionary_entry,
    is_cache_stale,
)

# Check if cache exists
entry = get_wiktionary_entry("bonjour", "fr")
if entry:
    print(f"Cached: {entry.last_checked}")

    # Check if stale
    if is_cache_stale(entry, max_age_days=30):
        print("Cache is stale, refreshing...")
        fresh_entry = fetch_wiktionary_entry("bonjour", "fr")
        save_wiktionary_entry(fresh_entry)
```

## Best Practices

### 1. Use Translation Method When Possible

```python
# Good: Uses Translation class method
links = phrase.translations["fr-FR"].get_wiktionary_links()

# Also good: For custom use cases
from src.wiktionary import get_or_fetch_wiktionary_entry
entry = get_or_fetch_wiktionary_entry("bonjour", "fr")
```

### 2. Batch Operations for Multiple Tokens

```python
# Good: Single batch operation
entries = batch_get_or_fetch_wiktionary_entries(tokens, "fr")

# Avoid: Multiple individual lookups in a loop
for token in tokens:  # Inefficient!
    entry = get_or_fetch_wiktionary_entry(token, "fr")
```

### 3. Handle Missing Entries Gracefully

```python
entry = get_or_fetch_wiktionary_entry("nonsenseword", "fr")
if entry.exists:
    link = entry.get_html_link("nonsenseword")
else:
    # Fallback behavior
    link = "nonsenseword"  # Plain text
```

### 4. Cache Expiry Strategy

- **Default (90 days)** - Good for most use cases
- **Shorter (30 days)** - For frequently changing entries
- **Force refresh** - When you know entry was updated

## Troubleshooting

### Issue: Tokens are not in order

**Problem:** Old translations created before token ordering was fixed.

**Solution:** Retokenize:
```python
from src.nlp import get_text_tokens

translation = phrase.translations["fr-FR"]
# Re-tokenize with order preservation
translation.tokens = get_text_tokens(
    translation.text,
    language_code=translation.language.language
)
```

### Issue: No wiktionary entry found

**Problem:** Word doesn't exist on Wiktionary or wrong language code.

**Solution:**
1. Verify language code is correct ('fr' not 'french')
2. Check Wiktionary manually: https://en.wiktionary.org/wiki/{word}
3. Ensure language section exists on the page

### Issue: Firestore permission denied

**Problem:** Missing Firestore credentials or wrong database.

**Solution:**
```python
# Verify Firestore connection
from src.connections.gcloud_auth import get_firestore_client

client = get_firestore_client("firephrases")
print(f"Connected to: {client.project}")
```

### Issue: Cache not updating

**Problem:** Force refresh not working.

**Solution:**
```python
# Clear cache entry manually
from src.wiktionary.cache import save_wiktionary_entry
from src.wiktionary.lookup import fetch_wiktionary_entry

# Force fetch and save
fresh_entry = fetch_wiktionary_entry("word", "fr")
save_wiktionary_entry(fresh_entry)
```

## Performance Considerations

### Batch Size Limits

Firestore limits batch operations to 500 documents. The code handles this automatically:

```python
# Can safely pass >500 tokens
tokens = ["word" + str(i) for i in range(1000)]
entries = batch_get_or_fetch_wiktionary_entries(tokens, "fr")
# Automatically splits into 2 batches of 500
```

### Web Request Rate Limiting

When fetching from Wiktionary:
- Use batch operations to minimize web requests
- Cache aggressively (90-day default)
- Consider adding delay for very large batches:

```python
import time

large_token_list = [...]  # 1000+ tokens
batch_size = 100

for i in range(0, len(large_token_list), batch_size):
    batch = large_token_list[i:i+batch_size]
    entries = batch_get_or_fetch_wiktionary_entries(batch, "fr")
    time.sleep(1)  # Be nice to Wiktionary servers
```

## See Also

- [Phrase Model Documentation](phrase_model.md)
- [Anki Tools Guide](anki_tools_guide.md)
- [Firestore Schema](../firestore.md)
- [Migration Script](../scripts/migrate_wiktionary_cache_to_firestore.py)
