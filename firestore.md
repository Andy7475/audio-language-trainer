# Firestore Database Schema

## Overview

This document describes the Firestore database structure for the language learning material system. The database stores phrases, translations, audio metadata, stories, Anki deck content, and Wiktionary cache data (wiktionary URLs for each word).

## Firestore name
firephrases

## Local Cache
A local cache mirror of GCS data is maintained at `outputs/gcs/` to speed up operations when creating flashcard decks and stories. This avoids repeated downloads from GCS by storing files in a structure matching the GCS bucket layout:
```
outputs/gcs/
├── audio-language-trainer-private-content/
│   ├── collections/
│   ├── phrases/
│   └── ...
└── audio-language-trainer-stories/
    └── ...
```

The `gcs_storage.py` functions automatically manage this cache with the `save_local` and `use_local` parameters.

**Key Principles:**
- British English text is the canonical reference for all phrases
- BCP-47 language codes (e.g., `en-US`, `fr-FR`, `zh-CN`) identify language variants
- Audio files are stored in Google Cloud Storage; Firestore contains metadata and URLs
- English lemmatisation (via spaCy) enables verb/vocabulary-based deck building
- Target languages don't require lemmatisation

---

## Collections

### 1. `phrases`

Root collection storing all phrases with English linguistic analysis.

**Document ID:** `{phrase_hash}` (e.g., `she_runs_to_the_store_daily_a3f8d2`)

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `english` | string | Yes | Original English phrase with original capitalisation |
| `english_lower` | string | Yes | Lowercase version for consistent lookups |
| `tokens` | array\<string\> | Yes | Tokenised words from the phrase |
| `lemmas` | array\<string\> | Yes | Lemmatised forms of all tokens |
| `verbs` | array\<string\> | Yes | Lemmatised verb forms only (can be empty) |
| `vocab` | array\<string\> | Yes | Lemmatised non-verb words (can be empty) |
| `created` | timestamp | Yes | Creation timestamp |
| `modified` | timestamp | No | Last modification timestamp |
| `source` | string | No | `"manual"`, `"tatoeba"`, or `"generated"` |

**Example:**
```json
{
  "english": "She runs to the store daily",
  "english_lower": "she runs to the store daily",
  "tokens": ["she", "runs", "to", "the", "store", "daily"],
  "lemmas": ["she", "run", "to", "the", "store", "daily"],
  "verbs": ["run"],
  "vocab": ["she", "to", "the", "store", "daily"],
  "created": "2025-01-15T10:30:00Z",
  "source": "manual",
}
```

---

### 1a. `phrases/{phrase_hash}/translations` (Subcollection)

Translations and audio metadata for each language variant.

**Document ID:** `{bcp47_code}` (e.g., `en-US`, `fr-FR`, `zh-CN`)

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `language` | string | Yes | BCP-47 language code |
| `text` | string | Yes | Translated phrase text |
| `tokens` | array\<string\> | Yes | Tokenised words (no lemmatisation) |
| `audio` | map | No | Nested audio metadata structure (see below) |
| `image_url` | string | No | GCS URL: `gs://bucket/images/{image_set}/{phrase_hash}.png` |
| `status` | string | Yes | `"verified"`, `"needs_audio"`, `"needs_review"`, `"corrupted"` |
| `modified` | timestamp | No | Last modification timestamp |

**Audio Structure:**

The `audio` field is a nested map organised by language code, context, and speed:

```json
{
  "en-US": {
    "learning": {
      "slow": {
        "url": "gs://bucket/audio/learning/{hash}/en-US_slow.mp3",
        "voice_model_id": "en-US-Wavenet-D",
        "voice_provider": "google",
        "duration_seconds": 3.2,
        "generated": "2025-01-15T10:35:00Z"
      },
      "normal": {
        "url": "gs://bucket/audio/learning/{hash}/en-US_normal.mp3",
        "voice_model_id": "en-US-Wavenet-A",
        "voice_provider": "google",
        "duration_seconds": 2.1,
        "generated": "2025-01-15T10:35:00Z"
      }
    },
    "story": {
      "normal": {
        "url": "gs://bucket/audio/story/{hash}/en-US_normal.mp3",
        "voice_model_id": "en-US-Wavenet-C",
        "voice_provider": "google",
        "speaker_tag": "alice",
        "duration_seconds": 2.5,
        "generated": "2025-01-15T10:40:00Z"
      }
    }
  }
}
```

**Context Types:**
- `learning`: Contains both `slow` and `normal` speeds for language learning
- `story`: Contains only `normal` speed for story dialogues (but explicitly adding normal allows for future changes)

**Example Translation Document:**
```json
{
  "language": "fr-FR",
  "text": "Elle court au magasin tous les jours",
  "tokens": ["elle", "court", "au", "magasin", "tous", "les", "jours"],
  "audio": {
    "fr-FR": {
      "learning": {
        "slow": { /* ... */ },
        "normal": { /* ... */ }
      },
      "story": {
        "normal": { /* ... */ }
      }
    }
  },
  "image_url": "gs://bucket/images/default/she_runs_to_the_store_daily_a3f8d2.png",
  "status": "verified",
  "modified": "2025-01-15T11:00:00Z"
}
```

---

### 2. `stories`

Collection of dialogue-based learning stories.

**Document ID:** `{story_id}` (e.g., `space_conversation_intro`)

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | Yes | Story title |
| `description` | string | No | Brief description of story content |
| `languages` | array\<string\> | Yes | List of BCP-47 language codes |
| `created` | timestamp | Yes | Creation timestamp |
| `modified` | timestamp | No | Last modification timestamp |

**Example:**
```json
{
  "title": "Space Exploration Dialogue",
  "description": "Two friends discuss reading and space",
  "languages": ["en-US", "fr-FR", "sv-SE"],
  "difficulty": "beginner",
  "created": "2025-01-15T12:00:00Z",
  "tags": ["science", "casual_conversation"]
}
```

---

### 2a. `stories/{story_id}/sections` (Subcollection)

Ordered sections within a story, each containing dialogue lines.

**Document ID:** `{section_id}` (e.g., `introduction`, `part_01`)

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `order` | number | Yes | Sequential ordering (0, 1, 2, ...) |
| `title` | string | No | Section title |
| `lines` | array\<map\> | Yes | Array of dialogue line objects |

**Line Object Structure:**
```json
{
  "speaker": "Alex",
  "phrase_hash": "i_am_reading_a_book_a34f5d",
}
```

**Example Section:**
```json
{
  "order": 0,
  "title": "Introduction",
  "lines": [
    {
      "speaker": "Alex",
      "phrase_hash": "i_am_reading_a_book_a34f5d",
    },
    {
      "speaker": "Sam",
      "phrase_hash": "thats_great_i_love_reading_b67a2c"
    }
  ]
}
```

---

### 3. `anki_decks`

Collection of Anki deck definitions referencing phrases.

**Document ID:** `{deck_id}` (e.g., `french_verbs_present_tense`)

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Human-readable deck name |
| `description` | string | No | Deck description |
| `source_language` | string | Yes | BCP-47 code for source language |
| `target_language` | string | Yes | BCP-47 code for target language |
| `phrase_hashes` | array\<string\> | Yes | Ordered list of phrase hashes |
| `phrase_count` | number | Yes | Total number of phrases |
| `created` | timestamp | Yes | Creation timestamp |
| `modified` | timestamp | No | Last modification timestamp |
| `target_lemmas` | array\<string\> | No | English (canonical) lemmas this deck targets |
| `tags` | array\<string\> | No | tags for the anki deck

**Example:**
```json
{
  "name": "French Verbs - Present Tense",
  "description": "Core verbs in present tense",
  "source_language": "en-US",
  "target_language": "fr-FR",
  "phrase_hashes": [
    "run_daily_a3f8d2",
    "eat_breakfast_b4e9c1",
    "sleep_well_c7a4f3"
  ],
  "phrase_count": 3,
  "created": "2025-01-15T14:00:00Z",
  "target_lemmas": ["run", "eat", "sleep"],
  "tags": ["verbs", "beginner"]
}
```

---

### 4. `wiktionary`

Cache of Wiktionary lookups to avoid repeated API calls. Uses a flat collection structure with composite document IDs.

**Structure:**
```
wiktionary/                           # Top-level collection
├── en_hello                          # English "hello"
├── fr_bonjour                        # French "bonjour"
├── fr_hello                          # French "hello" (same token, different language)
├── de_haus                           # German "Haus"
└── ja_世界                           # Japanese "世界"
```

**Document ID:** `{language_code}_{lowercase_token}` (e.g., `fr_bonjour`, `en_hello`, `ja_世界`)
- **Note:** Uses language code only, not territory (e.g., `fr` not `fr-FR`)
- Wiktionary doesn't distinguish between regional variants (fr-FR and fr-CA both use 'fr')

**Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `token` | string | Yes | The lowercase token being cached |
| `language_code` | string | Yes | ISO 639-1 language code (e.g., 'en', 'fr', 'ja') |
| `exists` | boolean | Yes | Whether a valid Wiktionary entry exists |
| `url` | string | No | Full Wiktionary URL (null if doesn't exist) |
| `section_anchor` | string | No | HTML anchor to language section (e.g., "#French") |
| `last_checked` | timestamp | Yes | UTC timestamp of last verification |
| `lookup_variant` | string | No | Which variant found entry: 'lowercase', 'capitalized', or 'original' |

**Example - Same token in multiple languages:**

`wiktionary/fr_pain`:
```json
{
  "token": "pain",
  "language_code": "fr",
  "exists": true,
  "url": "https://en.wiktionary.org/wiki/pain",
  "section_anchor": "#French",
  "last_checked": "2025-01-15T15:30:00.000Z",
  "lookup_variant": "lowercase"
}
```

`wiktionary/en_pain`:
```json
{
  "token": "pain",
  "language_code": "en",
  "exists": true,
  "url": "https://en.wiktionary.org/wiki/pain",
  "section_anchor": "#English",
  "last_checked": "2025-01-15T15:30:00.000Z",
  "lookup_variant": "lowercase"
}
```

**Example - No entry found:**

`wiktionary/fr_xyzabc123`:
```json
{
  "token": "xyzabc123",
  "language_code": "fr",
  "exists": false,
  "url": null,
  "section_anchor": null,
  "last_checked": "2025-01-15T15:35:00.000Z",
  "lookup_variant": null
}
```

**Design Rationale:**
1. **Flat structure** - Simpler queries, no subcollections needed
2. **Composite document ID** - Unique per language/token combination
3. **Language code only** - Wiktionary doesn't distinguish territories (fr-FR and fr-CA use same entries)
4. **Lowercase tokens** - Consistent lookup keys, original casing preserved in HTML generation
5. **Explicit exists flag** - Allows caching of "not found" results to avoid repeated web lookups

---

## Google Cloud Storage Structure

Audio files and images are stored in Google Cloud Storage, with metadata in Firestore.

```
gs://your-bucket/
├── audio/
│   ├── learning/
│   │   └── {phrase_hash}/
│   │       ├── en-US_slow.mp3
│   │       ├── en-US_normal.mp3
│   │       ├── en-GB_slow.mp3
│   │       ├── en-GB_normal.mp3
│   │       ├── fr-FR_slow.mp3
│   │       ├── fr-FR_normal.mp3
│   │       ├── zh-CN_slow.mp3
│   │       └── zh-CN_normal.mp3
│   └── story/
│       └── {phrase_hash}/
│           ├── en-US_normal.mp3
│           ├── fr-FR_normal.mp3
│           └── zh-CN_normal.mp3
└── images/
    ├── default/
    │   └── {phrase_hash}.png
    └── alternative_swedish/
        └── {phrase_hash}.png
```

**File Naming Convention:**
- Audio: `{bcp47_code}_{speed}.mp3` (e.g., `en-US_slow.mp3`)
- Images: `{phrase_hash}.png`

---

## Implementation Notes

### Phrase Hash Generation
Generate from English text with short hex suffix for uniqueness:
```python
import hashlib

def generate_phrase_hash(english_text: str) -> str:
    normalized = english_text.lower().strip()
    slug = normalized.replace(" ", "_").replace("'", "")[:50]
    hash_suffix = hashlib.sha256(normalized.encode()).hexdigest()[:6]
    return f"{slug}_{hash_suffix}"
```

### Query Patterns

**Find phrases by verb lemma:**
```python
phrases_ref.where("verbs", "array_contains", "run").get()
```

**Find phrases by vocabulary lemma:**
```python
phrases_ref.where("vocab", "array_contains", "store").get()
```

**Get translation for specific language:**
```python
translation = phrases_ref.document(phrase_hash).collection("translations").document("fr-FR").get()
```


### Audio Download Strategy
1. Bulk download: `gcloud storage cp --recursive gs://bucket/audio/learning/{hash}/ ./local/`
2. Fallback: Individual fetch for missing files
3. Check Firestore `audio.{lang}.{context}.{speed}.url` for exact paths

### Wiktionary Link Generation
Generate on-demand during deck/story export using cached data from `wiktionary_cache` collection.

### Security
- Firestore rules: Restrictive mode (deny all by default)
- Access: Service account authentication only
- No public read/write access

---

## Related Documentation

- **BCP-47 Language Codes:** [RFC 5646](https://tools.ietf.org/html/rfc5646)
- **Google Cloud TTS:** [Voice Models](https://cloud.google.com/text-to-speech/docs/voices)
- **spaCy Lemmatisation:** [English Model](https://spacy.io/models/en)# Firestore Metadata & Text Storage

