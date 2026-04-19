---
name: architecture
description: Core data models, Firestore schema, and Google Cloud Storage structure for the Audio Language Trainer.
---

# Audio Language Trainer Architecture

## Overview
This system is designed to create Anki flashcards and online stories/challenges to aid in language learning. The primary data models are `Phrase` and `Story`. The application integrates with Google Cloud Platform (Firestore for metadata, Google Cloud Storage for multimedia).

## Core Data Models
### Phrases (`src/phrases/phrase_model.py`)
- **Phrase**: The base class for all language data. A phrase is identified by a unique hash generated from its canonical English text.
  - Contains linguistic metadata extracted via NLP (`src/nlp.py`): tokens, lemmas, verbs, and vocab (non-verb words). This structure enables separate deck building for verbs and vocabulary.
  - Translations are stored as sub-items in a `translations` dictionary.
- **Translation**: Belongs to a `Phrase` and represents the localized text (e.g., `fr-FR`).
  - Contains text, tokens, and audio metadata.
- **PhraseAudio**: Audio metadata structure categorized by context (`flashcard` vs `story`) and speed (`normal` vs `slow`).

### Stories (`src/story.py`)
- Represents a dialogue-based story constructed from existing learned phrases.
- Contains metadata, sections, and dialogue lines linking back to `Phrase` hashes.

## Storage Layer
### Firestore (`firestore.md`)
- Database: `firephrases`
- **Collections**:
  - `phrases`: Core collection keyed by `{phrase_hash}`. Contains English text and lemma data.
  - `phrases/{phrase_hash}/translations`: Subcollection keyed by BCP-47 tag (e.g., `fr-FR`). Contains localized text, audio metadata, and image URLs.
  - `stories`: Dialogue stories keyed by `{story_id}`.
  - `anki_decks`: Definitions of anki decks mapped to phrase hashes.
  - `wiktionary`: A cache of Wiktionary lookups (`{lang}_{token}`).

### Google Cloud Storage (GCS)
- Stores all generated audio (`.mp3`) and images (`.png`).
- Path structure for audio: `audio/{context}/{speed}/{bcp47_code}_{hash}.mp3`
- A local cache of GCS is maintained in `outputs/gcs/` to prevent redundant downloads.
