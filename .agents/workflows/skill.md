---
name: workflows
description: The orchestration pipelines and Jupyter notebook workflows for generating language learning materials.
---

# Audio Language Trainer Workflows

## Overview
The system relies on Jupyter notebooks to orchestrate the generation and packaging of language learning materials. The standard pipeline is: Vocab/Verbs -> English Phrases -> Translated/Media Phrases -> Anki Decks / Stories.

## 1. Phrase Generation (`01 Flashcards - create phrases.ipynb`)
- **Input**: Text files containing lists of target verbs and vocabulary (e.g., `LM1000_verbs.txt`).
- **Process**:
  1. An LLM (`src.phrases.generation`) generates short, simple English phrases combining these verbs and vocab.
  2. For each generated string, a `Phrase` object is created.
  3. The phrase is translated into the target language (via Google Translate + Anthropic Claude refinement).
  4. An image is generated (via Imagen, Stability, or DeepAI).
  5. Text-to-Speech (TTS) audio is generated for both 'flashcard' and 'story' contexts.
  6. The `Phrase` and its multimedia are uploaded to Firestore and GCS.
  
## 2. Anki Deck Creation (`04 Anki Deck.ipynb`)
- **Input**: A collection of generated phrases in Firestore/GCS.
- **Process**:
  1. Queries Firestore for phrases belonging to a specific collection/deck (e.g., `LM1000 Pack01`).
  2. Uses `src.anki_tools.create_anki_deck` to bundle the phrases, target language audio, images, and translations into an Anki package.
  3. **Output**: `.apkg` files saved locally for import into Anki.

## 3. Story Generation (`05 - generate a story from vocab.ipynb`)
- **Input**: A list of learned phrases from a deck.
- **Process**:
  1. Extracts verbs/vocab from the phrases.
  2. Calls an LLM (`src.llm_tools.story_generation`) to generate a dialogue-based story incorporating these phrases.
  3. Instantiates a `Story` object and saves the structure.
  4. Publishes the story (`story.publish_story`), which generates combined audio files, story-specific translations, and an `index.html` file on GCS for web access.
