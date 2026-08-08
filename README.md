# audio-language-trainer

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Motivation
There is often a gap after completing a basic language course (like Section 1 on DuoLingo, or perhaps you've done a Foundation Michel Thomas method) - the advice is typically to start reading magazines and watching TV, but this is far too hard. I was wanting something that exposed me to some longer dialogue, rapidly increased my vocabulary, and also reinforced what I had learnt (or was still learning). A sort of 'stepping stone' towards watching a TV program, or going abroad and being comfortable in the middle of a group where people are conversing - they won't pause after 3 or 4 words, they just keep talking amongst each other!

This is designed to fill that gap.

## What it produces

Given a vocab list and a target language, it generates practice material in two formats:

**Flash cards**
- Phrases rather than isolated words ([Lexical Approach](https://en.wikipedia.org/wiki/Lexical_approach))
- Images and audio for multiple encodings ([dual-coding theory](https://en.wikipedia.org/wiki/Dual-coding_theory))
- [Cloze deletion](https://en.wikipedia.org/wiki/Cloze_test) rather than multiple choice
- Word-separated 'slow' audio to help with pronunciation
- Links into [Wiktionary](https://www.wiktionary.org/) for further study

**Long-form stories**
- Paired with a set of flash cards so you consolidate the same vocab in a longer dialogue
- Includes double-speed audio (helps parsing ability once you already know the vocab)
- Published to the web alongside a downloadable album with synchronised lyrics

## Setup

Full walkthrough (Google Cloud project/APIs, buckets, API keys, FFmpeg, Anki): **[docs/setup.md](docs/setup.md)**.

Short version once that's done:
```bash
uv sync
cp .env.example .env   # fill in your keys
uv run python scripts/check_gcloud_project.py
```

## Usage

Everything is driven from the numbered notebooks in [`notebooks/`](notebooks/), run in order for a new batch of material:

| Notebook | Does |
|---|---|
| `00 Phrase data preparation` | Turn a vocab list into verb/vocab word lists |
| `01 Flashcards - create phrases` | Generate English phrases from a word list, translate, add images/audio |
| `04 Anki Deck` | Package a collection of phrases into an `.apkg` file |
| `05 generate a story from vocab` | Build a dialogue story from a set of learnt phrases |
| `06 generate stories in new lang` | Reuse an existing story's structure in a new target language |
| `08 build speaking challenges` | Publish a story as a real-time speaking challenge |
| `10 Review flashcards` | Spot-check generated phrases/media before publishing |
| `13 Tag phrases and sync to Anki` | Tag phrases and push updates into a local Anki collection |

Each notebook only needs a `COLLECTION`/`DECK` (or similar) variable set at the top — the rest is handled by `src/`. See [docs/anki_tools_guide.md](docs/anki_tools_guide.md) for the Anki deck subsystem and [AUDIO_MODULE_GUIDE.md](AUDIO_MODULE_GUIDE.md) for the audio pipeline. Wiktionary word-links are powered by a local database — see [docs/setup.md](docs/setup.md#3-wiktionary-dictionary-links-required).

## Data model

Phrases and stories are stored in Firestore, with audio/images in Google Cloud Storage — see [firestore.md](firestore.md) for the schema.

## Roadmap
- Learning new alphabets — linking each new letter with an image, possibly useful for Chinese.

## Acknowledgements
https://www.saysomethingin.com/en/home/ - heavily inspired by the approach of this company, DuoLingo and the Michel Thomas method.
