# Setup

Everything below gets you from a fresh clone to running the notebooks. It does **not** cover the Shopify integration (`src/shop.py`) — that's a separate, self-contained piece.

## 1. Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `uv sync` installs everything from `pyproject.toml`/`uv.lock`
- [FFmpeg](https://ffmpeg.org/download.html) on your `PATH` (audio processing via `pydub`/`librosa`)
- [Anki desktop](https://apps.ankiweb.net/) if you want to sync decks directly into a local collection

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

## 2. Google Cloud project

Create a project (or reuse one) and note its **project ID**.

1. Enable these APIs:
   - Cloud Firestore API
   - Cloud Text-to-Speech API
   - Cloud Translation API
   - Cloud Natural Language API
   - Vertex AI API (used for Gemini image generation)
2. Authenticate your machine with Application Default Credentials:
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```
3. Create a **Firestore database** (Native mode) named `firephrases` — this exact name is hardcoded in [`src/connections/gcloud_auth.py`](../src/connections/gcloud_auth.py).
4. Create two **GCS buckets**. Bucket names are globally unique across all of GCP, so you can't reuse the defaults — create your own and update the constants in [`src/storage.py`](../src/storage.py):
   ```python
   PRIVATE_BUCKET = "your-private-bucket-name"
   PUBLIC_BUCKET = "your-public-bucket-name"
   ```
   Make `PUBLIC_BUCKET` publicly readable (uniform bucket-level access, grant `allUsers` the `Storage Object Viewer` role) — this is what serves generated stories/images/audio on the web.

Run `uv run python scripts/check_gcloud_project.py` any time to confirm your `gcloud` CLI and Python client library both resolve to the right project.

## 3. Wiktionary dictionary links (required)

Flash cards link words to Wiktionary. This is powered by a local SQLite database, not an API — `src/wiktionary/lookup.py` reads `src/wiktionary/wiktionary_pos.db`, and phrase generation ([`src/nlp.py`](../src/nlp.py), [`src/phrases/search.py`](../src/phrases/search.py)) fails without it.

Build it once from a [Wiktextract](https://kaikki.org/dictionary/) raw data dump (`raw-wiktextract-data.jsonl.gz`, downloaded to `outputs/wiktionary_dump/`):

```bash
uv run python scripts/create_wiktionary_sql.py
```

This produces a multi-GB `wiktionary_pos.db` — it's gitignored, so every clone needs to build its own.

## 4. API keys

Copy `.env.example` to `.env` and fill in:

| Variable | Used for |
|---|---|
| `GOOGLE_PROJECT_ID` | Google Cloud project ID |
| `VERTEX_REGION` | Vertex AI region for image generation (defaults to `global`) |
| `ANTHROPIC_API_KEY` | Phrase/story generation and translation refinement (Claude) |
| `AZURE_API_KEY`, `AZURE_REGION` | Azure Speech — covers TTS voices/languages Google doesn't |
| `ELEVENLABS_API_KEY` | Optional alternative TTS voice provider |
| `DEEPAI_API_KEY`, `STABILITY_API_KEY` | Optional alternative image providers |
| `OPENAI_API_KEY` | Powers the browser-based real-time speaking challenge (used client-side, not by any Python code) |
| `ANKI_COLLECTION_PATH` | Path to your local `collection.anki2`, for syncing decks straight into Anki desktop |
| `PYTHONPATH` | Must point at `src/` so bare imports like `from phrases.search import ...` resolve |

Only `GOOGLE_PROJECT_ID` and `ANTHROPIC_API_KEY` are required to generate material end-to-end; the rest unlock extra voice/image providers.

## 5. Verify

```bash
uv run python scripts/check_gcloud_project.py
uv run python scripts/verify_firestore_connection.py
uv run pytest tests/
```

Then open `notebooks/01 Flashcards - create phrases.ipynb` and work through the numbered notebooks — see the [README](../README.md#usage) for the workflow order.
