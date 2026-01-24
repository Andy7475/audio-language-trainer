# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run all tests: `python -m pytest tests/`
- Run a single test file: `python -m pytest tests/test_utils.py`
- Run a specific test: `python -m pytest tests/test_utils.py::test_sanitize_path_component`

## Code Style
- Imports: Group standard library, then third-party, then local imports with a blank line between groups
- Type hints: Use typing module for all function parameters and return values
- Documentation: Include docstrings for all functions using Google style format
- Error handling: Use try/except blocks with specific exception types and helpful error messages
- Naming: snake_case for functions/variables, CamelCase for classes
- Function parameters: Include type hints and default values where appropriate
- Testing: Use pytest with parametrized tests for comprehensive coverage

## Project Structure
- Source code in `src/` directory with corresponding tests in `tests/`
- Language-related data in `data/` directory
- Jupyter notebooks in `notebooks/` directory for experimentation
- Ignore config_loader and anything associated with it like the config import, we will parameters directly
- for any client connections put these in connections/ following the pattern in gcloud_auth.py

## Language Tags and Storage
- **Use BCP-47 language tags throughout** (e.g., `fr-FR`, `ja-JP`, `en-GB`, `uk-UA`)
- Use the `BCP47Language` Annotated type from `src.models` for type-safe language handling
- **Default English:** `en-GB` for consistency in storage paths and as the default for phrase images
- **Language in storage paths:** All GCS paths include the language tag (e.g., `phrases/fr-FR/audio/flashcard/slow/phrase_hash.mp3`)
- **GCS bucket access:** Use `src.storage` module for bucket constants (`PRIVATE_BUCKET`, `PUBLIC_BUCKET`) instead of config_loader
- **Path generators:** Use path generator functions from `src.storage` (e.g., `get_phrase_audio_path()`, `get_phrase_image_path()`)
- Maintain tag structure in all storage locations for consistency and to support future language-specific variations

## LLM Tools and Prompts
- Tool implementations: Located in `llm_tools/<tool_name>/` directory
- Prompt files: Located in `prompts/<tool_name>/` with structure:
  - `system.txt`: System prompt for the LLM
  - `user.txt`: User prompt template for the LLM
- Variable replacement: Use Python's `string.Template` class for variable substitution in prompts (e.g., `$variable_name` or `${variable_name}`)
- when adding clients make sure we obtain those from @src/connections/ - there is a pattern in gcloud_auth