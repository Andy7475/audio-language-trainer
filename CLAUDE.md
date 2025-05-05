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