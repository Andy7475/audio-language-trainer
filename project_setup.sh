#!/bin/bash

# Create main directories
mkdir -p src tests data notebooks

# Create subdirectories in src
mkdir -p src/__pycache__

# Create Python files in src
touch src/__init__.py
touch src/language_graph.py
touch src/phrase.py
touch src/audio_generation.py
touch src/translation.py
touch src/dialogue_generation.py
touch src/utils.py

# Create test files
touch tests/__init__.py
touch tests/test_language_graph.py
touch tests/test_phrase.py
touch tests/test_audio_generation.py
touch tests/test_translation.py
touch tests/test_dialogue_generation.py

# Create data file
touch data/known_vocab_list.json

# Create notebook file
touch notebooks/original_notebook.ipynb

# Create additional project files
touch setup.py

echo "Project structure created successfully!"