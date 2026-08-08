#!/usr/bin/env python
"""Test script for phrase generation with small vocab_dict."""

from phrases.generation import generate_phrases_from_vocab_dict

# Small test vocab_dict with a few verbs and vocabs
test_vocab_dict = {
    "verbs": ["want", "go", "see"],
    "vocab": [
        "apple",
        "table",
        "red",
        "big",
        "old",
        "door",
        "window",
        "house",
        "car",
        "book",
        "water",
        "chair",
        "small",
        "new",
        "good",
        "beautiful",
        "tree",
        "flower",
        "garden",
        "street",
        "shop",
        "restaurant",
        "coffee",
        "tea",
        "friend",
        "family",
        "child",
        "man",
        "woman",
        "dog",
        "cat",
        "bird",
        "cold",
        "hot",
        "warm",
        "happy",
        "sad",
        "tired",
        "hungry",
        "thirsty",
        "school",
        "work",
        "home",
        "city",
        "country",
        "beach",
        "mountain",
        "river",
        "fast",
        "slow",
    ],
}

print("=" * 60)
print("TESTING PHRASE GENERATION")
print("=" * 60)
print("\nInput vocab_dict:")
print(f"  Verbs: {test_vocab_dict['verbs']}")
print(f"  Vocab: {test_vocab_dict['vocab']}")
print("\nGenerating phrases...")
print("-" * 60)

try:
    phrases = generate_phrases_from_vocab_dict(test_vocab_dict)

    print(f"\n✓ Generation complete! {len(phrases)} phrases generated.")
    print("\nGenerated Phrases:")
    for i, phrase in enumerate(phrases, 1):
        print(f"  {i}. {phrase}")

except Exception as e:
    print("\n✗ Error during phrase generation:")
    print(f"  {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
