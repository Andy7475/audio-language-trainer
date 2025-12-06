"""Test script to verify Firestore connectivity and phrase model functionality."""

from src.phrases.phrase_model import (
    Phrase,
    get_phrase,
    upload_phrase,
    get_firestore_client,
)


def test_firestore_connection():
    """Test basic Firestore connection."""
    print("Testing Firestore connection...")
    try:
        client = get_firestore_client()
        print(f"✓ Successfully connected to Firestore database: {client.database}")
        return True
    except Exception as e:
        print(f"✗ Failed to connect to Firestore: {e}")
        return False


def test_phrase_model():
    """Test Phrase model creation and hash generation."""
    print("\nTesting Phrase model...")
    try:
        phrase = Phrase(
            english="She runs to the store daily",
            english_lower="she runs to the store daily",
            tokens=["she", "runs", "to", "the", "store", "daily"],
            lemmas=["she", "run", "to", "the", "store", "daily"],
            verbs=["run"],
            vocab=["she", "to", "the", "store", "daily"],
            source="manual",
        )

        phrase_hash = phrase.get_phrase_hash()
        print(f"✓ Created phrase: {phrase.english}")
        print(f"✓ Generated hash: {phrase_hash}")
        print(f"✓ Model dump works: {len(phrase.model_dump())} fields")
        return True, phrase
    except Exception as e:
        print(f"✗ Failed to create Phrase model: {e}")
        return False, None


def test_upload_and_retrieve(phrase: Phrase):
    """Test uploading and retrieving a phrase from Firestore."""
    print("\nTesting upload and retrieval...")
    try:
        # Upload the phrase
        phrase_hash = upload_phrase(phrase)
        print(f"✓ Successfully uploaded phrase with hash: {phrase_hash}")

        # Retrieve the phrase
        retrieved_phrase = get_phrase(phrase_hash)

        if retrieved_phrase:
            print(f"✓ Successfully retrieved phrase: {retrieved_phrase.english}")
            print(f"  - Tokens: {retrieved_phrase.tokens}")
            print(f"  - Verbs: {retrieved_phrase.verbs}")
            print(f"  - Vocab: {retrieved_phrase.vocab}")
            print(f"  - Source: {retrieved_phrase.source}")
            return True
        else:
            print("✗ Failed to retrieve phrase (not found)")
            return False

    except Exception as e:
        print(f"✗ Failed upload/retrieval test: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Firestore Connection and Phrase Model Test")
    print("=" * 60)

    # Test 1: Connection
    connection_ok = test_firestore_connection()

    if not connection_ok:
        print("\n⚠ Firestore connection failed. Cannot proceed with further tests.")
        print("Make sure your Google Cloud credentials are configured:")
        print("  - Run: gcloud auth application-default login")
        print("  - Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        return

    # Test 2: Model creation
    model_ok, phrase = test_phrase_model()

    if not model_ok:
        print("\n⚠ Phrase model test failed.")
        return

    # Test 3: Upload and retrieval
    upload_ok = test_upload_and_retrieve(phrase)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Connection:        {'✓ PASS' if connection_ok else '✗ FAIL'}")
    print(f"Model creation:    {'✓ PASS' if model_ok else '✗ FAIL'}")
    print(f"Upload/Retrieval:  {'✓ PASS' if upload_ok else '✗ FAIL'}")
    print("=" * 60)

    if connection_ok and model_ok and upload_ok:
        print("\n🎉 All tests passed! Firestore integration is working correctly.")
    else:
        print("\n⚠ Some tests failed. Please review the output above.")


if __name__ == "__main__":
    main()
