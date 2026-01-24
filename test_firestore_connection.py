"""Test script to verify Firestore connectivity and phrase model functionality."""

from phrases.search import get_phrase
from src.phrases.phrase_model import (
    Phrase,
    upload_phrase,
    get_firestore_client,
)
from src.logger import logger


def test_firestore_connection():
    """Test basic Firestore connection."""
    logger.info("Testing Firestore connection...")
    try:
        client = get_firestore_client()
        logger.info(
            f"✓ Successfully connected to Firestore database: {client.database}"
        )
        return True
    except Exception as e:
        logger.info(f"✗ Failed to connect to Firestore: {e}")
        return False


def test_phrase_model():
    """Test Phrase model creation and hash generation."""
    logger.info("\nTesting Phrase model...")
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
        logger.info(f"✓ Created phrase: {phrase.english}")
        logger.info(f"✓ Generated hash: {phrase_hash}")
        logger.info(f"✓ Model dump works: {len(phrase.model_dump())} fields")
        return True, phrase
    except Exception as e:
        logger.info(f"✗ Failed to create Phrase model: {e}")
        return False, None


def test_upload_and_retrieve(phrase: Phrase):
    """Test uploading and retrieving a phrase from Firestore."""
    logger.info("\nTesting upload and retrieval...")
    try:
        # Upload the phrase
        phrase_hash = upload_phrase(phrase)
        logger.info(f"✓ Successfully uploaded phrase with hash: {phrase_hash}")

        # Retrieve the phrase
        retrieved_phrase = get_phrase(phrase_hash)

        if retrieved_phrase:
            logger.info(f"✓ Successfully retrieved phrase: {retrieved_phrase.english}")
            logger.info(f"  - Tokens: {retrieved_phrase.tokens}")
            logger.info(f"  - Verbs: {retrieved_phrase.verbs}")
            logger.info(f"  - Vocab: {retrieved_phrase.vocab}")
            logger.info(f"  - Source: {retrieved_phrase.source}")
            return True
        else:
            logger.info("✗ Failed to retrieve phrase (not found)")
            return False

    except Exception as e:
        logger.info(f"✗ Failed upload/retrieval test: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Firestore Connection and Phrase Model Test")
    logger.info("=" * 60)

    # Test 1: Connection
    connection_ok = test_firestore_connection()

    if not connection_ok:
        logger.info(
            "\n⚠ Firestore connection failed. Cannot proceed with further tests."
        )
        logger.info("Make sure your Google Cloud credentials are configured:")
        logger.info("  - Run: gcloud auth application-default login")
        logger.info("  - Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        return

    # Test 2: Model creation
    model_ok, phrase = test_phrase_model()

    if not model_ok:
        logger.info("\n⚠ Phrase model test failed.")
        return

    # Test 3: Upload and retrieval
    upload_ok = test_upload_and_retrieve(phrase)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Connection:        {'✓ PASS' if connection_ok else '✗ FAIL'}")
    logger.info(f"Model creation:    {'✓ PASS' if model_ok else '✗ FAIL'}")
    logger.info(f"Upload/Retrieval:  {'✓ PASS' if upload_ok else '✗ FAIL'}")
    logger.info("=" * 60)

    if connection_ok and model_ok and upload_ok:
        logger.info(
            "\n🎉 All tests passed! Firestore integration is working correctly."
        )
    else:
        logger.info("\n⚠ Some tests failed. Please review the output above.")


if __name__ == "__main__":
    main()
