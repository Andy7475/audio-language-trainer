"""Google Cloud authentication utilities."""

import sys
from typing import Tuple, Optional

from google.auth.credentials import Credentials
from google.cloud.firestore import Client as FirestoreClient
from google.cloud import language_v1
from google.cloud import texttospeech
from google.cloud import translate_v2 as translate
from google.cloud import storage


# Cached authentication
_credentials: Optional[Credentials] = None
_project_id: Optional[str] = None

# Singleton clients
_firestore_client: Optional[FirestoreClient] = None
_nlp_client: Optional[language_v1.LanguageServiceClient] = None
_translate_client: Optional[translate.Client] = None
_texttospeech_client: Optional[texttospeech.TextToSpeechClient] = None
_texttospeech_long_client: Optional[
    texttospeech.TextToSpeechLongAudioSynthesizeClient
] = None
_storage_client: Optional[storage.Client] = None


def setup_authentication() -> Tuple[Credentials, str]:
    """Setup Google Cloud authentication.

    Returns cached credentials if already authenticated, otherwise performs
    authentication and caches the result.

    Returns:
        Tuple[Credentials, str]: Tuple of (credentials, project_id)

    Raises:
        SystemExit: If authentication fails
    """
    global _credentials, _project_id

    # Return cached authentication if available
    if _credentials is not None and _project_id is not None:
        return _credentials, _project_id

    try:
        from google.auth import default

        credentials, project = default()

        # Cache the authentication result
        _credentials = credentials
        _project_id = project

        print(f"✅ Authenticated with Google Cloud project: {project}")
        return credentials, project
    except Exception as e:
        print(f"❌ Failed to authenticate with Google Cloud: {e}")
        print("\nPlease ensure you have authenticated with Google Cloud:")
        print("  - Run: gcloud auth application-default login")
        print("  - Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        sys.exit(1)


def get_firestore_client(database_name: str = "firephrases") -> FirestoreClient:
    """Get a Firestore client instance (singleton).

    Args:
        database_name: Name of the Firestore database (default: "firephrases")

    Returns:
        FirestoreClient: Firestore client instance

    Raises:
        RuntimeError: If unable to create Firestore client
        SystemExit: If authentication fails
    """
    global _firestore_client

    # Return existing client if already initialized with same database
    if _firestore_client is not None:
        return _firestore_client

    try:
        # Setup authentication
        credentials, project_id = setup_authentication()

        # Create Firestore client with authenticated credentials
        _firestore_client = FirestoreClient(
            project=project_id, credentials=credentials, database=database_name
        )
        return _firestore_client
    except Exception as e:
        raise RuntimeError(f"Failed to create Firestore client: {e}")


def get_nlp_client() -> language_v1.LanguageServiceClient:
    """Get a Natural Language API client instance (singleton).

    Returns:
        LanguageServiceClient: Natural Language API client instance

    Raises:
        RuntimeError: If unable to create NLP client
        SystemExit: If authentication fails
    """
    global _nlp_client

    if _nlp_client is not None:
        return _nlp_client

    try:
        # Setup authentication (will use default credentials)
        setup_authentication()

        # Create Natural Language client
        _nlp_client = language_v1.LanguageServiceClient()
        print("✅ Natural Language API client initialized")
        return _nlp_client
    except Exception as e:
        raise RuntimeError(f"Failed to create Natural Language API client: {e}")


def get_translate_client() -> translate.Client:
    """Get a Google Translate API client instance (singleton).

    Returns:
        translate.Client: Google Translate API client instance

    Raises:
        RuntimeError: If unable to create Translate client
        SystemExit: If authentication fails
    """
    global _translate_client

    if _translate_client is not None:
        return _translate_client

    try:
        # Setup authentication (will use default credentials)
        setup_authentication()

        # Create Translate client
        _translate_client = translate.Client()
        print("✅ Google Translate API client initialized")
        return _translate_client
    except Exception as e:
        raise RuntimeError(f"Failed to create Google Translate API client: {e}")


def get_texttospeech_client() -> texttospeech.TextToSpeechClient:
    """Get a Google Text-to-Speech API client instance (singleton).

    Returns:
        texttospeech.TextToSpeechClient: Google Text-to-Speech API client instance

    Raises:
        RuntimeError: If unable to create Text-to-Speech client
        SystemExit: If authentication fails
    """
    global _texttospeech_client

    if _texttospeech_client is not None:
        return _texttospeech_client

    try:
        # Setup authentication (will use default credentials)
        setup_authentication()

        # Create Text-to-Speech client
        _texttospeech_client = texttospeech.TextToSpeechClient()
        print("✅ Google Text-to-Speech API client initialized")
        return _texttospeech_client
    except Exception as e:
        raise RuntimeError(f"Failed to create Text-to-Speech API client: {e}")


def get_storage_client() -> storage.Client:
    """Get a Google Cloud Storage client instance (singleton).

    Returns:
        storage.Client: Google Cloud Storage client instance

    Raises:
        RuntimeError: If unable to create Storage client
        SystemExit: If authentication fails
    """
    global _storage_client

    if _storage_client is not None:
        return _storage_client

    try:
        # Setup authentication (will use default credentials)
        setup_authentication()

        # Create Storage client
        _storage_client = storage.Client()
        print("✅ Google Cloud Storage client initialized")
        return _storage_client
    except Exception as e:
        raise RuntimeError(f"Failed to create Storage client: {e}")


def get_texttospeech_long_client() -> (
    texttospeech.TextToSpeechLongAudioSynthesizeClient
):
    """Get a Google Text-to-Speech Long Audio API client instance (singleton).

    Returns:
        texttospeech.TextToSpeechLongAudioSynthesizeClient: Long-form TTS client instance

    Raises:
        RuntimeError: If unable to create Long Audio TTS client
        SystemExit: If authentication fails
    """
    global _texttospeech_long_client

    if _texttospeech_long_client is not None:
        return _texttospeech_long_client

    try:
        # Setup authentication (will use default credentials)
        setup_authentication()

        # Create Long Audio TTS client
        _texttospeech_long_client = texttospeech.TextToSpeechLongAudioSynthesizeClient()
        print("✅ Google Text-to-Speech Long Audio API client initialized")
        return _texttospeech_long_client
    except Exception as e:
        raise RuntimeError(
            f"Failed to create Text-to-Speech Long Audio API client: {e}"
        )


def reset_clients() -> None:
    """Reset all cached client instances and authentication (useful for testing)."""
    global \
        _credentials, \
        _project_id, \
        _firestore_client, \
        _nlp_client, \
        _translate_client, \
        _texttospeech_client, \
        _texttospeech_long_client, \
        _storage_client
    _credentials = None
    _project_id = None
    _firestore_client = None
    _nlp_client = None
    _translate_client = None
    _texttospeech_client = None
    _texttospeech_long_client = None
    _storage_client = None
