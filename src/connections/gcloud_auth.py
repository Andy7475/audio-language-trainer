"""Google Cloud authentication utilities."""

import sys
from typing import Tuple, Optional

from google.auth.credentials import Credentials
from google.cloud.firestore import Client as FirestoreClient
from google.cloud import language_v1
from google.cloud import translate_v2 as translate


# Singleton clients
_firestore_client: Optional[FirestoreClient] = None
_nlp_client: Optional[language_v1.LanguageServiceClient] = None
_translate_client: Optional[translate.Client] = None


def setup_authentication() -> Tuple[Credentials, str]:
    """Setup Google Cloud authentication.

    Returns:
        Tuple[Credentials, str]: Tuple of (credentials, project_id)

    Raises:
        SystemExit: If authentication fails
    """
    try:
        from google.auth import default

        credentials, project = default()
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
            project=project_id,
            credentials=credentials,
            database=database_name
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


def reset_clients() -> None:
    """Reset all cached client instances (useful for testing)."""
    global _firestore_client, _nlp_client, _translate_client
    _firestore_client = None
    _nlp_client = None
    _translate_client = None