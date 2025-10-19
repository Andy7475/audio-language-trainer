"""Google Cloud authentication utilities."""

import sys
from typing import Tuple

from google.auth.credentials import Credentials
from google.cloud.firestore import Client as FirestoreClient

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
    """Get a Firestore client instance.

    Args:
        database_name: Name of the Firestore database (default: "firephrases")

    Returns:
        FirestoreClient: Firestore client instance

    Raises:
        RuntimeError: If unable to create Firestore client
        SystemExit: If authentication fails
    """
    try:
        # Setup authentication
        credentials, project_id = setup_authentication()

        # Create Firestore client with authenticated credentials
        client = FirestoreClient(
            project=project_id,
            credentials=credentials,
            database=database_name
        )
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to create Firestore client: {e}")