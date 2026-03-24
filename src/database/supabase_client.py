"""
Supabase client for Style Transfer AI.
Provides a singleton client and per-user authenticated client.
"""

import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

_client = None
_admin_client = None


def get_supabase_client():
    """
    Returns a shared Supabase client (uses the anon key).
    Used for auth operations (sign up, sign in, password reset).
    """
    global _client
    if _client is None:
        from supabase import create_client

        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_ANON_KEY", "")

        if not url or not key:
            raise RuntimeError(
                "Missing SUPABASE_URL or SUPABASE_ANON_KEY. "
                "Create a .env file in the project root with these values. "
                "See .env.example for reference."
            )

        _client = create_client(url, key)
    return _client


def get_supabase_admin_client():
    """Return a Supabase client initialized with the service role key.

    This is required for privileged operations like deleting auth users.
    Returns None if service role credentials are unavailable.
    """
    global _admin_client
    if _admin_client is not None:
        return _admin_client

    from supabase import create_client

    url = os.environ.get("SUPABASE_URL", "")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not url or not service_key:
        return None

    _admin_client = create_client(url, service_key)
    return _admin_client


def get_authenticated_client(access_token):
    """
    Returns a Supabase client authenticated with a user's access token.
    This ensures RLS policies are enforced for that specific user.

    Args:
        access_token (str): The user's JWT access token from sign_in().

    Returns:
        supabase.Client: An authenticated Supabase client.
    """
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_ANON_KEY", "")

    if not url or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_ANON_KEY. "
            "Create a .env file in the project root with these values."
        )

    client = create_client(url, key)
    client.postgrest.auth(access_token)
    return client
