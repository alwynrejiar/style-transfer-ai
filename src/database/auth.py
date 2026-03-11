"""
Authentication helpers for Style Transfer AI.
Wraps Supabase Auth for signup, login, logout, and password reset.
All functions return a consistent dict: {success, data, error}.
"""

from .supabase_client import get_supabase_client


def sign_up(email, password, user_name=""):
    """
    Create a new user account.
    A profiles row is auto-created by the database trigger.

    Args:
        email (str): User's email address.
        password (str): Password (min 6 characters, enforced by Supabase).
        user_name (str): Display name stored in user metadata and profiles table.

    Returns:
        dict: {success: bool, data: {user_id, email} | None, error: str | None}
    """
    try:
        client = get_supabase_client()
        result = client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {"name": user_name}
            }
        })

        if result.user:
            return {
                "success": True,
                "data": {
                    "user_id": str(result.user.id),
                    "email": result.user.email,
                },
                "error": None,
            }

        return {
            "success": False,
            "data": None,
            "error": "Sign up failed — no user returned. Check email/password.",
        }

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def sign_in(email, password):
    """
    Sign in with email and password.

    Returns:
        dict: {success, data: {user_id, email, access_token, refresh_token}, error}
    """
    try:
        client = get_supabase_client()
        result = client.auth.sign_in_with_password({
            "email": email,
            "password": password,
        })

        if result.session:
            return {
                "success": True,
                "data": {
                    "user_id": str(result.user.id),
                    "email": result.user.email,
                    "access_token": result.session.access_token,
                    "refresh_token": result.session.refresh_token,
                },
                "error": None,
            }

        return {
            "success": False,
            "data": None,
            "error": "Sign in failed — invalid credentials.",
        }

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def sign_out():
    """
    Sign out the current user (invalidates the session).

    Returns:
        dict: {success, data: None, error}
    """
    try:
        client = get_supabase_client()
        client.auth.sign_out()
        return {"success": True, "data": None, "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def reset_password(email):
    """
    Send a password reset email.

    Args:
        email (str): The user's email address.

    Returns:
        dict: {success, data: None, error}
    """
    try:
        client = get_supabase_client()
        client.auth.reset_password_email(email)
        return {
            "success": True,
            "data": None,
            "error": None,
        }

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def get_current_user(access_token):
    """
    Get the currently authenticated user's info.

    Args:
        access_token (str): JWT access token from sign_in().

    Returns:
        dict: {success, data: {user_id, email, name}, error}
    """
    try:
        client = get_supabase_client()
        result = client.auth.get_user(access_token)

        if result.user:
            metadata = result.user.user_metadata or {}
            return {
                "success": True,
                "data": {
                    "user_id": str(result.user.id),
                    "email": result.user.email,
                    "name": metadata.get("name", ""),
                },
                "error": None,
            }

        return {
            "success": False,
            "data": None,
            "error": "Could not retrieve user.",
        }

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def update_password(access_token, new_password):
    """
    Change the password for the currently authenticated user.

    Args:
        access_token (str): JWT access token from sign_in().
        new_password (str): The new password (min 6 characters).

    Returns:
        dict: {success, data: None, error}
    """
    try:
        client = get_supabase_client()
        client.auth.set_session(access_token, "")
        client.auth.update_user({"password": new_password})
        return {"success": True, "data": None, "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}
