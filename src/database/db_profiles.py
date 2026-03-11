"""
User profile CRUD operations for Supabase.
Mirrors the 10 demographic fields from src/utils/user_profile.py.
"""

from .supabase_client import get_authenticated_client


# The profile fields that can be read/written
PROFILE_FIELDS = [
    "name",
    "native_language",
    "english_fluency",
    "other_languages",
    "nationality",
    "cultural_background",
    "education_level",
    "field_of_study",
    "writing_experience",
    "writing_frequency",
    "preferred_model",
]


def get_user_profile(access_token, user_id):
    """
    Fetch the user's profile from the profiles table.

    Args:
        access_token (str): JWT access token.
        user_id (str): The user's UUID.

    Returns:
        dict: {success, data: {profile fields...} | None, error}
    """
    try:
        client = get_authenticated_client(access_token)
        result = (
            client.table("profiles")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )

        return {
            "success": True,
            "data": result.data,
            "error": None,
        }

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def update_user_profile(access_token, user_id, profile_data):
    """
    Update the user's profile. Only whitelisted fields are written.

    Args:
        access_token (str): JWT access token.
        user_id (str): The user's UUID.
        profile_data (dict): Fields to update (e.g. {"name": "Alice", "education_level": "Master's"}).

    Returns:
        dict: {success, data: updated row | None, error}
    """
    try:
        # Only allow known fields to be written
        safe_data = {k: v for k, v in profile_data.items() if k in PROFILE_FIELDS}

        if not safe_data:
            return {
                "success": False,
                "data": None,
                "error": "No valid profile fields provided.",
            }

        client = get_authenticated_client(access_token)
        result = (
            client.table("profiles")
            .update(safe_data)
            .eq("id", user_id)
            .execute()
        )

        return {
            "success": True,
            "data": result.data[0] if result.data else None,
            "error": None,
        }

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}
