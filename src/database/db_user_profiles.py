"""User overview profile helpers for the FastAPI account endpoints."""

from .supabase_client import get_authenticated_client


DIRECT_OVERVIEW_FIELDS = {
    "username",
    "role",
    "writing_fingerprint_score",
    "dominant_tone",
    "number_of_saved_profiles",
    "avatar_url",
}


def _overview_from_profile(profile):
    profile = profile or {}
    return {
        "username": profile.get("username") or profile.get("name") or "",
        "role": profile.get("role") or "",
        "writing_fingerprint_score": profile.get("writing_fingerprint_score"),
        "dominant_tone": profile.get("dominant_tone") or profile.get("tone_profile") or "",
        "number_of_saved_profiles": (
            profile.get("number_of_saved_profiles")
            if profile.get("number_of_saved_profiles") is not None
            else profile.get("total_analyses_count", 0)
        ),
        "avatar_url": profile.get("avatar_url") or "",
    }


def get_user_overview_profile(access_token, user_id):
    """Fetch the compact profile shape used by the account UI."""
    try:
        client = get_authenticated_client(access_token)
        result = client.table("profiles").select("*").eq("id", user_id).single().execute()
        return {"success": True, "data": _overview_from_profile(result.data), "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def update_user_overview_profile(access_token, user_id, profile_data):
    """Update overview fields while falling back to existing profile columns."""
    try:
        client = get_authenticated_client(access_token)
        current = client.table("profiles").select("*").eq("id", user_id).single().execute()
        current_profile = current.data or {}
        available_columns = set(current_profile.keys())

        safe_data = {}
        for field in DIRECT_OVERVIEW_FIELDS:
            if field in profile_data and field in available_columns:
                safe_data[field] = profile_data[field]

        if "username" in profile_data and "name" in available_columns:
            safe_data["name"] = profile_data["username"]
        if "dominant_tone" in profile_data and "tone_profile" in available_columns:
            safe_data["tone_profile"] = profile_data["dominant_tone"]
        if "number_of_saved_profiles" in profile_data and "total_analyses_count" in available_columns:
            safe_data["total_analyses_count"] = profile_data["number_of_saved_profiles"]

        if safe_data:
            updated = client.table("profiles").update(safe_data).eq("id", user_id).execute()
            row = updated.data[0] if updated.data else {**current_profile, **safe_data}
        else:
            row = current_profile

        return {"success": True, "data": _overview_from_profile(row), "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}
