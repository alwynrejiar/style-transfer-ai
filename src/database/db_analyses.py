"""
Style analysis CRUD operations for Supabase.
Replaces the local JSON file storage in src/storage/local_storage.py.
"""

from .supabase_client import get_authenticated_client


def save_analysis(access_token, user_id, analysis_data):
    """
    Save a style analysis result to the database.

    Accepts the same dict shape that save_style_profile_locally() receives,
    and maps it into the style_analyses table columns.

    Args:
        access_token (str): JWT access token.
        user_id (str): The user's UUID.
        analysis_data (dict): The full analysis profile dict. Expected keys:
            - metadata (dict): {model_used, processing_mode, file_info, ...}
            - consolidated_analysis (dict): The full 7-pass analysis
            - readability_metrics (dict): 14 readability scores
            - confidence_report (dict): Per-pass confidence data
            - style_fingerprint_summary (str)
            - most_distinctive_traits (list[str])
            - key_traits (list[str])
            - rewrite_directive (str)
            - do_not_lose (list[str])
            - avoid_in_rewrite (list[str])
            - cognitive_bridging (dict | None)

    Returns:
        dict: {success, data: {id, analysis_name, created_at} | None, error}
    """
    try:
        metadata = analysis_data.get("metadata", {})
        user_profile = analysis_data.get("user_profile", {})

        # Build the row
        row = {
            "user_id": user_id,
            "analysis_name": _build_analysis_name(user_profile, metadata),
            "processing_mode": metadata.get("processing_mode"),
            "model_used": metadata.get("model_used"),
            "source_files": metadata.get("file_info", []),
            "consolidated_analysis": analysis_data.get("consolidated_analysis", {}),
            "readability_metrics": analysis_data.get("readability_metrics", {}),
            "confidence_report": analysis_data.get("confidence_report", {}),
            "style_fingerprint_summary": analysis_data.get("style_fingerprint_summary"),
            "most_distinctive_traits": analysis_data.get("most_distinctive_traits", []),
            "key_traits": analysis_data.get("key_traits", []),
            "rewrite_directive": analysis_data.get("rewrite_directive"),
            "do_not_lose": analysis_data.get("do_not_lose", []),
            "avoid_in_rewrite": analysis_data.get("avoid_in_rewrite", []),
            "cognitive_bridging": analysis_data.get("cognitive_bridging"),
        }

        client = get_authenticated_client(access_token)
        result = (
            client.table("style_analyses")
            .insert(row)
            .execute()
        )

        inserted = result.data[0] if result.data else {}
        return {
            "success": True,
            "data": {
                "id": inserted.get("id"),
                "analysis_name": inserted.get("analysis_name"),
                "created_at": inserted.get("created_at"),
            },
            "error": None,
        }

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def list_analyses(access_token, user_id):
    """
    List all analyses for a user (lightweight — no large JSONB blobs).

    Returns:
        dict: {success, data: [{id, analysis_name, processing_mode, model_used, created_at}], error}
    """
    try:
        client = get_authenticated_client(access_token)
        result = (
            client.table("style_analyses")
            .select("id, analysis_name, processing_mode, model_used, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        return {
            "success": True,
            "data": result.data or [],
            "error": None,
        }

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def get_analysis(access_token, user_id, analysis_id):
    """
    Fetch a single analysis with all data.

    Returns:
        dict: {success, data: full row dict | None, error}
    """
    try:
        client = get_authenticated_client(access_token)
        result = (
            client.table("style_analyses")
            .select("*")
            .eq("id", analysis_id)
            .eq("user_id", user_id)
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


def delete_analysis(access_token, user_id, analysis_id):
    """
    Delete a style analysis.

    Returns:
        dict: {success, data: None, error}
    """
    try:
        client = get_authenticated_client(access_token)
        client.table("style_analyses").delete().eq("id", analysis_id).eq("user_id", user_id).execute()

        return {"success": True, "data": None, "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_analysis_name(user_profile, metadata):
    """Generate a human-readable analysis name from profile + metadata."""
    name = user_profile.get("name", "Unnamed")
    mode = metadata.get("processing_mode", "analysis")
    return f"{name} — {mode}"
