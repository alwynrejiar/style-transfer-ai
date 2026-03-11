"""
Style comparison CRUD operations for Supabase.
"""

from .supabase_client import get_authenticated_client


def save_comparison(access_token, user_id, comparison_data):
    """
    Save a style comparison result to the database.

    Args:
        access_token (str): JWT access token.
        user_id (str): The user's UUID.
        comparison_data (dict): Expected keys:
            - profile_a_id (str): UUID of first style analysis
            - profile_b_id (str): UUID of second style analysis
            - comparison_result (dict): Full comparison output (cosine, burrows_delta, ngram, etc.)
            - similarity_score (float): Combined similarity score

    Returns:
        dict: {success, data: {id, created_at} | None, error}
    """
    try:
        row = {
            "user_id": user_id,
            "profile_a_id": comparison_data.get("profile_a_id"),
            "profile_b_id": comparison_data.get("profile_b_id"),
            "comparison_result": comparison_data.get("comparison_result", {}),
            "similarity_score": comparison_data.get("similarity_score"),
        }

        client = get_authenticated_client(access_token)
        result = client.table("style_comparisons").insert(row).execute()

        inserted = result.data[0] if result.data else {}
        return {
            "success": True,
            "data": {"id": inserted.get("id"), "created_at": inserted.get("created_at")},
            "error": None,
        }

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def list_comparisons(access_token, user_id):
    """
    List all comparisons for a user (lightweight).

    Returns:
        dict: {success, data: [{id, profile_a_id, profile_b_id, similarity_score, created_at}], error}
    """
    try:
        client = get_authenticated_client(access_token)
        result = (
            client.table("style_comparisons")
            .select("id, profile_a_id, profile_b_id, similarity_score, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        return {"success": True, "data": result.data or [], "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def get_comparison(access_token, user_id, comparison_id):
    """
    Fetch a single comparison with all data.

    Returns:
        dict: {success, data: full row | None, error}
    """
    try:
        client = get_authenticated_client(access_token)
        result = (
            client.table("style_comparisons")
            .select("*")
            .eq("id", comparison_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        return {"success": True, "data": result.data, "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def delete_comparison(access_token, user_id, comparison_id):
    """
    Delete a style comparison.

    Returns:
        dict: {success, data: None, error}
    """
    try:
        client = get_authenticated_client(access_token)
        client.table("style_comparisons").delete().eq("id", comparison_id).eq("user_id", user_id).execute()
        return {"success": True, "data": None, "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}
