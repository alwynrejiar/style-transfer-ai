"""
Generated content and style transfer CRUD operations for Supabase.
"""

from .supabase_client import get_authenticated_client


# ==========================================================================
# GENERATED CONTENT
# ==========================================================================

def save_generated_content(access_token, user_id, content_data):
    """
    Save generated content (article, email, story, etc.) to the database.

    Args:
        access_token (str): JWT access token.
        user_id (str): The user's UUID.
        content_data (dict): Expected keys:
            - style_analysis_id (str | None): UUID of the profile used
            - content_type (str): email, article, story, essay, etc.
            - topic (str)
            - content (str): The generated text
            - target_length (int)
            - actual_length (int)
            - tone (str)
            - model_used (str)
            - quality_metrics (dict)
            - style_adherence_score (float)

    Returns:
        dict: {success, data: {id, created_at} | None, error}
    """
    try:
        row = {
            "user_id": user_id,
            "style_analysis_id": content_data.get("style_analysis_id"),
            "content_type": content_data.get("content_type"),
            "topic": content_data.get("topic"),
            "content": content_data.get("content"),
            "target_length": content_data.get("target_length"),
            "actual_length": content_data.get("actual_length"),
            "tone": content_data.get("tone"),
            "model_used": content_data.get("model_used"),
            "quality_metrics": content_data.get("quality_metrics", {}),
            "style_adherence_score": content_data.get("style_adherence_score"),
        }

        client = get_authenticated_client(access_token)
        result = client.table("generated_content").insert(row).execute()

        inserted = result.data[0] if result.data else {}
        return {
            "success": True,
            "data": {"id": inserted.get("id"), "created_at": inserted.get("created_at")},
            "error": None,
        }

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def list_generated_content(access_token, user_id):
    """
    List all generated content for a user (lightweight).

    Returns:
        dict: {success, data: [{id, content_type, topic, model_used, style_adherence_score, created_at}], error}
    """
    try:
        client = get_authenticated_client(access_token)
        result = (
            client.table("generated_content")
            .select("id, content_type, topic, model_used, style_adherence_score, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        return {"success": True, "data": result.data or [], "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def get_generated_content(access_token, user_id, content_id):
    """
    Fetch a single generated content item with all data.

    Returns:
        dict: {success, data: full row | None, error}
    """
    try:
        client = get_authenticated_client(access_token)
        result = (
            client.table("generated_content")
            .select("*")
            .eq("id", content_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        return {"success": True, "data": result.data, "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def delete_generated_content(access_token, user_id, content_id):
    """
    Delete a generated content item.

    Returns:
        dict: {success, data: None, error}
    """
    try:
        client = get_authenticated_client(access_token)
        client.table("generated_content").delete().eq("id", content_id).eq("user_id", user_id).execute()
        return {"success": True, "data": None, "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


# ==========================================================================
# STYLE TRANSFERS
# ==========================================================================

def save_style_transfer(access_token, user_id, transfer_data):
    """
    Save a style transfer result to the database.

    Args:
        access_token (str): JWT access token.
        user_id (str): The user's UUID.
        transfer_data (dict): Expected keys:
            - style_analysis_id (str | None): UUID of target style profile
            - transfer_type (str): direct_transfer, style_blend, etc.
            - intensity (float): 0.0–1.0
            - original_content (str)
            - transferred_content (str)
            - preserve_elements (list[str])
            - model_used (str)
            - style_match_score (float)

    Returns:
        dict: {success, data: {id, created_at} | None, error}
    """
    try:
        row = {
            "user_id": user_id,
            "style_analysis_id": transfer_data.get("style_analysis_id"),
            "transfer_type": transfer_data.get("transfer_type"),
            "intensity": transfer_data.get("intensity"),
            "original_content": transfer_data.get("original_content"),
            "transferred_content": transfer_data.get("transferred_content"),
            "preserve_elements": transfer_data.get("preserve_elements", []),
            "model_used": transfer_data.get("model_used"),
            "style_match_score": transfer_data.get("style_match_score"),
        }

        client = get_authenticated_client(access_token)
        result = client.table("style_transfers").insert(row).execute()

        inserted = result.data[0] if result.data else {}
        return {
            "success": True,
            "data": {"id": inserted.get("id"), "created_at": inserted.get("created_at")},
            "error": None,
        }

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def list_style_transfers(access_token, user_id):
    """
    List all style transfers for a user (lightweight).

    Returns:
        dict: {success, data: [{id, transfer_type, intensity, model_used, style_match_score, created_at}], error}
    """
    try:
        client = get_authenticated_client(access_token)
        result = (
            client.table("style_transfers")
            .select("id, transfer_type, intensity, model_used, style_match_score, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        return {"success": True, "data": result.data or [], "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def get_style_transfer(access_token, user_id, transfer_id):
    """
    Fetch a single style transfer with all data.

    Returns:
        dict: {success, data: full row | None, error}
    """
    try:
        client = get_authenticated_client(access_token)
        result = (
            client.table("style_transfers")
            .select("*")
            .eq("id", transfer_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        return {"success": True, "data": result.data, "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def delete_style_transfer(access_token, user_id, transfer_id):
    """
    Delete a style transfer.

    Returns:
        dict: {success, data: None, error}
    """
    try:
        client = get_authenticated_client(access_token)
        client.table("style_transfers").delete().eq("id", transfer_id).eq("user_id", user_id).execute()
        return {"success": True, "data": None, "error": None}

    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}
