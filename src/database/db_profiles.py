"""User profile CRUD operations for Supabase.
Includes demographic fields, writing style identity, and content profile."""

from .supabase_client import get_authenticated_client


# Demographic fields (user enters these)
DEMOGRAPHIC_FIELDS = [
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

# Writing style fields (populated from analysis results)
WRITING_STYLE_FIELDS = [
    "vocabulary_level",
    "avg_sentence_length",
    "formality_level",
    "tone_profile",
    "readability_score",
    "grade_level",
    "style_fingerprint",
    "dominant_traits",
    "writing_strengths",
    "writing_weaknesses",
    "preferred_content_types",
    "total_analyses_count",
    "last_analysis_at",
]

# All writable fields
PROFILE_FIELDS = DEMOGRAPHIC_FIELDS + WRITING_STYLE_FIELDS


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


def update_writing_style(access_token, user_id, analysis_data):
    """
    Update the user's writing style identity from a completed analysis.
    Call this after saving a style analysis to keep the profile current.

    Extracts key writing characteristics from the analysis result and
    stores them on the user's profile as their "writing DNA".

    Args:
        access_token (str): JWT access token.
        user_id (str): The user's UUID.
        analysis_data (dict): The full analysis profile dict (same shape as save_analysis receives).

    Returns:
        dict: {success, data: updated profile | None, error}
    """
    try:
        # Extract writing characteristics from the analysis
        consolidated = analysis_data.get("consolidated_analysis", {})
        readability = analysis_data.get("readability_metrics", {})
        traits = analysis_data.get("most_distinctive_traits", [])

        # Pull style details from the 7-pass analysis
        lexical = consolidated.get("lexical", {})
        syntactic = consolidated.get("syntactic", {})
        voice = consolidated.get("voice", {})

        # Build the style update
        style_update = {}

        # Vocabulary level
        vocab = lexical.get("vocabulary_tier")
        if vocab:
            style_update["vocabulary_level"] = vocab

        # Average sentence length
        avg_sl = syntactic.get("avg_sentence_length")
        if avg_sl is not None:
            style_update["avg_sentence_length"] = float(avg_sl)

        # Formality level
        formality = voice.get("formality_level")
        if formality:
            style_update["formality_level"] = formality

        # Tone / emotional register
        tone = voice.get("emotional_register") or voice.get("tone")
        if tone:
            style_update["tone_profile"] = tone

        # Readability scores
        flesch = readability.get("flesch_reading_ease")
        if flesch is not None:
            style_update["readability_score"] = float(flesch)

        grade = readability.get("flesch_kincaid_grade")
        if grade is not None:
            style_update["grade_level"] = float(grade)

        # Style fingerprint summary
        fingerprint = analysis_data.get("style_fingerprint_summary")
        if fingerprint:
            style_update["style_fingerprint"] = fingerprint

        # Dominant traits (top 5)
        if traits:
            style_update["dominant_traits"] = traits[:5]

        # Do-not-lose = strengths, avoid-in-rewrite = weaknesses
        strengths = analysis_data.get("do_not_lose", [])
        if strengths:
            style_update["writing_strengths"] = strengths

        weaknesses = analysis_data.get("avoid_in_rewrite", [])
        if weaknesses:
            style_update["writing_weaknesses"] = weaknesses

        # Update analysis timestamp
        from datetime import datetime, timezone
        style_update["last_analysis_at"] = datetime.now(timezone.utc).isoformat()

        if not style_update:
            return {
                "success": True,
                "data": None,
                "error": "No writing style data found in analysis.",
            }

        client = get_authenticated_client(access_token)

        # Increment total_analyses_count using current value
        current = (
            client.table("profiles")
            .select("total_analyses_count")
            .eq("id", user_id)
            .single()
            .execute()
        )
        current_count = (current.data or {}).get("total_analyses_count") or 0
        style_update["total_analyses_count"] = current_count + 1

        # Apply the update
        result = (
            client.table("profiles")
            .update(style_update)
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
