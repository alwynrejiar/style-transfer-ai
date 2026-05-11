"""Local file-backed persistence for user-owned application data."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


LOCAL_DATA_ROOT = Path(os.environ.get("LOCAL_DATA_DIR", "local_data")).resolve()
LOCAL_USERS_DIR = LOCAL_DATA_ROOT / "users"
LOCAL_FILES_DIR = LOCAL_DATA_ROOT / "files"

for directory in (LOCAL_USERS_DIR, LOCAL_FILES_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "anonymous")).strip("._")
    return cleaned[:128] or "anonymous"


def _user_dir(user_id: str) -> Path:
    return LOCAL_USERS_DIR / _safe_segment(user_id)


def _collection_dir(user_id: str, collection: str) -> Path:
    path = _user_dir(user_id) / collection
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_path(user_id: str, collection: str, item_id: str) -> Path:
    return _collection_dir(user_id, collection) / f"{_safe_segment(item_id)}.json"


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _iter_json_files(user_id: str, collection: str) -> Iterable[Path]:
    directory = _collection_dir(user_id, collection)
    return sorted(directory.glob("*.json"))


def _confidence_percent(confidence_report: Dict[str, Any]) -> int:
    value = (
        confidence_report.get("overall_profile_confidence")
        or confidence_report.get("overall_confidence")
        or confidence_report.get("overall")
        or 0.75
    )
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.75
    if score <= 1:
        score *= 100
    return int(max(0, min(100, round(score))))


def _build_analysis_name(user_profile: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    name = user_profile.get("name") or "Unnamed"
    mode = metadata.get("processing_mode") or "analysis"
    return f"{name} - {mode}"


def _analysis_row(user_id: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    item_id = str(uuid.uuid4())
    timestamp = _now()
    metadata = dict(analysis_data.get("metadata") or {})
    user_profile = dict(analysis_data.get("user_profile") or {})
    processing_mode = metadata.get("processing_mode") or analysis_data.get("processing_mode") or "analysis"
    model_used = metadata.get("model_used") or analysis_data.get("model_used") or "unknown"
    confidence_report = analysis_data.get("confidence_report") or {}

    metadata.setdefault("processing_mode", processing_mode)
    metadata.setdefault("model_used", model_used)

    return {
        "id": item_id,
        "user_id": user_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "analysis_name": _build_analysis_name(user_profile, metadata),
        "processing_mode": processing_mode,
        "model_used": model_used,
        "source_files": metadata.get("file_info") or metadata.get("source_files") or [],
        "metadata": metadata,
        "user_profile": user_profile,
        "passes": analysis_data.get("passes", {}),
        "synthesis": analysis_data.get("synthesis", {}),
        "consolidated_analysis": (
            analysis_data.get("consolidated_analysis")
            or analysis_data.get("synthesis")
            or {}
        ),
        "readability_metrics": analysis_data.get("readability_metrics", {}),
        "confidence_report": confidence_report,
        "confidence_score": _confidence_percent(confidence_report),
        "style_fingerprint_summary": analysis_data.get("style_fingerprint_summary", ""),
        "most_distinctive_traits": analysis_data.get("most_distinctive_traits", []),
        "key_traits": analysis_data.get("key_traits", []),
        "rewrite_directive": analysis_data.get("rewrite_directive", ""),
        "do_not_lose": analysis_data.get("do_not_lose", []),
        "avoid_in_rewrite": analysis_data.get("avoid_in_rewrite", []),
        "cognitive_bridging": analysis_data.get("cognitive_bridging"),
        "analysis_data": analysis_data,
    }


def save_analysis(_access_token: str, user_id: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        row = _analysis_row(user_id, analysis_data)
        _write_json(_json_path(user_id, "profiles", row["id"]), row)
        _refresh_profile_count(user_id)
        return {
            "success": True,
            "data": {
                "id": row["id"],
                "analysis_name": row["analysis_name"],
                "created_at": row["created_at"],
            },
            "error": None,
        }
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def list_analyses(_access_token: str, user_id: str) -> Dict[str, Any]:
    try:
        rows = [_read_json(path, {}) for path in _iter_json_files(user_id, "profiles")]
        rows = [row for row in rows if row.get("id")]
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return {
            "success": True,
            "data": [
                {
                    "id": row.get("id"),
                    "analysis_name": row.get("analysis_name"),
                    "name": row.get("analysis_name"),
                    "processing_mode": row.get("processing_mode"),
                    "model_used": row.get("model_used"),
                    "confidence_score": row.get("confidence_score", 75),
                    "created_at": row.get("created_at"),
                }
                for row in rows
            ],
            "error": None,
        }
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def get_analysis(_access_token: str, user_id: str, analysis_id: str) -> Dict[str, Any]:
    try:
        row = _read_json(_json_path(user_id, "profiles", analysis_id))
        if not row:
            return {"success": False, "data": None, "error": "Profile not found"}
        return {"success": True, "data": row, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def delete_analysis(_access_token: str, user_id: str, analysis_id: str) -> Dict[str, Any]:
    try:
        path = _json_path(user_id, "profiles", analysis_id)
        if path.exists():
            path.unlink()
        _delete_comparisons_for_profile(user_id, analysis_id)
        _refresh_profile_count(user_id)
        return {"success": True, "data": None, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def save_comparison(_access_token: str, user_id: str, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        timestamp = _now()
        row = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "profile_a_id": comparison_data.get("profile_a_id"),
            "profile_b_id": comparison_data.get("profile_b_id"),
            "comparison_result": comparison_data.get("comparison_result", {}),
            "similarity_score": comparison_data.get("similarity_score"),
        }
        _write_json(_json_path(user_id, "comparisons", row["id"]), row)
        return {"success": True, "data": {"id": row["id"], "created_at": row["created_at"]}, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def list_comparisons(_access_token: str, user_id: str) -> Dict[str, Any]:
    try:
        rows = [_read_json(path, {}) for path in _iter_json_files(user_id, "comparisons")]
        rows = [row for row in rows if row.get("id")]
        rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return {"success": True, "data": rows, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def save_generated_content(_access_token: str, user_id: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        timestamp = _now()
        item_id = str(uuid.uuid4())
        row = {
            "id": item_id,
            "user_id": user_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "style_analysis_id": content_data.get("style_analysis_id"),
            "content_type": content_data.get("content_type"),
            "topic": content_data.get("topic"),
            "content": content_data.get("content", ""),
            "target_length": content_data.get("target_length"),
            "actual_length": content_data.get("actual_length"),
            "tone": content_data.get("tone"),
            "model_used": content_data.get("model_used"),
            "quality_metrics": content_data.get("quality_metrics", {}),
            "style_adherence_score": content_data.get("style_adherence_score"),
        }
        json_path = _json_path(user_id, "generated_content", item_id)
        text_path = json_path.with_suffix(".txt")
        row["local_json_path"] = str(json_path)
        row["local_text_path"] = str(text_path)
        _write_json(json_path, row)
        text_path.write_text(row["content"], encoding="utf-8")
        return {"success": True, "data": {"id": item_id, "created_at": timestamp}, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def get_user_overview_profile(_access_token: str, user_id: str) -> Dict[str, Any]:
    try:
        profile = _read_profile_overview(user_id)
        return {"success": True, "data": profile, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def update_user_overview_profile(_access_token: str, user_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        profile = _read_profile_overview(user_id)
        allowed = {
            "username",
            "role",
            "writing_fingerprint_score",
            "dominant_tone",
            "number_of_saved_profiles",
            "avatar_url",
        }
        for key, value in (profile_data or {}).items():
            if key in allowed:
                profile[key] = value
        profile["updated_at"] = _now()
        _write_json(_user_dir(user_id) / "overview.json", profile)
        return {"success": True, "data": profile, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def save_avatar(user_id: str, image_bytes: bytes, extension: str) -> Dict[str, Any]:
    try:
        ext = str(extension or "").lower().strip(".")
        if ext not in {"png", "jpg", "jpeg", "webp"}:
            return {"success": False, "data": None, "error": "Unsupported image extension"}

        safe_user = _safe_segment(user_id)
        avatar_dir = LOCAL_FILES_DIR / safe_user / "avatars"
        avatar_dir.mkdir(parents=True, exist_ok=True)
        for old_avatar in avatar_dir.glob("avatar.*"):
            old_avatar.unlink()

        path = avatar_dir / f"avatar.{ext}"
        path.write_bytes(image_bytes)
        avatar_url = f"/local-files/{safe_user}/avatars/avatar.{ext}"
        return {
            "success": True,
            "data": {"avatar_url": avatar_url, "path": str(path)},
            "error": None,
        }
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def delete_user_local_data(user_id: str) -> Dict[str, Any]:
    try:
        for path in (_user_dir(user_id), LOCAL_FILES_DIR / _safe_segment(user_id)):
            if path.exists():
                shutil.rmtree(path)
        return {"success": True, "data": None, "error": None}
    except Exception as e:
        return {"success": False, "data": None, "error": str(e)}


def _read_profile_overview(user_id: str) -> Dict[str, Any]:
    path = _user_dir(user_id) / "overview.json"
    profile = _read_json(path, {}) or {}
    profile.setdefault("username", "")
    profile.setdefault("role", "")
    profile.setdefault("writing_fingerprint_score", None)
    profile.setdefault("dominant_tone", "")
    profile["number_of_saved_profiles"] = len(list(_iter_json_files(user_id, "profiles")))
    profile.setdefault("avatar_url", "")
    profile.setdefault("created_at", _now())
    profile.setdefault("updated_at", profile["created_at"])
    return profile


def _refresh_profile_count(user_id: str) -> None:
    profile = _read_profile_overview(user_id)
    profile["number_of_saved_profiles"] = len(list(_iter_json_files(user_id, "profiles")))
    profile["updated_at"] = _now()
    _write_json(_user_dir(user_id) / "overview.json", profile)


def _delete_comparisons_for_profile(user_id: str, profile_id: str) -> None:
    for path in _iter_json_files(user_id, "comparisons"):
        row = _read_json(path, {}) or {}
        if row.get("profile_a_id") == profile_id or row.get("profile_b_id") == profile_id:
            path.unlink()
