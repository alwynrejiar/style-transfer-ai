from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Optional, Union

from fastapi import Depends, FastAPI, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.analysis.analyzer import analyze_style
from src.analysis.metrics import calculate_style_similarity, extract_deep_stylometry
from src.config.settings import AVAILABLE_MODELS, PROCESSING_MODES
from src.database.auth import (
    delete_account,
    get_current_user,
    sign_in,
    sign_in_with_google,
    sign_out,
    sign_up,
    update_password,
)
from src.database.db_analyses import delete_analysis, get_analysis, list_analyses, save_analysis
from src.database.db_comparisons import list_comparisons, save_comparison
from src.database.db_content import save_generated_content
from src.database.db_profiles import get_user_profile, update_user_profile
from src.database.supabase_client import get_supabase_client
from src.generation import ContentGenerator, StyleTransfer
from src.analysis.analogy_engine import AnalogyInjector, ANALOGY_DOMAINS
from src.models.ollama_client import is_ollama_installed, list_ollama_models

app = FastAPI(title="Stylomex API", version="1.0.0")

app.mount("/app", StaticFiles(directory="app", html=True), name="app-static")
app.mount("/docs", StaticFiles(directory="docs", html=True), name="docs-static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PASS_ORDER = [
    "lexical",
    "syntactic",
    "voice",
    "discourse",
    "rhythm",
    "psycholinguistic",
]


def success(data: Any) -> Dict[str, Any]:
    return {"success": True, "data": data}


def failure(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"success": False, "error": message})


class AuthContext(BaseModel):
    access_token: str
    user_id: str
    email: str


async def bearer_auth(authorization: Optional[str] = Header(default=None)) -> AuthContext:
    if not authorization or not authorization.startswith("Bearer "):
        raise ValueError("Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise ValueError("Missing bearer token")

    user = await run_in_threadpool(get_current_user, token)
    if not user.get("success") or not user.get("data"):
        raise ValueError(user.get("error") or "Invalid token")

    data = user["data"]
    return AuthContext(access_token=token, user_id=data["user_id"], email=data.get("email", ""))


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError):
    return failure(str(exc), status_code=401)


class SignUpRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class SignInRequest(BaseModel):
    email: str
    password: str


class GoogleAuthStartRequest(BaseModel):
    redirect_to: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
    confirm_new_password: str
    refresh_token: Optional[str] = ""


class DeleteAccountRequest(BaseModel):
    confirm_email: str


class AnalyzeRequest(BaseModel):
    text: str
    model: str = "gemma3:1b"
    mode: str = "fast"
    author_name: str = "Anonymous_User"


class GenerateRequest(BaseModel):
    topic: Optional[str] = None
    prompt: Optional[str] = None
    profileId: Optional[str] = None
    profile_id: Optional[str] = None
    length: Optional[Union[int, str]] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class AnalogyRequest(BaseModel):
    text: str
    domain: str
    model_name: Optional[str] = "gemma3:1b"
    use_local: bool = True


class TransferRequest(BaseModel):
    text: str
    profileId: Optional[str] = None
    profile_id: Optional[str] = None
    instructions: str = ""
    options: Dict[str, Any] = Field(default_factory=dict)


class SaveProfileRequest(BaseModel):
    profileData: Optional[Dict[str, Any]] = None
    profile_data: Optional[Dict[str, Any]] = None
    name: str = "Anonymous_User"
    analysis_data: Optional[Dict[str, Any]] = None


class CompareRequest(BaseModel):
    profileAId: Optional[str] = None
    profileBId: Optional[str] = None
    profile_a_id: Optional[str] = None
    profile_b_id: Optional[str] = None


class SaveProfileMetaRequest(BaseModel):
    profile: Dict[str, Any]


class UpdateProfileRequest(BaseModel):
    profile: Dict[str, Any]


@app.post("/api/auth/signup")
async def api_sign_up(body: SignUpRequest):
    result = await run_in_threadpool(sign_up, body.email, body.password, body.name)
    if not result.get("success"):
        return failure(result.get("error") or "Sign up failed")
    return success(result.get("data"))


@app.post("/api/auth/signin")
async def api_sign_in(body: SignInRequest):
    result = await run_in_threadpool(sign_in, body.email, body.password)
    if not result.get("success"):
        return failure(result.get("error") or "Sign in failed", status_code=401)
    return success(result.get("data"))


@app.post("/api/auth/google/start")
async def api_auth_google_start(body: GoogleAuthStartRequest):
    redirect_to = (body.redirect_to or "").strip() or "http://127.0.0.1:8000/app/"
    result = await run_in_threadpool(sign_in_with_google, redirect_to)
    if not result.get("success"):
        return failure(result.get("error") or "Unable to start Google auth", status_code=400)
    return success(result.get("data"))


@app.post("/api/auth/signout")
async def api_sign_out(_: AuthContext = Depends(bearer_auth)):
    result = await run_in_threadpool(sign_out)
    if not result.get("success"):
        return failure(result.get("error") or "Sign out failed")
    return success({"signed_out": True})


@app.get("/api/auth/me")
async def api_auth_me(auth: AuthContext = Depends(bearer_auth)):
    return success({"user_id": auth.user_id, "email": auth.email})


@app.post("/api/auth/password")
async def api_change_password(body: ChangePasswordRequest, auth: AuthContext = Depends(bearer_auth)):
    if not body.current_password or not body.new_password:
        return failure("Current password and new password are required.")
    if body.new_password != body.confirm_new_password:
        return failure("New password and confirmation do not match.")
    if len(body.new_password) < 6:
        return failure("New password must be at least 6 characters.")

    verify = await run_in_threadpool(sign_in, auth.email, body.current_password)
    if not verify.get("success"):
        return failure("Current password is incorrect.", status_code=401)

    result = await run_in_threadpool(
        update_password,
        auth.access_token,
        body.new_password,
        body.refresh_token or "",
    )
    if not result.get("success"):
        return failure(result.get("error") or "Failed to update password")
    return success({"updated": True})


@app.post("/api/account/delete")
async def api_delete_account(body: DeleteAccountRequest, auth: AuthContext = Depends(bearer_auth)):
    if (body.confirm_email or "").strip().lower() != (auth.email or "").strip().lower():
        return failure("Confirmation email does not match the signed-in account.")

    result = await run_in_threadpool(delete_account, auth.access_token, auth.user_id)
    if not result.get("success"):
        return failure(result.get("error") or "Failed to delete account")
    return success({"deleted": True})


@app.post("/api/analyze")
async def api_analyze(body: AnalyzeRequest):
    if body.model not in AVAILABLE_MODELS:
        return failure(f"Unknown model: {body.model}")
    if body.mode not in PROCESSING_MODES:
        return failure(f"Unknown mode: {body.mode}")
    if len(body.text.split()) < 20:
        return failure("Please provide more input text for analysis.")

    async def stream() -> AsyncGenerator[str, None]:
        for pass_name in PASS_ORDER:
            yield json.dumps({"type": "pass", "pass": pass_name, "status": "running"}) + "\n"

        loop = asyncio.get_running_loop()
        analysis_future = loop.run_in_executor(
            None,
            analyze_style,
            body.text,
            True,
            body.model,
            None,
            None,
            body.mode,
            2,
        )

        elapsed_seconds = 0
        while not analysis_future.done():
            yield json.dumps({
                "type": "heartbeat",
                "status": "running",
                "elapsed_seconds": elapsed_seconds,
                "message": "Analysis is still running. Large samples can take 1-3 minutes.",
            }) + "\n"
            await asyncio.sleep(3)
            elapsed_seconds += 3

        try:
            result = await analysis_future
        except Exception as exc:
            yield json.dumps({"type": "error", "error": f"Analysis failed: {exc}"}) + "\n"
            return

        passes = result.get("passes", {})
        for pass_name in PASS_ORDER:
            yield json.dumps({
                "type": "pass",
                "pass": pass_name,
                "status": "complete",
                "data": passes.get(pass_name, {}),
            }) + "\n"

        yield json.dumps({
            "type": "pass",
            "pass": "synthesis",
            "status": "complete",
            "data": result.get("synthesis", {}),
        }) + "\n"

        yield json.dumps({"type": "result", "data": result}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/generate")
async def api_generate(body: GenerateRequest, auth: AuthContext = Depends(bearer_auth)):
    profile_id = body.profileId or body.profile_id
    profile = {}
    if profile_id:
        profile_result = await run_in_threadpool(get_analysis, auth.access_token, auth.user_id, profile_id)
        if not profile_result.get("success"):
            return failure(profile_result.get("error") or "Profile not found", status_code=404)
        profile = profile_result.get("data") or {}

    topic = (body.topic or body.prompt or "").strip()
    if not topic:
        return failure("Please provide a topic or prompt.")

    options = body.options or {}
    length_value = options.get("length", body.length if body.length is not None else 500)
    if isinstance(length_value, str):
        length_map = {"short": 300, "medium": 500, "long": 900}
        target_length = length_map.get(length_value.lower(), 500)
    else:
        target_length = int(length_value)

    async def stream() -> AsyncGenerator[str, None]:
        generator = ContentGenerator()
        generated = await run_in_threadpool(
            generator.generate_content,
            profile,
            options.get("contentType", "article"),
            topic,
            target_length,
            options.get("tone", "neutral"),
            options.get("context", ""),
            True,
            options.get("model", "gemma3:1b"),
            None,
            None,
        )

        text = generated.get("generated_content") or ""
        for token in text.split(" "):
            yield token + " "

        if text:
            content_data = {
                "style_analysis_id": profile_id,
                "content_type": options.get("contentType", "article"),
                "topic": topic,
                "content": text,
                "target_length": target_length,
                "actual_length": len(text.split()),
                "tone": options.get("tone", "neutral"),
                "model_used": options.get("model", "gemma3:1b"),
                "quality_metrics": generated.get("quality_metrics", {}),
                "style_adherence_score": generated.get("style_adherence_score", 0.0),
            }
            await run_in_threadpool(save_generated_content, auth.access_token, auth.user_id, content_data)

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


@app.post("/api/analogy")
async def api_analogy(body: AnalogyRequest):
    if not is_ollama_installed():
        return failure("Model unavailable — is Ollama running?", status_code=503)

    if not body.text or not body.text.strip():
        return failure("Please provide input text for analogy generation.", status_code=400)

    if body.domain not in ANALOGY_DOMAINS:
        return failure(f"Invalid domain. Must be one of: {list(ANALOGY_DOMAINS.keys())}", status_code=400)

    try:
        injector = AnalogyInjector(domain=body.domain)
        result = await run_in_threadpool(
            injector.augment_text,
            body.text,
            use_local=body.use_local,
            model_name=body.model_name
        )

        expanded_text = result.get("expanded_text", "")
        analogy_output = result.get("augmented_text", "")
        if not analogy_output.strip():
            # Keep API useful even if the model returns an empty transform.
            analogy_output = expanded_text

        return success({
            "expanded_text": expanded_text,
            "analogy_output": analogy_output,
            "stages_run": result.get("stages_run", []),
            "analogy_count": result.get("analogy_count", 0),
            "density_report": result.get("density_report", {}),
            "domain": body.domain
        })
    except Exception as e:
        return failure(str(e), status_code=500)


@app.post("/api/transfer")
async def api_transfer(body: TransferRequest, auth: AuthContext = Depends(bearer_auth)):
    profile_id = body.profileId or body.profile_id
    profile = {}
    if profile_id:
        profile_result = await run_in_threadpool(get_analysis, auth.access_token, auth.user_id, profile_id)
        if not profile_result.get("success"):
            return failure(profile_result.get("error") or "Profile not found", status_code=404)
        profile = profile_result.get("data") or {}
    elif body.instructions:
        profile = {
            "deep_analysis": body.instructions,
            "metadata": {"source_files": "instruction-based transfer"},
        }

    options = body.options or {}

    styler = StyleTransfer()
    result = await run_in_threadpool(
        styler.transfer_style,
        body.text,
        profile,
        options.get("transferType", "direct_transfer"),
        float(options.get("intensity", 1.0)),
        options.get("preserveElements", []),
        True,
        options.get("model", "gemma3:1b"),
        None,
        None,
    )

    if result.get("error"):
        return failure(result["error"])
    return success({
        **result,
        "transferred_text": result.get("transferred_content", ""),
    })


@app.get("/api/profiles")
async def api_list_profiles(auth: AuthContext = Depends(bearer_auth)):
    result = await run_in_threadpool(list_analyses, auth.access_token, auth.user_id)
    if not result.get("success"):
        return failure(result.get("error") or "Failed to list profiles")
    return success(result.get("data") or [])


@app.post("/api/profiles")
async def api_save_profile(body: SaveProfileRequest, auth: AuthContext = Depends(bearer_auth)):
    payload = body.profileData or body.profile_data or body.analysis_data or {}
    if not payload:
        return failure("Missing profile data")

    if "user_profile" not in payload:
        payload["user_profile"] = {"name": body.name or "Anonymous_User"}
    elif not payload.get("user_profile", {}).get("name"):
        payload["user_profile"]["name"] = body.name or "Anonymous_User"

    if "metadata" not in payload:
        payload["metadata"] = {"processing_mode": "enhanced", "model_used": "gemma3:1b"}

    result = await run_in_threadpool(save_analysis, auth.access_token, auth.user_id, payload)
    if not result.get("success"):
        return failure(result.get("error") or "Failed to save profile")
    return success(result.get("data"))


@app.get("/api/profiles/{profile_id}")
async def api_get_profile(profile_id: str, auth: AuthContext = Depends(bearer_auth)):
    result = await run_in_threadpool(get_analysis, auth.access_token, auth.user_id, profile_id)
    if not result.get("success"):
        return failure(result.get("error") or "Profile not found", status_code=404)
    return success(result.get("data"))


@app.delete("/api/profiles/{profile_id}")
async def api_delete_profile(profile_id: str, auth: AuthContext = Depends(bearer_auth)):
    result = await run_in_threadpool(delete_analysis, auth.access_token, auth.user_id, profile_id)
    if not result.get("success"):
        return failure(result.get("error") or "Failed to delete profile")
    return success({"deleted": True})


@app.post("/api/comparisons")
async def api_create_comparison(body: CompareRequest, auth: AuthContext = Depends(bearer_auth)):
    profile_a_id = body.profileAId or body.profile_a_id
    profile_b_id = body.profileBId or body.profile_b_id
    if not profile_a_id or not profile_b_id:
        return failure("Both profile IDs are required")

    a = await run_in_threadpool(get_analysis, auth.access_token, auth.user_id, profile_a_id)
    b = await run_in_threadpool(get_analysis, auth.access_token, auth.user_id, profile_b_id)
    if not a.get("success") or not b.get("success"):
        return failure("One or both profiles not found", status_code=404)

    data_a = a.get("data") or {}
    data_b = b.get("data") or {}

    text_a = data_a.get("style_fingerprint_summary") or json.dumps(data_a.get("consolidated_analysis", {}))
    text_b = data_b.get("style_fingerprint_summary") or json.dumps(data_b.get("consolidated_analysis", {}))

    deep_a = extract_deep_stylometry(text_a)
    deep_b = extract_deep_stylometry(text_b)
    similarity = calculate_style_similarity(deep_a, deep_b)

    comparison_payload = {
        "profile_a_id": profile_a_id,
        "profile_b_id": profile_b_id,
        "comparison_result": similarity,
        "similarity_score": similarity.get("combined_score", 0.0),
    }
    saved = await run_in_threadpool(save_comparison, auth.access_token, auth.user_id, comparison_payload)
    if not saved.get("success"):
        return failure(saved.get("error") or "Failed to save comparison")

    return success({
        "comparison": similarity,
        "similarity_score": round(similarity.get("combined_score", 0.0) * 100, 2),
        "feature_overlap": round(similarity.get("ngram_overlap", 0.0) * 100, 2),
        "saved": saved.get("data"),
    })


@app.get("/api/comparisons")
async def api_list_comparison(auth: AuthContext = Depends(bearer_auth)):
    result = await run_in_threadpool(list_comparisons, auth.access_token, auth.user_id)
    if not result.get("success"):
        return failure(result.get("error") or "Failed to list comparisons")
    return success(result.get("data") or [])


@app.get("/api/health")
async def api_health():
    ollama_ok = await run_in_threadpool(is_ollama_installed)
    models = []
    if ollama_ok:
        model_result = await run_in_threadpool(list_ollama_models)
        models = model_result[0] if isinstance(model_result, tuple) else []

    supabase_ok = True
    try:
        await run_in_threadpool(get_supabase_client)
    except Exception:
        supabase_ok = False

    return success({
        "ollama": bool(ollama_ok),
        "models": models,
        "supabase": supabase_ok,
    })


@app.get("/")
async def root():
    return success({"service": "Stylomex API", "status": "ok"})
