from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

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


VALID_PROVIDERS = {"ollama", "gemini", "openrouter", "openai"}
CONTENT_TYPE_ALIASES = {
    "blog": "blog_post",
    "social": "social_media",
}


def _detect_provider_from_model(model: str) -> str:
    normalized = str(model or "").strip().lower()
    if normalized.startswith("gemini"):
        return "gemini"
    if (
        normalized.startswith("anthropic/")
        or normalized.startswith("deepseek/")
        or normalized.startswith("meta-llama/")
        or "claude" in normalized
    ):
        return "openrouter"
    if normalized.startswith("gpt-"):
        return "openai"
    return "ollama"


def _resolve_provider(provider: Optional[str], model_key: str) -> str:
    requested = (provider or "").strip().lower()
    model_info = AVAILABLE_MODELS.get(model_key, {})
    model_provider = str(model_info.get("provider") or model_info.get("type") or "").strip().lower()
    inferred = _detect_provider_from_model(model_key)
    resolved = requested or model_provider or inferred
    print(f"DEBUG resolve: provider={provider}, model_key={model_key}, requested={requested}, model_provider={model_provider}, inferred={inferred}, resolved={resolved}")
    if resolved not in VALID_PROVIDERS:
        raise ValueError(f"Unknown provider: {resolved}")
    if model_provider and resolved != model_provider:
        raise ValueError(
            f"Model '{model_key}' belongs to provider '{model_provider}', got provider '{resolved}'."
        )
    return resolved


def _resolve_model_id(model_key: str) -> str:
    model_info = AVAILABLE_MODELS.get(model_key, {})
    return str(model_info.get("model_id") or model_key)


def _normalize_content_type(content_type: Any) -> str:
    normalized = str(content_type or "article").strip().lower()
    return CONTENT_TYPE_ALIASES.get(normalized, normalized or "article")


def _normalize_model_key(model: Optional[str]) -> str:
    normalized = str(model or "").strip()
    if not normalized:
        return ""

    # Accept legacy/formatted UI labels like "Gemini: gemini-2.0-flash".
    if ":" in normalized and " " in normalized:
        maybe_model = normalized.split(":", 1)[1].strip()
        if maybe_model:
            normalized = maybe_model

    # Backward-compatible alias.
    if normalized == "openrouter/claude":
        normalized = "anthropic/claude-3.5-sonnet"

    # Case-insensitive canonicalization to configured keys.
    lower_map = {key.lower(): key for key in AVAILABLE_MODELS.keys()}
    return lower_map.get(normalized.lower(), normalized)


def _provider_key_error(
    provider: str,
    gemini_api_key: Optional[str],
    openrouter_api_key: Optional[str],
    openai_api_key: Optional[str],
) -> Optional[str]:
    if provider == "gemini" and not (gemini_api_key or "").strip():
        return "Gemini API key is required when provider is 'gemini'."
    if provider == "openrouter" and not (openrouter_api_key or "").strip():
        return "OpenRouter API key is required when provider is 'openrouter'."
    if provider == "openai" and not (openai_api_key or "").strip():
        return "OpenAI API key is required for OpenAI models."
    return None


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
    provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None


class GenerateRequest(BaseModel):
    topic: Optional[str] = None
    prompt: Optional[str] = None
    profileId: Optional[str] = None
    profile_id: Optional[str] = None
    length: Optional[Union[int, str]] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None
    provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None


class AnalogyRequest(BaseModel):
    text: str
    domain: str
    model: Optional[str] = None
    model_name: Optional[str] = "gemma3:1b"
    use_local: bool = True
    provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None


class TransferRequest(BaseModel):
    text: str
    profileId: Optional[str] = None
    profile_id: Optional[str] = None
    instructions: str = ""
    options: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None
    provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None


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
    profile_a_data: Optional[Dict[str, Any]] = None
    profile_b_data: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None


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
    model = _normalize_model_key(body.model)
    try:
        provider = _resolve_provider(body.provider, model)
    except ValueError as exc:
        return failure(str(exc))
    if model not in AVAILABLE_MODELS and provider == "ollama":
        return failure(
            f"Unknown model: {model}. Available models: {', '.join(AVAILABLE_MODELS.keys())}"
        )
    body.model = model
    print("Analyze request model:", model)
    print("Available models:", list(AVAILABLE_MODELS.keys()))
    if body.mode not in PROCESSING_MODES:
        return failure(f"Unknown mode: {body.mode}")
    key_error = _provider_key_error(provider, body.gemini_api_key, body.openrouter_api_key, body.openai_api_key)
    if key_error:
        return failure(key_error)
    if len(body.text.split()) < 20:
        return failure("Please provide more input text for analysis.")

    model_id = _resolve_model_id(body.model)
    use_local = provider == "ollama"
    api_type = None if use_local else provider
    api_client = None
    if provider == "gemini":
        api_client = (body.gemini_api_key or "").strip()
    elif provider == "openrouter":
        api_client = (body.openrouter_api_key or "").strip()
    elif provider == "openai":
        api_client = (body.openai_api_key or "").strip()

    async def stream() -> AsyncGenerator[str, None]:
        for pass_name in PASS_ORDER:
            yield json.dumps({"type": "pass", "pass": pass_name, "status": "running"}) + "\n"

        loop = asyncio.get_running_loop()
        analysis_future = loop.run_in_executor(
            None,
            analyze_style,
            body.text,
            use_local,
            model_id,
            api_type,
            api_client,
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
    model_key = _normalize_model_key(str(body.model or options.get("model") or "gemma3:1b"))
    try:
        provider = _resolve_provider(body.provider, model_key)
    except ValueError as exc:
        return failure(str(exc))
    if model_key not in AVAILABLE_MODELS and provider == "ollama":
        return failure(
            f"Unknown model: {model_key}. Available models: {', '.join(AVAILABLE_MODELS.keys())}"
        )
    key_error = _provider_key_error(provider, body.gemini_api_key, body.openrouter_api_key, body.openai_api_key)
    if key_error:
        return failure(key_error)
    model_id = _resolve_model_id(model_key)
    use_local = provider == "ollama"
    api_type = None if use_local else provider
    api_client = None
    if provider == "gemini":
        api_client = (body.gemini_api_key or "").strip()
    elif provider == "openrouter":
        api_client = (body.openrouter_api_key or "").strip()
    elif provider == "openai":
        api_client = (body.openai_api_key or "").strip()

    length_value = options.get("length", body.length if body.length is not None else 500)
    if isinstance(length_value, str):
        length_map = {"short": 300, "medium": 500, "long": 900}
        target_length = length_map.get(length_value.lower(), 500)
    else:
        target_length = int(length_value)

    content_type = _normalize_content_type(options.get("contentType", "article"))
    tone = options.get("tone", "neutral")
    context = options.get("context", "")

    async def stream() -> AsyncGenerator[str, None]:
        generator = ContentGenerator()
        generated = await run_in_threadpool(
            generator.generate_content,
            profile,
            content_type,
            topic,
            target_length,
            tone,
            context,
            use_local,
            model_id,
            api_type,
            api_client,
        )

        text = str(generated.get("generated_content") or "").strip()
        print("Generated text length:", len(text))

        if generated.get("error"):
            yield f"Generation failed: {generated.get('error')}"
            return

        if not text:
            yield "No content was generated."
            return

        for token in text.split(" "):
            yield token + " "

        if text:
            content_data = {
                "style_analysis_id": profile_id,
                "content_type": content_type,
                "topic": topic,
                "content": text,
                "target_length": target_length,
                "actual_length": len(text.split()),
                "tone": tone,
                "model_used": model_key,
                "quality_metrics": generated.get("quality_metrics", {}),
                "style_adherence_score": generated.get("style_adherence_score", 0.0),
            }
            await run_in_threadpool(save_generated_content, auth.access_token, auth.user_id, content_data)

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


@app.post("/api/analogy")
async def api_analogy(body: AnalogyRequest):
    if not body.text or not body.text.strip():
        return failure("Please provide input text for analogy generation.", status_code=400)

    if body.domain not in ANALOGY_DOMAINS:
        return failure(f"Invalid domain. Must be one of: {list(ANALOGY_DOMAINS.keys())}", status_code=400)

    model_key = _normalize_model_key(str(body.model or body.model_name or "gemma3:1b"))
    provider_input = body.provider
    if not provider_input:
        provider_input = "ollama" if body.use_local else _detect_provider_from_model(model_key)

    try:
        provider = _resolve_provider(provider_input, model_key)
    except ValueError as exc:
        return failure(str(exc))

    key_error = _provider_key_error(provider, body.gemini_api_key, body.openrouter_api_key, body.openai_api_key)
    if key_error:
        return failure(key_error)

    use_local = provider == "ollama"
    if use_local and not is_ollama_installed():
        return failure("Model unavailable — is Ollama running?", status_code=503)

    api_type = None if use_local else provider
    api_client = None
    if provider == "gemini":
        api_client = (body.gemini_api_key or "").strip()
    elif provider == "openrouter":
        api_client = (body.openrouter_api_key or "").strip()
    elif provider == "openai":
        api_client = (body.openai_api_key or "").strip()

    try:
        injector = AnalogyInjector(domain=body.domain)
        result = await run_in_threadpool(
            injector.augment_text,
            body.text,
            use_local=use_local,
            model_name=_resolve_model_id(model_key),
            api_type=api_type,
            api_client=api_client,
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
    model_key = _normalize_model_key(str(body.model or options.get("model") or "gemma3:1b"))
    try:
        provider = _resolve_provider(body.provider, model_key)
    except ValueError as exc:
        return failure(str(exc))
    if model_key not in AVAILABLE_MODELS and provider == "ollama":
        return failure(
            f"Unknown model: {model_key}. Available models: {', '.join(AVAILABLE_MODELS.keys())}"
        )
    key_error = _provider_key_error(provider, body.gemini_api_key, body.openrouter_api_key, body.openai_api_key)
    if key_error:
        return failure(key_error)
    use_local = provider == "ollama"
    api_type = None if use_local else provider
    api_client = None
    if provider == "gemini":
        api_client = (body.gemini_api_key or "").strip()
    elif provider == "openrouter":
        api_client = (body.openrouter_api_key or "").strip()
    elif provider == "openai":
        api_client = (body.openai_api_key or "").strip()

    styler = StyleTransfer()
    result = await run_in_threadpool(
        styler.transfer_style,
        body.text,
        profile,
        options.get("transferType", "direct_transfer"),
        float(options.get("intensity", 1.0)),
        options.get("preserveElements", []),
        use_local,
        _resolve_model_id(model_key),
        api_type,
        api_client,
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
        payload["metadata"] = {"processing_mode": "enhanced", "model_used": "gemini-1.5-flash"}

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
    model_key = _normalize_model_key(str(body.model or "gemma3:1b"))
    try:
        provider = _resolve_provider(body.provider, model_key)
    except ValueError as exc:
        return failure(str(exc))
    key_error = _provider_key_error(provider, body.gemini_api_key, body.openrouter_api_key, body.openai_api_key)
    if key_error:
        return failure(key_error)

    profile_a_id = body.profileAId or body.profile_a_id
    profile_b_id = body.profileBId or body.profile_b_id
    if profile_a_id and profile_b_id:
        a = await run_in_threadpool(get_analysis, auth.access_token, auth.user_id, profile_a_id)
        b = await run_in_threadpool(get_analysis, auth.access_token, auth.user_id, profile_b_id)
        if not a.get("success") or not b.get("success"):
            return failure("One or both profiles not found", status_code=404)
        data_a = a.get("data") or {}
        data_b = b.get("data") or {}
    elif body.profile_a_data and body.profile_b_data:
        data_a = body.profile_a_data
        data_b = body.profile_b_data
    else:
        return failure("Provide either both profile IDs or both profile data payloads.")

    text_a = data_a.get("style_fingerprint_summary") or json.dumps(data_a.get("consolidated_analysis", {}))
    text_b = data_b.get("style_fingerprint_summary") or json.dumps(data_b.get("consolidated_analysis", {}))

    deep_a = extract_deep_stylometry(text_a)
    deep_b = extract_deep_stylometry(text_b)
    similarity = calculate_style_similarity(deep_a, deep_b)

    saved = None
    if profile_a_id and profile_b_id:
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
        "saved": saved.get("data") if saved else None,
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
        "backend_build": "20260511-gemini-rest-v2",
        "gemini_transport": "rest",
    })


@app.get("/api/models")
async def api_models():
    return success({
        "models": AVAILABLE_MODELS,
        "model_keys": list(AVAILABLE_MODELS.keys()),
        "gemini_transport": "rest",
    })


@app.get("/")
async def root():
    return success({"service": "Stylomex API", "status": "ok"})
