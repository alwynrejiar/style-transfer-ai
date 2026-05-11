import { getToken, showAuthModal, signOutSession } from "./auth.js?v=20260504-logo-v2";

const DEFAULT_AI_MODEL_STORAGE_KEY = "default_ai_model";
const GEMINI_API_KEY_STORAGE_KEY = "gemini_api_key";
const OPENROUTER_API_KEY_STORAGE_KEY = "openrouter_api_key";
const OPENAI_API_KEY_STORAGE_KEY = "openai_api_key";
const SETTINGS_STORAGE_KEY = "stylomex.settings.v1";

function authHeaders(extra = {}) {
  const token = getToken();
  const headers = { ...extra };
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

async function parseResponse(response) {
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = { success: false, error: text || "Invalid server response" };
  }

  if (response.status === 401) {
    showAuthModal();
    throw new Error("Your session has expired. Please sign in again.");
  }

  if (!response.ok || payload.success === false) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }

  return payload.data;
}

function extractErrorMessage(rawText, fallbackMessage) {
  if (!rawText) return fallbackMessage;
  try {
    const payload = JSON.parse(rawText);
    if (payload && typeof payload === "object") {
      return payload.error || payload.message || fallbackMessage;
    }
  } catch {
    // Keep raw text fallback.
  }
  return rawText;
}

function detectProviderFromModel(model) {
  const normalized = String(model || "").trim().toLowerCase();
  if (normalized.startsWith("gemini")) return "gemini";
  if (
    normalized.startsWith("anthropic/") ||
    normalized.startsWith("deepseek/") ||
    normalized.startsWith("meta-llama/") ||
    normalized.includes("claude")
  ) return "openrouter";
  if (normalized.startsWith("gpt-")) return "openai";
  return "ollama";
}

function readSavedDefaultModel() {
  const direct = String(localStorage.getItem(DEFAULT_AI_MODEL_STORAGE_KEY) || "").trim();
  if (direct) return direct;

  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return "";
    const parsed = JSON.parse(raw);
    return String(parsed?.defaultModel || "").trim();
  } catch {
    return "";
  }
}

function normalizeModelValue(model) {
  let raw = String(model || "").trim();
  if (!raw) return "";

  // Accept previously saved label-like values such as "Gemini: gemini-2.0-flash".
  if (raw.includes(":") && raw.includes(" ")) {
    const maybeModel = raw.split(":", 2)[1]?.trim();
    if (maybeModel) raw = maybeModel;
  }

  if (raw === "openrouter/claude") return "anthropic/claude-3.5-sonnet";
  return raw;
}

export function getAIConfig() {
  const model = normalizeModelValue(readSavedDefaultModel() || "gemini-1.5-flash");
  const provider = detectProviderFromModel(model);
  const geminiApiKey = String(localStorage.getItem(GEMINI_API_KEY_STORAGE_KEY) || "").trim();
  const openrouterApiKey = String(localStorage.getItem(OPENROUTER_API_KEY_STORAGE_KEY) || "").trim();
  const openaiApiKey = String(localStorage.getItem(OPENAI_API_KEY_STORAGE_KEY) || "").trim();

  return {
    model,
    provider,
    gemini_api_key: provider === "gemini" ? geminiApiKey : null,
    openrouter_api_key: provider === "openrouter" ? openrouterApiKey : null,
    openai_api_key: provider === "openai" ? openaiApiKey : null,
  };
}

export async function apiGet(path) {
  const response = await fetch(path, {
    headers: authHeaders(),
  });
  return parseResponse(response);
}

export async function apiPost(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body || {}),
  });
  return parseResponse(response);
}

export async function apiPatch(path, body) {
  const response = await fetch(path, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body || {}),
  });
  return parseResponse(response);
}

export async function apiDelete(path) {
  const response = await fetch(path, {
    method: "DELETE",
    headers: authHeaders(),
  });
  return parseResponse(response);
}

export async function streamAnalyze(payload, handlers) {
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });

  if (response.status === 401) {
    showAuthModal();
    throw new Error("Please sign in first.");
  }

  if (!response.ok || !response.body) {
    const text = await response.text();
    throw new Error(extractErrorMessage(text, "Unable to analyze text"));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatchEvent = (event) => {
    if (event.type === "pass") {
      handlers?.onPass?.(event);
    } else if (event.type === "result") {
      handlers?.onResult?.(event.data);
    } else if (event.type === "error") {
      handlers?.onError?.(event.error || "Analysis failed");
    } else if (event.type === "heartbeat") {
      handlers?.onProgress?.(event);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      const tail = buffer.trim();
      if (tail) {
        try {
          dispatchEvent(JSON.parse(tail));
        } catch {
          // Ignore final malformed fragment.
        }
      }
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) continue;
      let event;
      try {
        event = JSON.parse(line);
      } catch {
        continue;
      }
      dispatchEvent(event);
    }
  }
}

export async function streamGenerate(payload, handlers) {
  const response = await fetch("/api/generate", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload),
  });

  if (response.status === 401) {
    showAuthModal();
    throw new Error("Please sign in first.");
  }

  if (!response.ok || !response.body) {
    const text = await response.text();
    throw new Error(extractErrorMessage(text, "Unable to generate content"));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let generatedText = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      const tail = decoder.decode();
      if (tail) {
        generatedText += tail;
        handlers?.onToken?.(tail);
      }
      break;
    }
    const chunk = decoder.decode(value, { stream: true });
    if (chunk) {
      generatedText += chunk;
      handlers?.onToken?.(chunk);
    }
  }

  handlers?.onDone?.();
  return generatedText;
}

export async function ensureHealthy() {
  try {
    await apiGet("/api/health");
    return true;
  } catch {
    return false;
  }
}

export { signOutSession };











