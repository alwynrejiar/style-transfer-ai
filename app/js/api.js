import { getToken, showAuthModal, signOutSession } from "./auth.js?v=20260324-google-auth-v11";

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
    throw new Error(text || "Unable to analyze text");
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
    throw new Error(text || "Unable to generate content");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    if (chunk) handlers?.onToken?.(chunk);
  }

  handlers?.onDone?.();
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










