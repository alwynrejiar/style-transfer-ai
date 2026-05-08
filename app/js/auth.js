const STORAGE_KEY = "stylomex.session";
const SUPABASE_CDN = "https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/dist/umd/supabase.min.js";

let session = null;
let supabaseClient = null;

function consumeOAuthHashSession() {
  const rawHash = window.location.hash || "";
  if (!rawHash.includes("access_token=")) return;

  const params = new URLSearchParams(rawHash.replace(/^#/, ""));
  const accessToken = params.get("access_token") || "";
  if (!accessToken) return;

  const refreshToken = params.get("refresh_token") || "";
  const expiresIn = Number(params.get("expires_in") || 0);
  const now = Math.floor(Date.now() / 1000);

  saveSession({
    access_token: accessToken,
    refresh_token: refreshToken,
    expires_at: expiresIn ? now + expiresIn : undefined,
  });

  // Clear OAuth fragment and restore app routing hash.
  window.location.hash = "#/analyze";
}

function emitAuthChange() {
  window.dispatchEvent(new CustomEvent("auth:change", { detail: session }));
}

function saveSession(nextSession) {
  session = nextSession;
  if (nextSession) {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(nextSession));
  } else {
    sessionStorage.removeItem(STORAGE_KEY);
  }
  emitAuthChange();
}

function restoreSession() {
  if (session) return;
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    session = raw ? JSON.parse(raw) : null;
  } catch {
    session = null;
  }
}

export async function ensureSupabase() {
  if (window.supabase?.createClient && supabaseClient) return supabaseClient;
  if (!window.supabase?.createClient) {
    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = SUPABASE_CDN;
      script.async = true;
      script.onload = resolve;
      script.onerror = () => reject(new Error("Failed to load Supabase SDK"));
      document.head.appendChild(script);
    });
  }

  const url = window.SUPABASE_URL || "";
  const key = window.SUPABASE_ANON_KEY || "";
  if (url && key && window.supabase?.createClient) {
    supabaseClient = window.supabase.createClient(url, key);
  }
  return supabaseClient;
}

restoreSession();
consumeOAuthHashSession();
ensureSupabase().catch(() => {});

export function getSession() {
  restoreSession();
  return session;
}

export function getToken() {
  return getSession()?.access_token || "";
}

export function isAuthenticated() {
  return Boolean(getToken());
}

function getModalRoot() {
  const root = document.getElementById("auth-modal");
  if (!root) throw new Error("Missing #auth-modal root");
  return root;
}

function renderAuthModal(mode = "signin") {
  const root = getModalRoot();

  root.hidden = false;
  root.innerHTML = `
    <div class="auth-modal-overlay" role="dialog" aria-modal="true" aria-labelledby="auth-title">
      <section class="auth-modal-card">
        <h2 id="auth-title" class="page-title" style="font-size:2rem;margin-bottom:8px">Welcome to Stylomex.AI</h2>
        <p class="page-subtitle">Sign in to continue to your writing workspace.</p>

        <div class="auth-tabs">
          <button class="auth-tab ${mode === "signin" ? "active" : ""}" data-tab="signin">Sign in</button>
          <button class="auth-tab ${mode === "signup" ? "active" : ""}" data-tab="signup">Create account</button>
        </div>

        <button type="button" class="btn auth-google-btn" id="auth-google">
          <span class="auth-google-icon" aria-hidden="true">
            <svg viewBox="0 0 48 48" width="18" height="18" role="img" focusable="false">
              <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.7 17.74 9.5 24 9.5z"/>
              <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
              <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
              <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
              <path fill="none" d="M0 0h48v48H0z"/>
            </svg>
          </span>
          <span>Continue with Google</span>
        </button>

        <div class="auth-divider"><span>or continue with email</span></div>

        <form id="auth-form" class="stack-form" novalidate>
          <div id="auth-name-wrap" class="stack-form ${mode === "signup" ? "" : "hidden"}">
            <label for="auth-name" style="margin-top: 0;">Name</label>
            <input id="auth-name" name="name" type="text" placeholder="Your name" />
          </div>

          <label for="auth-email">Email</label>
          <input id="auth-email" name="email" type="email" placeholder="you@example.com" required />

          <label for="auth-password">Password</label>
          <input id="auth-password" name="password" type="password" placeholder="********" required />

          <div style="display:flex;gap:8px;margin-top:12px">
            <button type="submit" class="btn btn-dark" id="auth-submit">${mode === "signup" ? "Create account" : "Sign in"}</button>
            <button type="button" class="btn" id="auth-close">Cancel</button>
          </div>
        </form>

        <div id="auth-feedback" class="toast hidden"></div>
      </section>
    </div>
  `;

  const tabs = root.querySelectorAll(".auth-tab");
  let googleBtn = root.querySelector("#auth-google");
  const nameWrap = root.querySelector("#auth-name-wrap");
  const submitBtn = root.querySelector("#auth-submit");
  const feedback = root.querySelector("#auth-feedback");
  const form = root.querySelector("#auth-form");

  if (!googleBtn) {
    const tabsWrap = root.querySelector(".auth-tabs");
    if (tabsWrap) {
      tabsWrap.insertAdjacentHTML(
        "afterend",
        `
          <button type="button" class="btn auth-google-btn" id="auth-google">
            <span class="auth-google-icon" aria-hidden="true">
              <svg viewBox="0 0 48 48" width="18" height="18" role="img" focusable="false">
                <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.7 17.74 9.5 24 9.5z"/>
                <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                <path fill="none" d="M0 0h48v48H0z"/>
              </svg>
            </span>
            <span>Continue with Google</span>
          </button>
          <div class="auth-divider"><span>or continue with email</span></div>
        `,
      );
      googleBtn = root.querySelector("#auth-google");
    }
  }

  let currentMode = mode;

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      currentMode = tab.dataset.tab;
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      nameWrap.classList.toggle("hidden", currentMode !== "signup");
      submitBtn.textContent = currentMode === "signup" ? "Create account" : "Sign in";
      feedback.className = "toast hidden";
      feedback.textContent = "";
    });
  });

  root.querySelector("#auth-close")?.addEventListener("click", hideAuthModal);

  googleBtn?.addEventListener("click", async () => {
    feedback.className = "toast hidden";
    feedback.textContent = "";

    try {
      const redirectTo = `${window.location.origin}/app/`;
      const response = await fetch("/api/auth/google/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ redirect_to: redirectTo }),
      });
      const result = await response.json();

      if (!response.ok || !result.success || !result?.data?.url) {
        throw new Error(result.error || "Unable to start Google sign in.");
      }

      window.location.href = result.data.url;
    } catch (error) {
      feedback.className = "toast err";
      feedback.textContent = error.message || "Google sign in failed.";
    }
  });

  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    feedback.className = "toast hidden";
    feedback.textContent = "";

    const data = new FormData(form);
    const payload = {
      email: String(data.get("email") || "").trim(),
      password: String(data.get("password") || ""),
      name: String(data.get("name") || "").trim(),
    };

    if (!payload.email || !payload.password) {
      feedback.className = "toast err";
      feedback.textContent = "Email and password are required.";
      return;
    }

    try {
      const endpoint = currentMode === "signup" ? "/api/auth/signup" : "/api/auth/signin";
      const requestBody = currentMode === "signup"
        ? { email: payload.email, password: payload.password, name: payload.name }
        : { email: payload.email, password: payload.password };

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });

      const result = await response.json();
      if (!response.ok || !result.success) {
        throw new Error(result.error || "Authentication failed");
      }

      if (currentMode === "signup") {
        feedback.className = "toast ok";
        feedback.textContent = "Account created. Please sign in now.";
        return;
      }

      saveSession(result.data);
      hideAuthModal();
    } catch (error) {
      feedback.className = "toast err";
      feedback.textContent = error.message || "Authentication failed.";
    }
  });
}

export function showAuthModal() {
  renderAuthModal("signin");
}

export function hideAuthModal() {
  const root = getModalRoot();
  root.hidden = true;
  root.innerHTML = "";
}

export async function signInWithCredentials(email, password) {
  const response = await fetch("/api/auth/signin", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const result = await response.json();
  if (!response.ok || !result.success) throw new Error(result.error || "Sign in failed");
  saveSession(result.data);
  return result.data;
}

export async function signUpWithCredentials(email, password, name) {
  const response = await fetch("/api/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, name }),
  });
  const result = await response.json();
  if (!response.ok || !result.success) throw new Error(result.error || "Sign up failed");
  return result.data;
}

export async function signOutSession() {
  const token = getToken();
  try {
    await fetch("/api/auth/signout", {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
  } catch {
    // Ignore network errors and clear local auth state anyway.
  }
  saveSession(null);
  window.location.href = "/docs/index.html";
}









