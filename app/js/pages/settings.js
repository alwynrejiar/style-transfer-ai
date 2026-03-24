import { apiGet, apiPost } from "../api.js?v=20260324-google-auth-v14";
import { getSession } from "../auth.js?v=20260324-google-auth-v14";

const SETTINGS_STORAGE_KEY = "stylomex.settings.v1";

const DEFAULT_SETTINGS = {
  defaultModel: "gemma3:1b",
  temperature: 0.7,
  systemInstructions: "",
  analysisDepth: "standard",
  autoDeleteSourceText: false,
  localOnlyMode: true,
};

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_SETTINGS, ...(parsed || {}) };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function saveSettings(nextSettings) {
  localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(nextSettings));
}

function showInlineToast(node, message, type = "ok") {
  if (!node) return;
  node.className = `toast ${type}`;
  node.textContent = message;
  node.classList.remove("hidden");
  window.setTimeout(() => node.classList.add("hidden"), 2400);
}

function collectFormSettings(form) {
  const formData = new FormData(form);
  return {
    defaultModel: String(formData.get("defaultModel") || DEFAULT_SETTINGS.defaultModel),
    temperature: Number(formData.get("temperature") || DEFAULT_SETTINGS.temperature),
    systemInstructions: String(formData.get("systemInstructions") || "").trim(),
    analysisDepth: String(formData.get("analysisDepth") || DEFAULT_SETTINGS.analysisDepth),
    autoDeleteSourceText: Boolean(form.querySelector("#auto-delete-source")?.checked),
    localOnlyMode: Boolean(form.querySelector("#local-only-mode")?.checked),
  };
}

export async function mountSettingsPage(root) {
  const session = getSession() || {};
  const settings = loadSettings();
  let currentEmail = session.email || "";

  try {
    const me = await apiGet("/api/auth/me");
    currentEmail = me?.email || currentEmail;
  } catch {
    // Settings route can be viewed signed out; account actions will require auth.
  }

  root.innerHTML = `
    <section class="container page-enter">
      <header class="page-head">
        <h1 class="page-title">Settings</h1>
        <p class="page-subtitle">Configure your application preferences.</p>
      </header>

      <form id="settings-form" class="settings-layout">
        <section class="settings-card">
          <h3 class="settings-title">AI & Model Configuration</h3>
          <hr class="settings-rule" />

          <label for="default-model">Default Model Selection</label>
          <select id="default-model" name="defaultModel" class="settings-select">
            <option value="gemma3:1b" ${settings.defaultModel === "gemma3:1b" ? "selected" : ""}>gemma3:1b</option>
            <option value="phi-4" ${settings.defaultModel === "phi-4" ? "selected" : ""}>phi-4</option>
            <option value="llama3:8b" ${settings.defaultModel === "llama3:8b" ? "selected" : ""}>llama3:8b</option>
            <option value="deepseek-v3" ${settings.defaultModel === "deepseek-v3" ? "selected" : ""}>deepseek-v3</option>
          </select>

          <div class="settings-slider-row">
            <label for="temperature">Creativity (Temperature)</label>
            <span id="temperature-value" class="settings-value">${Number(settings.temperature).toFixed(1)}</span>
          </div>
          <input id="temperature" name="temperature" type="range" class="settings-slider" min="0" max="1" step="0.1" value="${Number(settings.temperature).toFixed(1)}" />

          <label for="system-instructions">System Instructions (Global Style Bias)</label>
          <textarea id="system-instructions" class="settings-textarea" name="systemInstructions" rows="4" placeholder="e.g., Always write in a professional tone. Use Oxford commas.">${settings.systemInstructions}</textarea>
        </section>

        <section class="settings-card">
          <h3 class="settings-title">Stylometry & Privacy</h3>
          <hr class="settings-rule" />

          <label>Analysis Depth</label>
          <div class="settings-radio-group">
            <label class="settings-radio-label">
              <input type="radio" name="analysisDepth" value="standard" ${settings.analysisDepth === "standard" ? "checked" : ""} />
              <span>Standard</span>
            </label>
            <label class="settings-radio-label">
              <input type="radio" name="analysisDepth" value="enhanced" ${settings.analysisDepth === "enhanced" ? "checked" : ""} />
              <span>Enhanced (Multi-pass)</span>
            </label>
          </div>

          <label class="settings-checkbox-container" for="auto-delete-source">
            <span>Auto-delete source text after profile creation</span>
            <input id="auto-delete-source" type="checkbox" ${settings.autoDeleteSourceText ? "checked" : ""} />
          </label>

          <label class="settings-checkbox-container" for="local-only-mode">
            <span>Ollama Local-Only</span>
            <input id="local-only-mode" type="checkbox" ${settings.localOnlyMode ? "checked" : ""} />
          </label>
        </section>

        <section class="settings-card">
          <h3 class="settings-title">Account & Security</h3>
          <hr class="settings-rule" />

          <label for="account-email">Profile Info (Email)</label>
          <input id="account-email" type="text" value="${currentEmail || "Sign in required"}" readonly />

          <button id="change-password-btn" type="button" class="settings-btn">Change Password</button>
        </section>

        <section class="settings-card settings-danger-card">
          <h3 class="settings-title settings-danger-title">Danger Zone</h3>
          <p class="settings-danger-text">Delete Account: Permanently remove your account, all saved stylometric profiles, and generation history. This action cannot be undone.</p>
          <label for="confirm-delete-email">Type your email to confirm</label>
          <input id="confirm-delete-email" type="email" class="settings-input" placeholder="${currentEmail || "you@example.com"}" />
          <button id="delete-account-btn" type="button" class="settings-btn settings-delete-btn">Delete My Account</button>
        </section>

        <div class="settings-actions">
          <button id="save-settings-btn" type="submit" class="settings-btn settings-btn-primary">Save Changes</button>
          <div id="settings-toast" class="toast hidden"></div>
        </div>
      </form>

      <div id="password-modal" class="settings-modal-overlay hidden" aria-hidden="true">
        <section class="settings-modal-card">
          <h3 class="settings-title">Change Password</h3>
          <form id="password-form" class="stack-form">
            <label for="current-password">Current Password</label>
            <input id="current-password" name="currentPassword" type="password" class="settings-input" required />

            <label for="new-password">New Password</label>
            <input id="new-password" name="newPassword" type="password" class="settings-input" required />

            <label for="confirm-new-password">Confirm New Password</label>
            <input id="confirm-new-password" name="confirmNewPassword" type="password" class="settings-input" required />

            <div class="settings-modal-actions">
              <button type="submit" class="settings-btn settings-btn-primary">Update Password</button>
              <button id="close-password-modal" type="button" class="settings-btn">Cancel</button>
            </div>
            <div id="password-toast" class="toast hidden"></div>
          </form>
        </section>
      </div>
    </section>
  `;

  const form = root.querySelector("#settings-form");
  const temperatureInput = root.querySelector("#temperature");
  const temperatureValue = root.querySelector("#temperature-value");
  const saveToast = root.querySelector("#settings-toast");

  const changePasswordBtn = root.querySelector("#change-password-btn");
  const passwordModal = root.querySelector("#password-modal");
  const closePasswordModalBtn = root.querySelector("#close-password-modal");
  const passwordForm = root.querySelector("#password-form");
  const passwordToast = root.querySelector("#password-toast");

  const deleteButton = root.querySelector("#delete-account-btn");
  const confirmDeleteEmail = root.querySelector("#confirm-delete-email");

  temperatureInput?.addEventListener("input", () => {
    if (temperatureValue && temperatureInput) {
      temperatureValue.textContent = Number(temperatureInput.value).toFixed(1);
    }
  });

  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const nextSettings = collectFormSettings(form);
    saveSettings(nextSettings);
    showInlineToast(saveToast, "Settings Updated Successfully", "ok");
  });

  function togglePasswordModal(show) {
    if (!passwordModal) return;
    passwordModal.classList.toggle("hidden", !show);
    passwordModal.setAttribute("aria-hidden", show ? "false" : "true");
  }

  changePasswordBtn?.addEventListener("click", () => togglePasswordModal(true));
  closePasswordModalBtn?.addEventListener("click", () => togglePasswordModal(false));

  passwordForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const data = new FormData(passwordForm);
    const currentPassword = String(data.get("currentPassword") || "").trim();
    const newPassword = String(data.get("newPassword") || "").trim();
    const confirmNewPassword = String(data.get("confirmNewPassword") || "").trim();

    if (!currentPassword || !newPassword || !confirmNewPassword) {
      showInlineToast(passwordToast, "Please fill in all password fields.", "err");
      return;
    }

    try {
      await apiPost("/api/auth/password", {
        current_password: currentPassword,
        new_password: newPassword,
        confirm_new_password: confirmNewPassword,
        refresh_token: getSession()?.refresh_token || "",
      });
      showInlineToast(passwordToast, "Password updated successfully.", "ok");
      passwordForm.reset();
      window.setTimeout(() => togglePasswordModal(false), 600);
    } catch (error) {
      showInlineToast(passwordToast, error.message || "Failed to update password.", "err");
    }
  });

  deleteButton?.addEventListener("click", async () => {
    const typedEmail = String(confirmDeleteEmail?.value || "").trim().toLowerCase();
    const expectedEmail = String(currentEmail || "").trim().toLowerCase();

    if (!expectedEmail) {
      showInlineToast(saveToast, "Sign in required to delete account.", "err");
      return;
    }

    if (!typedEmail || typedEmail !== expectedEmail) {
      showInlineToast(saveToast, "Type your exact account email to confirm deletion.", "err");
      return;
    }

    try {
      await apiPost("/api/account/delete", { confirm_email: typedEmail });
      sessionStorage.removeItem("stylomex.session");
      localStorage.removeItem(SETTINGS_STORAGE_KEY);
      window.location.href = "/docs/index.html";
    } catch (error) {
      showInlineToast(saveToast, error.message || "Failed to delete account.", "err");
    }
  });
}














