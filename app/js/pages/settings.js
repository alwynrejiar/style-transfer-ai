import { apiGet, apiPost } from "../api.js?v=20260511-gemini-rest-v1";
import { getSession } from "../auth.js?v=20260504-logo-v2";

const SETTINGS_STORAGE_KEY = "stylomex.settings.v1";
const DEFAULT_AI_MODEL_STORAGE_KEY = "default_ai_model";
const GEMINI_API_KEY_STORAGE_KEY = "gemini_api_key";
const OPENROUTER_API_KEY_STORAGE_KEY = "openrouter_api_key";
const OPENAI_API_KEY_STORAGE_KEY = "openai_api_key";
const AI_TEMPERATURE_STORAGE_KEY = "ai_temperature";
const GLOBAL_SYSTEM_INSTRUCTION_STORAGE_KEY = "global_system_instruction";
const ANALYSIS_DEPTH_STORAGE_KEY = "analysis_depth";
const AUTO_DELETE_SOURCE_TEXT_STORAGE_KEY = "auto_delete_source_text";

const DEFAULT_SETTINGS = {
  defaultModel: "gemma3:1b",
  temperature: 0.7,
  systemInstructions: "",
  analysisDepth: "standard",
  autoDeleteSourceText: false,
};

const MODEL_OPTIONS = [
  { value: "gemma3:1b", label: "Local: gemma3:1b", provider: "ollama" },
  { value: "gemini-1.5-flash", label: "Gemini: gemini-1.5-flash", provider: "gemini" },
  { value: "gemini-2.0-flash", label: "Gemini: gemini-2.0-flash", provider: "gemini" },
  { value: "anthropic/claude-3.5-sonnet", label: "OpenRouter: anthropic/claude-3.5-sonnet", provider: "openrouter" },
  { value: "meta-llama/llama-3.3-70b-instruct:free", label: "OpenRouter: meta-llama/llama-3.3-70b-instruct:free", provider: "openrouter" },
  { value: "deepseek/deepseek-r1:free", label: "OpenRouter: deepseek/deepseek-r1:free", provider: "openrouter" },
  { value: "gpt-4o-mini", label: "OpenAI: gpt-4o-mini", provider: "openai" },
  { value: "gpt-4o", label: "OpenAI: gpt-4o", provider: "openai" },
  { value: "gpt-5.1", label: "OpenAI: gpt-5.1", provider: "openai" },
];

const PROVIDER_CONFIG = {
  gemini: {
    label: "Gemini",
    keyStorage: GEMINI_API_KEY_STORAGE_KEY,
    keyLabel: "Gemini API Key",
  },
  openrouter: {
    label: "OpenRouter",
    keyStorage: OPENROUTER_API_KEY_STORAGE_KEY,
    keyLabel: "OpenRouter API Key",
  },
  openai: {
    label: "OpenAI",
    keyStorage: OPENAI_API_KEY_STORAGE_KEY,
    keyLabel: "OpenAI API Key",
  },
  ollama: {
    label: "Local / Ollama",
    keyStorage: "",
    keyLabel: "",
  },
};

function normalizeModelValue(model) {
  let raw = String(model || "").trim();
  if (!raw) return "";
  if (raw.includes(":") && raw.includes(" ")) {
    const maybeModel = raw.split(":", 2)[1]?.trim();
    if (maybeModel) raw = maybeModel;
  }
  if (raw === "openrouter/claude") return "anthropic/claude-3.5-sonnet";
  return raw;
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw);
    const merged = { ...DEFAULT_SETTINGS, ...(parsed || {}) };
    const storedTemperature = localStorage.getItem(AI_TEMPERATURE_STORAGE_KEY);
    const storedInstruction = localStorage.getItem(GLOBAL_SYSTEM_INSTRUCTION_STORAGE_KEY);
    const storedAnalysisDepth = localStorage.getItem(ANALYSIS_DEPTH_STORAGE_KEY);
    const storedAutoDelete = localStorage.getItem(AUTO_DELETE_SOURCE_TEXT_STORAGE_KEY);

    if (storedTemperature !== null && storedTemperature !== "") {
      merged.temperature = Number(storedTemperature);
    }
    if (storedInstruction !== null) {
      merged.systemInstructions = String(storedInstruction || "").trim();
    }
    if (storedAnalysisDepth) {
      merged.analysisDepth = String(storedAnalysisDepth);
    }
    if (storedAutoDelete !== null && storedAutoDelete !== "") {
      merged.autoDeleteSourceText = storedAutoDelete === "true";
    }

    return merged;
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function saveSettings(nextSettings) {
  localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(nextSettings));
  localStorage.setItem(AI_TEMPERATURE_STORAGE_KEY, String(nextSettings.temperature));
  localStorage.setItem(GLOBAL_SYSTEM_INSTRUCTION_STORAGE_KEY, String(nextSettings.systemInstructions || "").trim());
  localStorage.setItem(ANALYSIS_DEPTH_STORAGE_KEY, String(nextSettings.analysisDepth || DEFAULT_SETTINGS.analysisDepth));
  localStorage.setItem(AUTO_DELETE_SOURCE_TEXT_STORAGE_KEY, String(Boolean(nextSettings.autoDeleteSourceText)));
}

function readStoredKey(storageKey) {
  return String(localStorage.getItem(storageKey) || "").trim();
}

function maskKeyForDisplay(value) {
  const key = String(value || "").trim();
  if (!key) return "";
  if (key.length <= 4) return "****";
  return `${key.slice(0, 4)}****`;
}

function refreshMaskedKeyInput(input, storageKey, fallbackPlaceholder) {
  if (!input) return;
  const key = readStoredKey(storageKey);
  input.value = "";
  input.placeholder = key ? `Saved: ${maskKeyForDisplay(key)}` : fallbackPlaceholder;
}

function showInlineToast(node, message, type = "ok") {
  if (!node) return;
  node.className = `toast ${type}`;
  node.textContent = message;
  node.classList.remove("hidden");
  window.setTimeout(() => node.classList.add("hidden"), 2400);
}

function getProviderForModel(model) {
  const normalized = normalizeModelValue(model);
  const lower = normalized.toLowerCase();
  if (lower.startsWith("gemini")) return "gemini";
  if (
    lower.startsWith("anthropic/") ||
    lower.startsWith("deepseek/") ||
    lower.startsWith("meta-llama/") ||
    lower.includes("claude")
  ) return "openrouter";
  if (lower.startsWith("gpt-")) return "openai";
  const match = MODEL_OPTIONS.find((option) => option.value === normalized);
  return match?.provider || "ollama";
}

function renderSettingsCard({ id, title, subtitle, body, isActive = false }) {
  const activeClass = isActive ? " active" : "";
  return `
    <section class="settings-card settings-panel${activeClass}" data-section="${id}">
      <header class="settings-card-head">
        <div>
          <p class="settings-eyebrow">${subtitle}</p>
          <h3 class="settings-title">${title}</h3>
        </div>
      </header>
      ${body}
    </section>
  `;
}

function renderSubsection({ title, description, body }) {
  return `
    <div class="settings-subsection">
      <div class="settings-subsection-head">
        <div>
          <h4>${title}</h4>
          ${description ? `<p class="settings-helper-text">${description}</p>` : ""}
        </div>
      </div>
      <div class="settings-subsection-body">
        ${body}
      </div>
    </div>
  `;
}

function renderToggleGroup(name, options, selectedValue) {
  const buttons = options
    .map(
      (option) => `
        <button
          type="button"
          class="settings-segment${option.value === selectedValue ? " active" : ""}"
          data-toggle-value="${option.value}"
          aria-pressed="${option.value === selectedValue ? "true" : "false"}"
        >
          ${option.label}
        </button>
      `,
    )
    .join("");

  const descriptions = options
    .map(
      (option) => `
        <div class="settings-toggle-description${option.value === selectedValue ? " active" : ""}" data-toggle-description="${option.value}">
          <strong>${option.label}</strong>
          <span>${option.description}</span>
        </div>
      `,
    )
    .join("");

  return `
    <div class="settings-toggle-group" data-toggle-group="${name}">
      <input type="hidden" name="${name}" value="${selectedValue}">
      <div class="settings-segmented" role="tablist" aria-label="${name}">
        ${buttons}
      </div>
      <div class="settings-toggle-descriptions">
        ${descriptions}
      </div>
    </div>
  `;
}

function collectFormSettings(form) {
  const formData = new FormData(form);
  return {
    defaultModel: String(formData.get("defaultModel") || DEFAULT_SETTINGS.defaultModel),
    temperature: Number(formData.get("temperature") || DEFAULT_SETTINGS.temperature),
    systemInstructions: String(formData.get("systemInstructions") || "").trim(),
    analysisDepth: String(formData.get("analysisDepth") || DEFAULT_SETTINGS.analysisDepth),
    autoDeleteSourceText: Boolean(form.querySelector("#auto-delete-source")?.checked),
  };
}

export async function mountSettingsPage(root) {
  const session = getSession() || {};
  const settings = loadSettings();
  settings.defaultModel = normalizeModelValue(settings.defaultModel) || DEFAULT_SETTINGS.defaultModel;
  const storedDefaultModel = normalizeModelValue(localStorage.getItem(DEFAULT_AI_MODEL_STORAGE_KEY));
  if (storedDefaultModel) settings.defaultModel = storedDefaultModel;
  const detectedProvider = getProviderForModel(settings.defaultModel);
  let currentEmail = session.email || "";
  const accountProvider =
    session.provider ||
    session?.user?.app_metadata?.provider ||
    session?.app_metadata?.provider ||
    "";

  try {
    const me = await apiGet("/api/auth/me");
    currentEmail = me?.email || currentEmail;
  } catch {
    // Settings route can be viewed signed out; account actions will require auth.
  }

  const modelOptions = MODEL_OPTIONS.filter((option) => option.provider === detectedProvider)
    .map(
      (option) =>
        `<option value="${option.value}" ${settings.defaultModel === option.value ? "selected" : ""}>${option.label}</option>`,
    )
    .join("");

  const analysisToggle = renderToggleGroup(
    "analysisDepth",
    [
      {
        value: "standard",
        label: "Standard",
        description: "Faster analysis with lower compute cost.",
      },
      {
        value: "enhanced",
        label: "Enhanced (Multi-pass)",
        description: "Runs multiple stylometric passes for deeper profiling.",
      },
    ],
    settings.analysisDepth,
  );

  const aiBody = [
    renderSubsection({
      title: "Model Selection",
      description: "Choose a provider, then select the model you want as your default.",
      body: `
        <div class="settings-field-grid">
          <div class="settings-field">
            <label for="model-provider">Model Provider</label>
            <select id="model-provider" class="settings-select">
              <option value="gemini" ${detectedProvider === "gemini" ? "selected" : ""}>Gemini</option>
              <option value="openrouter" ${detectedProvider === "openrouter" ? "selected" : ""}>OpenRouter</option>
              <option value="openai" ${detectedProvider === "openai" ? "selected" : ""}>OpenAI</option>
              <option value="ollama" ${detectedProvider === "ollama" ? "selected" : ""}>Local / Ollama</option>
            </select>
          </div>
          <div class="settings-field">
            <label for="default-model">Model</label>
            <select id="default-model" name="defaultModel" class="settings-select">${modelOptions}</select>
          </div>
        </div>

        <div class="settings-key-area" id="api-key-panel">
          <div class="settings-key-head">
            <label id="provider-key-label" for="provider-api-key">API Key</label>
            <span id="api-key-status" class="settings-status-badge"></span>
          </div>
          <input id="provider-api-key" type="password" class="settings-input" placeholder="Enter API key" autocomplete="off" />
          <p id="provider-key-note" class="settings-helper-text">Keys are stored locally in this browser and never sent to storage databases.</p>
          <div class="settings-inline-actions">
            <button id="save-api-keys-btn" type="button" class="settings-btn settings-btn-primary">Save API Key</button>
            <button id="clear-api-keys-btn" type="button" class="settings-btn">Clear API Key</button>
          </div>
          <div id="api-keys-toast" class="toast hidden"></div>
        </div>
      `,
    }),
    renderSubsection({
      title: "Creativity (Temperature)",
      description:
        "Lower values produce more analytical and deterministic output. Higher values produce more creative and imaginative output.",
      body: `
        <div class="settings-slider-row">
          <label for="temperature">Temperature</label>
          <span id="temperature-value" class="settings-value">${Number(settings.temperature).toFixed(1)}</span>
        </div>
        <input id="temperature" name="temperature" type="range" class="settings-slider" min="0" max="1" step="0.1" value="${Number(settings.temperature).toFixed(1)}" />
        <div class="settings-recommendations">
          <div class="settings-recommendations-title">Recommended Temperature</div>
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Temperature</th>
              </tr>
            </thead>
            <tbody>
              <tr><td>Analyze</td><td>0.2</td></tr>
              <tr><td>Compare</td><td>0.2–0.4</td></tr>
              <tr><td>Generate</td><td>0.7</td></tr>
              <tr><td>Student Analogy</td><td>0.8</td></tr>
            </tbody>
          </table>
        </div>
      `,
    }),
    renderSubsection({
      title: "System Instructions",
      description: "These instructions are injected into all AI tasks.",
      body: `
        <label for="system-instructions">Global System Instructions</label>
        <textarea id="system-instructions" class="settings-textarea" name="systemInstructions" rows="4" placeholder="e.g. Always write in a professional tone. Use Oxford commas.">${settings.systemInstructions}</textarea>
      `,
    }),
  ].join("");

  const analysisBody = [
    renderSubsection({
      title: "Analysis Mode",
      description: "Switch between fast standard processing or multi-pass depth.",
      body: analysisToggle,
    }),
    renderSubsection({
      title: "Privacy",
      description: "Automatically remove raw uploaded text after profile generation.",
      body: `
        <label class="settings-toggle" for="auto-delete-source">
          <input id="auto-delete-source" type="checkbox" ${settings.autoDeleteSourceText ? "checked" : ""} />
          <span class="settings-toggle-ui" aria-hidden="true"></span>
          <span class="settings-toggle-text">Auto-delete source text after profile creation</span>
        </label>
      `,
    }),
  ].join("");

  const accountBody = [
    renderSubsection({
      title: "Account Information",
      body: `
        <div class="settings-field-grid">
          <div class="settings-field">
            <label for="account-email">Email</label>
            <input id="account-email" type="text" value="${currentEmail || "Sign in required"}" readonly />
          </div>
          <div class="settings-field">
            <label for="account-provider">Provider</label>
            <input id="account-provider" type="text" value="${accountProvider || "-"}" readonly />
          </div>
        </div>
      `,
    }),
    renderSubsection({
      title: "Change Password",
      description: "Update your password for this account.",
      body: `
        <button id="change-password-btn" type="button" class="settings-btn">Change Password</button>
      `,
    }),
    renderSubsection({
      title: "Danger Zone",
      body: `
        <div class="settings-danger-card">
          <p class="settings-danger-text">Delete Account: Permanently remove your account, all saved stylometric profiles, and generation history. This action cannot be undone.</p>
          <label for="confirm-delete-email">Type your email to confirm</label>
          <input id="confirm-delete-email" type="email" class="settings-input" placeholder="${currentEmail || "you@example.com"}" />
          <button id="delete-account-btn" type="button" class="settings-btn settings-delete-btn">Delete My Account</button>
        </div>
      `,
    }),
  ].join("");

  root.innerHTML = `
    <section class="container page-enter">
      <header class="page-head">
        <h1 class="page-title">Settings</h1>
      </header>

      <form id="settings-form" class="settings-form">
        <div class="settings-shell">
          <aside class="settings-sidebar">
            <nav class="settings-nav" aria-label="Settings Sections">
              <button type="button" class="settings-nav-link active" data-section-link="ai">AI & Model Configuration</button>
              <button type="button" class="settings-nav-link" data-section-link="analysis">Analysis Depth</button>
              <button type="button" class="settings-nav-link" data-section-link="account">Account & Security</button>
            </nav>
          </aside>

          <section class="settings-content">
            <div class="settings-panels">
              ${renderSettingsCard({
    id: "ai",
    title: "AI & Model Configuration",
    subtitle: "&nbsp;",
    body: aiBody,
    isActive: true,
  })}
              ${renderSettingsCard({
    id: "analysis",
    title: "Analysis Depth",
    subtitle: "&nbsp;",
    body: analysisBody,
  })}
              ${renderSettingsCard({
    id: "account",
    title: "Account & Security",
    subtitle: "&nbsp;",
    body: accountBody,
  })}
            </div>

            <div class="settings-actions">
              <button id="save-settings-btn" type="submit" class="settings-btn settings-btn-primary">Save Changes</button>
              <div id="settings-toast" class="toast hidden"></div>
            </div>
          </section>
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
  const defaultModelSelect = root.querySelector("#default-model");
  const modelProviderSelect = root.querySelector("#model-provider");
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
  const providerKeyInput = root.querySelector("#provider-api-key");
  const apiKeyLabel = root.querySelector("#provider-key-label");
  const apiKeyStatus = root.querySelector("#api-key-status");
  const apiKeyPanel = root.querySelector("#api-key-panel");
  const apiKeyNote = root.querySelector("#provider-key-note");
  const saveApiKeysBtn = root.querySelector("#save-api-keys-btn");
  const clearApiKeysBtn = root.querySelector("#clear-api-keys-btn");
  const apiKeysToast = root.querySelector("#api-keys-toast");

  function buildModelOptions(provider, selectedValue) {
    const options = MODEL_OPTIONS.filter((option) => option.provider === provider);
    if (!options.length) return;
    defaultModelSelect.innerHTML = options
      .map(
        (option) =>
          `<option value="${option.value}" ${option.value === selectedValue ? "selected" : ""}>${option.label}</option>`,
      )
      .join("");
  }

  function getProviderConfig(provider) {
    return PROVIDER_CONFIG[provider] || PROVIDER_CONFIG.ollama;
  }

  function updateApiKeyPanel(provider, selectedModel = "") {
    const providerConfig = getProviderConfig(provider);
    if (!apiKeyPanel || !apiKeyLabel || !providerKeyInput) return;

    if (!providerConfig.keyStorage) {
      apiKeyPanel.classList.add("hidden");
      return;
    }

    apiKeyPanel.classList.remove("hidden");
    apiKeyLabel.textContent = providerConfig.keyLabel;
    refreshMaskedKeyInput(providerKeyInput, providerConfig.keyStorage, `Enter ${providerConfig.label} API key`);
    if (apiKeyNote) {
      const normalizedModel = normalizeModelValue(selectedModel || defaultModelSelect?.value || settings.defaultModel);
      if (provider === "openrouter" && normalizedModel.toLowerCase().startsWith("deepseek/")) {
        apiKeyNote.textContent = "Uses OpenRouter API access.";
      } else {
        apiKeyNote.textContent = "Keys are stored locally in this browser and never sent to storage databases.";
      }
    }

    const storedKey = readStoredKey(providerConfig.keyStorage);
    if (apiKeyStatus) {
      if (storedKey) {
        apiKeyStatus.textContent = `${providerConfig.label} key connected`;
        apiKeyStatus.classList.add("connected");
      } else {
        apiKeyStatus.textContent = "No key saved";
        apiKeyStatus.classList.remove("connected");
      }
    }
  }

  buildModelOptions(detectedProvider, settings.defaultModel);
  updateApiKeyPanel(detectedProvider, settings.defaultModel);

  modelProviderSelect?.addEventListener("change", () => {
    const provider = modelProviderSelect.value;
    const nextDefault = MODEL_OPTIONS.find((option) => option.provider === provider)?.value || settings.defaultModel;
    buildModelOptions(provider, nextDefault);
    updateApiKeyPanel(provider, nextDefault);
  });

  defaultModelSelect?.addEventListener("change", () => {
    const provider = modelProviderSelect?.value || getProviderForModel(defaultModelSelect.value);
    updateApiKeyPanel(provider, defaultModelSelect.value);
  });

  temperatureInput?.addEventListener("input", () => {
    if (temperatureValue && temperatureInput) {
      temperatureValue.textContent = Number(temperatureInput.value).toFixed(1);
    }
  });

  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const nextSettings = collectFormSettings(form);
    saveSettings(nextSettings);
    localStorage.setItem(DEFAULT_AI_MODEL_STORAGE_KEY, normalizeModelValue(nextSettings.defaultModel));
    showInlineToast(saveToast, "Settings Updated Successfully", "ok");
  });

  saveApiKeysBtn?.addEventListener("click", () => {
    const provider = modelProviderSelect?.value || detectedProvider;
    const providerConfig = getProviderConfig(provider);
    const keyValue = String(providerKeyInput?.value || "").trim();
    const selectedModel = normalizeModelValue(defaultModelSelect?.value || settings.defaultModel || DEFAULT_SETTINGS.defaultModel);

    if (!providerConfig.keyStorage) {
      showInlineToast(apiKeysToast, "Local models do not require an API key.", "ok");
      return;
    }

    if (keyValue) {
      localStorage.setItem(providerConfig.keyStorage, keyValue);
      showInlineToast(apiKeysToast, "API key saved locally.", "ok");
    } else {
      showInlineToast(apiKeysToast, "No key entered. Existing key kept.", "ok");
    }

    if (selectedModel) {
      localStorage.setItem(DEFAULT_AI_MODEL_STORAGE_KEY, selectedModel);
    }

    refreshMaskedKeyInput(providerKeyInput, providerConfig.keyStorage, `Enter ${providerConfig.label} API key`);
    updateApiKeyPanel(provider, selectedModel);
  });

  clearApiKeysBtn?.addEventListener("click", () => {
    const provider = modelProviderSelect?.value || detectedProvider;
    const providerConfig = getProviderConfig(provider);

    if (!providerConfig.keyStorage) {
      showInlineToast(apiKeysToast, "Local models do not require an API key.", "ok");
      return;
    }

    localStorage.removeItem(providerConfig.keyStorage);
    refreshMaskedKeyInput(providerKeyInput, providerConfig.keyStorage, `Enter ${providerConfig.label} API key`);
    updateApiKeyPanel(provider, defaultModelSelect?.value || settings.defaultModel);
    showInlineToast(apiKeysToast, "API key cleared from this browser.", "ok");
  });

  root.querySelectorAll(".settings-nav-link").forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.getAttribute("data-section-link");
      if (!target) return;
      root.querySelectorAll(".settings-nav-link").forEach((navButton) => {
        navButton.classList.toggle("active", navButton === button);
      });
      root.querySelectorAll(".settings-panel").forEach((panel) => {
        const matches = panel.getAttribute("data-section") === target;
        panel.classList.toggle("active", matches);
      });
    });
  });

  root.querySelectorAll(".settings-toggle-group").forEach((group) => {
    const hiddenInput = group.querySelector("input[type=\"hidden\"]");
    const buttons = group.querySelectorAll(".settings-segment");
    const descriptions = group.querySelectorAll(".settings-toggle-description");

    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const value = button.getAttribute("data-toggle-value") || "";
        if (hiddenInput) hiddenInput.value = value;
        buttons.forEach((btn) => btn.classList.toggle("active", btn === button));
        buttons.forEach((btn) => btn.setAttribute("aria-pressed", btn === button ? "true" : "false"));
        descriptions.forEach((desc) => {
          desc.classList.toggle("active", desc.getAttribute("data-toggle-description") === value);
        });
      });
    });
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
      localStorage.removeItem(DEFAULT_AI_MODEL_STORAGE_KEY);
      localStorage.removeItem(GEMINI_API_KEY_STORAGE_KEY);
      localStorage.removeItem(OPENROUTER_API_KEY_STORAGE_KEY);
      localStorage.removeItem(OPENAI_API_KEY_STORAGE_KEY);
      localStorage.removeItem(AI_TEMPERATURE_STORAGE_KEY);
      localStorage.removeItem(GLOBAL_SYSTEM_INSTRUCTION_STORAGE_KEY);
      localStorage.removeItem(ANALYSIS_DEPTH_STORAGE_KEY);
      localStorage.removeItem(AUTO_DELETE_SOURCE_TEXT_STORAGE_KEY);
      window.location.href = "/docs/index.html";
    } catch (error) {
      showInlineToast(saveToast, error.message || "Failed to delete account.", "err");
    }
  });
}

