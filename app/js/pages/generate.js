import { apiGet, getAIConfig, streamGenerate } from "../api.js?v=20260511-generate-output-v1";
import { mountLoader } from "../components/loader.js";

function formatProfileDate(ds) {
  if (!ds) return "Unknown date";
  return new Date(ds).toLocaleString(undefined, {
    dateStyle: "medium", timeStyle: "short"
  });
}

function getProfileDisplayName(profile) {
  const label = profile?.analysis_name || profile?.name || "";
  if (label && String(label).trim()) return String(label).trim();
  return `Profile ${profile?.id || ""}`.trim();
}

function isDeepSeekModel(model) {
  return String(model || "").trim().toLowerCase().startsWith("deepseek/");
}

export async function mountGeneratePage(root) {
  root.innerHTML = `
    <section class="container page-enter page-form-balance">
      <header class="page-head">
        <h1 class="page-title">Generate and Transfer</h1>
      </header>

      <section class="gen-layout">
        <form id="generate-form" class="gen-form-minimal">
          <div class="gen-grid-row">
            <div class="gen-field">
              <label for="gen-profile">Style Profile</label>
              <select id="gen-profile" name="profileId">
                <option value="">-- No Profile (Base Model) --</option>
              </select>
            </div>
            <div class="gen-field">
              <label for="gen-type">Content Type</label>
              <select id="gen-type" name="contentType">
                <option value="article">Article</option>
                <option value="email">Email</option>
                <option value="story">Story</option>
                <option value="essay">Essay</option>
                <option value="letter">Letter</option>
                <option value="review">Review</option>
                <option value="blog">Blog</option>
                <option value="social">Social</option>
                <option value="academic">Academic</option>
                <option value="creative">Creative</option>
              </select>
            </div>
          </div>

          <div class="gen-grid-row">
            <div class="gen-field">
              <label for="gen-tone">Desired Tone</label>
              <select id="gen-tone" name="tone">
                <option value="neutral">Neutral</option>
                <option value="formal">Formal</option>
                <option value="casual">Casual</option>
                <option value="professional">Professional</option>
                <option value="creative">Creative</option>
                <option value="persuasive">Persuasive</option>
              </select>
            </div>
            <div class="gen-field">
              <label for="gen-length">Word Count</label>
              <input id="gen-length" name="length" type="number" value="300" />
            </div>
          </div>

          <div class="gen-field">
            <label for="gen-model">Model (from Settings)</label>
            <select id="gen-model" name="model" disabled>
              <option value="gemma3:1b">Local: gemma3:1b</option>
              <option value="gemini-1.5-flash">Gemini: gemini-1.5-flash</option>
              <option value="gemini-2.0-flash">Gemini: gemini-2.0-flash</option>
              <option value="anthropic/claude-3.5-sonnet">OpenRouter: anthropic/claude-3.5-sonnet</option>
              <option value="meta-llama/llama-3.3-70b-instruct:free">OpenRouter: meta-llama/llama-3.3-70b-instruct:free</option>
              <option value="deepseek/deepseek-r1:free">OpenRouter: deepseek/deepseek-r1:free</option>
              <option value="gpt-4o-mini">OpenAI: gpt-4o-mini</option>
              <option value="gpt-4o">OpenAI: gpt-4o</option>
              <option value="gpt-5.1">OpenAI: gpt-5.1</option>
            </select>
          </div>

          <p id="generate-provider-help" class="muted">Model is controlled by Settings.</p>

          <div class="gen-field gen-field-full">
            <label for="gen-topic">Topic / Subject</label>
            <textarea id="gen-topic" name="topic" rows="3" required placeholder="e.g. about my pet dog remmy"></textarea>
          </div>

          <div class="gen-field gen-field-full">
            <label for="gen-context">Additional Context (Optional)</label>
            <textarea id="gen-context" name="context" rows="3" placeholder="Any specific instructions..."></textarea>
          </div>

          <button type="submit" class="btn btn-dark gen-submit">Generate Content</button>
        </form>

        <div class="gen-output-wrap">
          <label class="gen-output-label">Generated Output</label>
          <div id="gen-stream" class="gen-output-box gen-output-minimal"><span class="muted">Your generated content will appear here...</span></div>
          <div id="gen-status" class="gen-status muted" aria-live="polite"></div>
        </div>
      </section>
    </section>
  `;

  const genForm = root.querySelector("#generate-form");
  const streamBox = root.querySelector("#gen-stream");
  const statusBox = root.querySelector("#gen-status");
  const profileSelect = root.querySelector("#gen-profile");
  const modelSelect = root.querySelector("#gen-model");
  const providerHelp = root.querySelector("#generate-provider-help");

  if (streamBox) {
    streamBox.style.color = "#111318";
    streamBox.style.webkitTextFillColor = "#111318";
    streamBox.style.opacity = "1";
  }

  try {
    const res = await apiGet("/api/profiles");
    const profiles = Array.isArray(res) ? res : (res.data || []);

    profiles.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${getProfileDisplayName(p)} (${formatProfileDate(p.created_at)})`;
      profileSelect.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed to load profiles:", err);
  }

  function updateProviderHelp() {
    const ai = getAIConfig();
    if (modelSelect) {
      modelSelect.value = ai.model;
    }
    if (ai.provider === "gemini") {
      providerHelp.textContent = ai.gemini_api_key
        ? "Gemini key found in local browser storage."
        : "Please add your Gemini API key in Settings.";
      return;
    }

    if (ai.provider === "openrouter") {
      providerHelp.textContent = ai.openrouter_api_key
        ? "OpenRouter key found in local browser storage."
        : (isDeepSeekModel(ai.model)
          ? "OpenRouter API key required for DeepSeek."
          : "Please add your OpenRouter API key in Settings.");
      return;
    }

    if (ai.provider === "openai") {
      providerHelp.textContent = ai.openai_api_key
        ? "OpenAI key found in local browser storage."
        : "Please add your OpenAI API key in Settings.";
      return;
    }

    providerHelp.textContent = "Using local Ollama model from Settings.";
  }

  updateProviderHelp();

  genForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    streamBox.textContent = "";
    if (statusBox) statusBox.textContent = "";

    const data = new FormData(genForm);
    const ai = getAIConfig();

    if (ai.provider === "gemini" && !ai.gemini_api_key) {
      streamBox.textContent = "Please add your Gemini API key in Settings.";
      return;
    }

    if (ai.provider === "openrouter" && !ai.openrouter_api_key) {
      streamBox.textContent = isDeepSeekModel(ai.model)
        ? "OpenRouter API key required for DeepSeek."
        : "Please add your OpenRouter API key in Settings.";
      return;
    }

    if (ai.provider === "openai" && !ai.openai_api_key) {
      streamBox.textContent = "Please add your OpenAI API key in Settings.";
      return;
    }

    const payload = {
      topic: String(data.get("topic") || "").trim(),
      profileId: String(data.get("profileId") || "").trim() || null,
      length: parseInt(data.get("length") || "300", 10),
      model: ai.model,
      provider: ai.provider,
      options: {
        provider: ai.provider,
        model: ai.model,
        apiKey: ai.provider === "gemini" ? ai.gemini_api_key : null,
        contentType: String(data.get("contentType") || "article"),
        tone: String(data.get("tone") || "neutral"),
        length: parseInt(data.get("length") || "300", 10),
        context: String(data.get("context") || "").trim(),
      }
    };

    if (ai.provider === "gemini") {
      payload.gemini_api_key = ai.gemini_api_key;
    } else if (ai.provider === "openrouter") {
      payload.openrouter_api_key = ai.openrouter_api_key;
    } else if (ai.provider === "openai") {
      payload.openai_api_key = ai.openai_api_key;
    }

    mountLoader(streamBox, "Generating content...");
    if (statusBox) statusBox.textContent = "Generating...";

    try {
      let generatedText = "";
      await streamGenerate(payload, {
        onToken: (token) => {
          generatedText += token;
          streamBox.textContent = generatedText;
        },
      });

      if (!generatedText.trim()) {
        streamBox.textContent = "No content was generated.";
      }
      if (statusBox) {
        statusBox.textContent = generatedText.trim().startsWith("Generation failed:")
          ? ""
          : "Generation complete";
      }
    } catch (error) {
      streamBox.textContent = error.message || "Generation failed";
      if (statusBox) statusBox.textContent = "";
    }
  });

}
