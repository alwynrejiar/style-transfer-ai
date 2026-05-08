import { apiPost, streamAnalyze } from "../api.js?v=20260324-google-auth-v14";

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatNumber(value, digits = 1) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "-";
}

function normalizeResult(input) {
  if (input && typeof input === "object") return input;
  if (typeof input !== "string") return input;
  const text = input.trim();
  if (!text.startsWith("{") && !text.startsWith("[")) return input;
  try {
    return JSON.parse(text);
  } catch {
    return input;
  }
}

function formatTraitItems(items) {
  if (!Array.isArray(items)) return [];
  return items
    .map((item) => {
      if (!item) return "";
      if (typeof item === "string") return item;
      const trait = item.trait || item.name || "Trait";
      const strength = Number(item.strength);
      if (Number.isFinite(strength)) {
        return `${trait} (${(strength * 100).toFixed(0)}%)`;
      }
      return String(trait);
    })
    .filter(Boolean);
}

function listSection(title, items) {
  const safeItems = Array.isArray(items) ? items.filter(Boolean) : [];
  const normalizedItems = formatTraitItems(safeItems);
  if (!normalizedItems.length) return "";
  return `
    <div class="analysis-section">
      <h4>${escapeHtml(title)}</h4>
      <p>${normalizedItems.map((item) => escapeHtml(item)).join(" | ")}</p>
    </div>
  `;
}

function readableAnalysisMarkup(result) {
  const normalized = normalizeResult(result);

  if (!normalized || typeof normalized !== "object") {
    return `<p>${escapeHtml(String(result || "No result"))}</p>`;
  }

  const synthesis = normalized.synthesis || {};
  const readability = normalized.readability_metrics || {};
  const confidence = normalized.confidence_report?.overall_profile_confidence ?? synthesis.profile_confidence;
  const confidencePct = Number.isFinite(Number(confidence)) ? `${(Number(confidence) * 100).toFixed(1)}%` : "-";
  const summary =
    normalized.style_fingerprint_summary ||
    synthesis.style_fingerprint_summary ||
    normalized.human_readable_report ||
    "No human-readable summary available.";

  const distinctiveTraits = normalized.most_distinctive_traits || synthesis.most_distinctive_traits;
  const keyTraits = formatTraitItems(normalized.key_traits || synthesis.key_traits);
  const keepList = normalized.do_not_lose || synthesis.do_not_lose;
  const avoidList = normalized.avoid_in_rewrite || synthesis.avoid_in_rewrite;
  const rewriteDirective = normalized.rewrite_directive || synthesis.rewrite_directive;

  return `
    <div class="analysis-readable">
      <div class="analysis-section">
        <h4>Style Fingerprint Summary</h4>
        <p>${escapeHtml(summary)}</p>
      </div>

      <div class="analysis-section">
        <h4>Core Metrics</h4>
        <div class="analysis-metric-grid">
          <div><span>Profile Confidence</span><strong>${confidencePct}</strong></div>
          <div><span>Flesch Reading Ease</span><strong>${formatNumber(readability.flesch_reading_ease)}</strong></div>
          <div><span>Flesch-Kincaid Grade</span><strong>${formatNumber(readability.flesch_kincaid_grade)}</strong></div>
          <div><span>Complex Word Ratio</span><strong>${formatNumber(readability.complex_word_ratio, 3)}</strong></div>
        </div>
      </div>

      ${listSection("Most Distinctive Traits", distinctiveTraits)}
      ${listSection("Key Traits", keyTraits)}
      ${listSection("Preserve in Rewrite", keepList)}
      ${listSection("Avoid in Rewrite", avoidList)}

      ${rewriteDirective ? `
        <div class="analysis-section">
          <h4>Rewrite Directive</h4>
          <p>${escapeHtml(rewriteDirective)}</p>
        </div>
      ` : ""}

      <details class="analysis-raw-json">
        <summary>Show Raw JSON</summary>
        <pre>${escapeHtml(JSON.stringify(normalized, null, 2))}</pre>
      </details>
    </div>
  `;
}

export async function mountAnalyzePage(root) {
  root.innerHTML = `
    <section class="container page-enter">
      <header class="page-head">
        <h1 class="page-title">Style Analysis</h1>
        <p class="page-subtitle">Run a multi-pass stylometric analysis on your writing sample.</p>
      </header>

      <div class="card">
        <form id="analyze-form" class="stack-form">
          <label for="author-name">Profile Name</label>
          <input id="author-name" name="name" type="text" placeholder="e.g. Austen sample" />
          
          <label for="analyze-text">Text to Analyze</label>
          <textarea id="analyze-text" name="text" rows="10" placeholder="Paste at least a few paragraphs for stronger results" required></textarea>
          
          <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;">
            <button type="submit" class="btn btn-dark btn-glow">Analyze</button>
            <button type="button" class="btn" id="analyze-save">Save Profile</button>
          </div>
          
          <div>
            <label>Generated Profile Output</label>
            <div id="analyze-stream" class="stream-box analyze-output-box" style="min-height: 200px; max-height: 460px; padding: 15px; white-space: pre-wrap; overflow-y: auto; overflow-x: hidden;">Awaiting analysis...</div>
          </div>
        </form>
      </div>

      <p id="analyze-status" class="muted" style="margin-top: 15px;">Run analysis, then click Save Profile.</p>
      <section id="analyze-feedback"></section>
    </section>
  `;

  const form = root.querySelector("#analyze-form");
  const saveBtn = root.querySelector("#analyze-save");
  const statusEl = root.querySelector("#analyze-status");
  const feedbackEl = root.querySelector("#analyze-feedback");
  const streamBox = root.querySelector("#analyze-stream");

  if (streamBox) {
    streamBox.style.color = "#111318";
    streamBox.style.webkitTextFillColor = "#111318";
    streamBox.style.opacity = "1";
  }

  let lastResult = null;

  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    lastResult = null;
    feedbackEl.innerHTML = "";
    streamBox.innerHTML = "";
    statusEl.textContent = "Starting analysis...";

    const data = new FormData(form);
    const text = String(data.get("text") || "").trim();
    const name = String(data.get("name") || "").trim();

    try {
      await streamAnalyze(
        { text, author_name: name || "Anonymous_User" },
        {
          onPass: () => {},
          onProgress: (evt) => {
            const elapsed = Number(evt?.elapsed_seconds || 0);
            statusEl.textContent = `Analysis in progress (${elapsed}s elapsed).`;
            if (evt.message) {
              const line = document.createElement("div");
              line.textContent = evt.message;
              streamBox.appendChild(line);
              streamBox.scrollTop = streamBox.scrollHeight;
            }
          },
          onResult: (result) => {
            lastResult = normalizeResult(result);
            statusEl.textContent = "Analysis finished.";
            streamBox.innerHTML = readableAnalysisMarkup(lastResult);
            streamBox.scrollTop = 0;
            feedbackEl.innerHTML = "";
          },
          onError: (message) => {
            statusEl.textContent = "Analysis failed.";
            feedbackEl.innerHTML = `<div class="toast err">${message}</div>`;
          },
        }
      );

      if (!lastResult) {
        statusEl.textContent = "Analysis did not return a result.";
        feedbackEl.innerHTML = "<div class='toast err'>No analysis result was returned.</div>";
      }
    } catch (error) {
      statusEl.textContent = "Analysis failed.";
      feedbackEl.innerHTML = `<div class="toast err">${error.message}</div>`;
    }
  });

  saveBtn?.addEventListener("click", async () => {
    if (!lastResult) {
      feedbackEl.insertAdjacentHTML("beforeend", "<div class='toast err'>Run an analysis before saving.</div>");
      return;
    }

    const name = String(root.querySelector("#author-name")?.value || "").trim();
    if (!name) {
      feedbackEl.insertAdjacentHTML("beforeend", "<div class='toast err'>Enter a Profile Name before saving.</div>");
      return;
    }

    try {
      const saved = await apiPost("/api/profiles", {
        name,
        profileData: lastResult,
      });
      const savedName = saved?.analysis_name || name;
      feedbackEl.insertAdjacentHTML("beforeend", `<div class='toast ok'>Saved profile: ${savedName}</div>`);
    } catch (error) {
      feedbackEl.insertAdjacentHTML("beforeend", `<div class='toast err'>${error.message}</div>`);
    }
  });
}














