import { apiPost, streamAnalyze } from "../api.js?v=20260324-google-auth-v11";

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
            <button type="submit" class="btn btn-dark">Analyze</button>
            <button type="button" class="btn" id="analyze-save">Save Profile</button>
          </div>
          
          <div>
            <label>Generated Profile Output</label>
            <div id="analyze-stream" class="stream-box report-pre" style="min-height: 200px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; background: #fafafa; white-space: pre-wrap; font-family: monospace; overflow-y: auto;">Awaiting analysis...</div>
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
              streamBox.innerHTML += `<div>${evt.message}</div>`;
            }
          },
          onResult: (result) => {
            lastResult = result;
            statusEl.textContent = "Analysis finished.";
            if (result && typeof result === "object") {
                streamBox.textContent = JSON.stringify(result, null, 2);
            } else {
                streamBox.textContent = String(result);
            }
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













