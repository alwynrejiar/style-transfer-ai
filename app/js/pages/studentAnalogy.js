import { apiPost } from "../api.js?v=20260324-google-auth-v11";
import { mountLoader } from "../components/loader.js";

const ANALOGY_DOMAINS = [
  { id: "sports", label: "Sports" },
  { id: "gaming", label: "Gaming" },
  { id: "cooking", label: "Cooking" },
  { id: "nature", label: "Nature" },
  { id: "daily_life", label: "Daily Life" },
  { id: "tech", label: "Tech" },
  { id: "general_simplification", label: "General Simplification" }
];

function stripConceptMapSection(text) {
  if (!text) return "";

  // Remove trailing "Concept Map" sections (heading + table/list content).
  return text
    .replace(/\n{1,}(?:\*\*)?concept\s*map(?:\*\*)?\s*:?\s*\n[\s\S]*$/i, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export async function mountStudentAnalogyPage(root) {
  root.innerHTML = `
    <section class="container page-enter">
      <header class="page-head">
        <h1 class="page-title">Student Analogy (v2)</h1>
        <p class="page-subtitle">Cognitive Load Optimization: enter content, choose an analogy type, then get the final analogy output.</p>
      </header>

      <section class="card" style="margin-bottom: 20px;">
        <h3 class="section-title">Step 1: Enter Your Content</h3>
        <p class="section-desc" style="margin-bottom: 12px; font-size: 0.9rem; color: #666;">
          Tell us what content you want the analogy for.
        </p>
        <form id="topic-form" class="stack-form">
          <label for="topic-input">Content / Topic</label>
          <textarea id="topic-input" name="topic" rows="3" placeholder="e.g., 'solar eclipse' or 'cellular respiration'" required></textarea>
          
          <button type="submit" class="btn btn-dark" style="margin-top: 10px;">Continue to Analogy Type</button>
        </form>
        
        <div style="margin-top: 20px;">
          <label style="font-weight: 600; font-size: 0.95rem; margin-bottom: 8px; display: block;">Prepared Content</label>
          <div id="gen-stream" class="stream-box report-pre" style="min-height: 150px; background: #fafafa; border: 1px solid #ddd; border-radius: 6px; padding: 16px; white-space: pre-wrap; font-family: var(--font-body);"><span class="muted" style="color: #888;">Expanded content will appear here...</span></div>
        </div>
      </section>

      <section class="card" id="step-2-section" style="display: none; margin-bottom: 20px;">
        <h3 class="section-title">Step 2: Choose Analogy Type</h3>
        <p class="section-desc" style="margin-bottom: 12px; font-size: 0.9rem; color: #666;">
          Select what type of analogy should be injected into your content.
        </p>
        <form id="analogy-form" class="stack-form">
          <label for="analogy-domain">Select analogy domain:</label>
          <select id="analogy-domain" name="domain">
            ${ANALOGY_DOMAINS.map((domain) => `<option value="${domain.id}">${domain.label}</option>`).join("")}
          </select>

          <button type="submit" class="btn btn-dark" style="margin-top: 10px;">Generate Final Output</button>
        </form>
      </section>

      <section class="card" id="final-output-card" style="margin-top:14px; display: none;">
        <h3 class="section-title">Step 3: Final Analogy Output</h3>
        <div id="analogy-output" class="report-pre" style="white-space: pre-wrap; font-family: var(--font-body); min-height: 100px;">Output will appear here.</div>
      </section>
    </section>
  `;

  const topicForm = root.querySelector("#topic-form");
  const analogyForm = root.querySelector("#analogy-form");
  const streamBox = root.querySelector("#gen-stream");
  const outputEl = root.querySelector("#analogy-output");
  const step2Section = root.querySelector("#step-2-section");
  const finalOutputCard = root.querySelector("#final-output-card");

  let expandedText = "";

  topicForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    expandedText = "";
    step2Section.style.display = "none";
    finalOutputCard.style.display = "none";

    const data = new FormData(topicForm);
    const topic = String(data.get("topic") || "").trim();

    if (!topic) return;

    mountLoader(streamBox, "Preparing content (this may take a moment)...");

    try {
      const data = await apiPost("/api/analogy", {
        text: topic,
        domain: "general_simplification"
      });

      if (!data) {
        throw new Error("Failed to prepare content.");
      }

      expandedText = String(data.expanded_text || data.analogy_output || "").trim();
      if (!expandedText) {
        throw new Error("No content returned from backend.");
      }
      
      streamBox.innerHTML = `<div>${expandedText}</div>\n\n<strong style="color:green">OK - Content Ready</strong>`;
      
      step2Section.style.display = "block";
    } catch (error) {
      streamBox.innerHTML = `<div class="toast err">${error.message || "Error occurred"}</div>`;
    }
  });

  analogyForm?.addEventListener("submit", async (event) => {
    event.preventDefault();

    const data = new FormData(analogyForm);
    const domain = String(data.get("domain") || "general_simplification");

    if (!expandedText.trim()) {
       finalOutputCard.style.display = "block";
       outputEl.innerHTML = "<div class=\"toast err\">Please expand content in Step 1 first.</div>";
       return;
    }

    finalOutputCard.style.display = "block";
    mountLoader(outputEl, "Generating analogy transformation...");

    try {
      const data = await apiPost("/api/analogy", {
        text: expandedText.trim(),
        domain: domain
      });

      if (!data) {
        throw new Error("Failed to generate analogy.");
      }

      const output = stripConceptMapSection(String(data.analogy_output || "").trim());
      outputEl.textContent = output || "No output returned.";
    } catch (error) {
      outputEl.innerHTML = `<div class="toast err">${error.message || "Error occurred"}</div>`;
    }
  });
}










