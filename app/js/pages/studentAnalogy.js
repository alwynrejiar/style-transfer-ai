import { apiPost, streamGenerate } from "../api.js";

const ANALOGY_DOMAINS = [
  "general_simplification",
  "daily_life",
  "tech",
  "sports",
  "gaming",
  "nature",
  "cooking",
];

function buildInstruction(domain) {
  const formattedDomain = domain.replace('_', ' ').toUpperCase();
  return [
    `CRITICAL INSTRUCTION: Completely transform the input text into a comprehensive analogy based on the domain: ${formattedDomain}.`,
    "Instead of just simplifying the text, map the core concepts directly to elements within this analogy domain to ensure a student audience deeply understands it.",
    "Use cognitive load optimization: one idea per sentence and short paragraphs.",
    "Define complex terms immediately using the analogy.",
    "Include a clear step-by-step structure where the analogy follows the narrative of the original text.",
    "Ensure the final output is highly engaging, relatable, and maintains the accurate meaning of the original concepts while drastically reducing complexity."
  ].join(" ");
}

export async function mountStudentAnalogyPage(root) {
  root.innerHTML = `
    <section class="container page-enter">
      <header class="page-head">
        <h1 class="page-title">Student Analogy</h1>
        <p class="page-subtitle">Cognitive Load Optimization: simplify complex ideas with relatable analogies for students.</p>
      </header>

      <section class="card" style="margin-bottom: 20px;">
        <h3 class="section-title">Step 1: Generate Base Content</h3>
        <p class="section-desc" style="margin-bottom: 12px; font-size: 0.9rem; color: #666;">
          Type your topic below. We will generate detailed content about it.
        </p>
        <form id="topic-form" class="stack-form">
          <label for="topic-input">Topic / Subject</label>
          <textarea id="topic-input" name="topic" rows="3" placeholder="Paste or type your topic below." required></textarea>
          
          <button type="submit" class="btn btn-dark" style="margin-top: 10px;">Generate Content</button>
        </form>
        
        <div style="margin-top: 20px;">
          <label style="font-weight: 600; font-size: 0.95rem; margin-bottom: 8px; display: block;">Generated Content</label>
          <div id="gen-stream" class="stream-box report-pre" style="min-height: 150px; background: #fafafa; border: 1px solid #ddd; border-radius: 6px; padding: 16px; white-space: pre-wrap; font-family: var(--font-body);"><span class="muted" style="color: #888;">Generated content will appear here...</span></div>
        </div>
      </section>

      <section class="card" id="step-2-section" style="display: none; margin-bottom: 20px;">
        <h3 class="section-title">Step 2: Create Analogy</h3>
        <p class="section-desc" style="margin-bottom: 12px; font-size: 0.9rem; color: #666;">
          Select an analogy domain to simplify the generated content.
        </p>
        <form id="analogy-form" class="stack-form">
          <label for="analogy-domain">Select analogy domain:</label>
          <select id="analogy-domain" name="domain">
            ${ANALOGY_DOMAINS.map((domain) => `<option value="${domain}">${domain}</option>`).join("")}
          </select>

          <button type="submit" class="btn btn-dark" style="margin-top: 10px;">Generate Analogy Content</button>
        </form>
      </section>

      <section class="card" style="margin-top:14px">
        <h3 class="section-title">Final Analogy Output</h3>
        <pre id="analogy-output" class="report-pre">Output will appear here.</pre>
      </section>
    </section>
  `;

  const topicForm = root.querySelector("#topic-form");
  const analogyForm = root.querySelector("#analogy-form");
  const streamBox = root.querySelector("#gen-stream");
  const outputEl = root.querySelector("#analogy-output");
  const step2Section = root.querySelector("#step-2-section");

  let generatedText = "";

  topicForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    streamBox.innerHTML = "";
    generatedText = "";
    step2Section.style.display = "none";
    outputEl.textContent = "Output will appear here.";

    const data = new FormData(topicForm);
    const topic = String(data.get("topic") || "").trim();

    if (!topic) return;

    const payload = {
      topic: topic,
      profileId: null,
      length: 300,
      options: {
        contentType: "article",
        tone: "neutral",
        context: "Create a detailed explanation of the topic."
      }
    };

    try {
      streamBox.textContent = "Generating content...";
      
      let tempText = "";
      await streamGenerate(payload, {
        onToken: (token) => {
          if (tempText === "") streamBox.textContent = "";
          streamBox.textContent += token;
          tempText += token;
        },
      });
      generatedText = tempText;
      streamBox.innerHTML += "\\n\\n<strong style='color:green'>? Generation Complete</strong>";
      step2Section.style.display = "block";
    } catch (error) {
      streamBox.innerHTML = "<div class='toast err'>" + (error.message || "Generation failed") + "</div>";
    }
  });

  analogyForm?.addEventListener("submit", async (event) => {
    event.preventDefault();

    const data = new FormData(analogyForm);
    const domain = String(data.get("domain") || "general_simplification");

    if (!generatedText.trim()) {
      outputEl.textContent = "Please generate content in Step 1 first.";
      return;
    }

    outputEl.textContent = "Generating student-friendly analogy explanation...";

    try {
      const result = await apiPost("/api/transfer", {
        text: generatedText.trim(),
        instructions: buildInstruction(domain)
      });

      outputEl.textContent = result?.transferred_text || "No output returned.";
    } catch (error) {
      outputEl.textContent = error?.message || "Failed to generate analogy output.";
    }
  });
}
