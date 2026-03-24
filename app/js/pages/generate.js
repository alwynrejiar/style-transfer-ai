import { apiGet, apiPost, streamGenerate } from "../api.js?v=20260324-google-auth-v11";
import { mountLoader } from "../components/loader.js";

function formatProfileDate(ds) {
  if (!ds) return "Unknown date";
  return new Date(ds).toLocaleString(undefined, {
    dateStyle: "medium", timeStyle: "short"
  });
}

export async function mountGeneratePage(root) {
  root.innerHTML = `
    <section class="container page-enter">
      <header class="page-head">
        <h1 class="page-title">Generate and Transfer</h1>
        <p class="page-subtitle">Create original content based on a style profile or transfer existing text.</p>
      </header>

      <section class="gen-two-col" style="display: flex; gap: 40px; flex-wrap: wrap;">
        <article class="card" style="flex: 1; min-width: 300px;">
          <h3 class="section-title">Content Generation with Style Profile</h3>
          <form id="generate-form" class="stack-form">
            <label for="gen-profile">Style Profile</label>
            <select id="gen-profile" name="profileId">
              <option value="">-- No Profile (Base Model) --</option>
            </select>

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

            <label for="gen-topic">Topic / Subject</label>
            <textarea id="gen-topic" name="topic" rows="3" required placeholder="e.g. about my pet dog remmy"></textarea>

            <label for="gen-length">Target Length in words</label>
            <input id="gen-length" name="length" type="number" value="300" />

            <label for="gen-tone">Desired Tone</label>
            <select id="gen-tone" name="tone">
              <option value="neutral">Neutral</option>
              <option value="formal">Formal</option>
              <option value="casual">Casual</option>
              <option value="professional">Professional</option>
              <option value="creative">Creative</option>
              <option value="persuasive">Persuasive</option>
            </select>

            <label for="gen-context">Additional Context (Optional)</label>
            <textarea id="gen-context" name="context" rows="2" placeholder="Any specific instructions..."></textarea>

            <button type="submit" class="btn btn-dark" style="margin-top: 10px;">Generate Content</button>
          </form>

          <div style="margin-top: 30px;">
            <label style="font-weight: 600; font-size: 0.95rem; margin-bottom: 8px; display: block;">Generated Output</label>
            <div id="gen-stream" class="stream-box report-pre" style="min-height: 200px; background: #fafafa; border: 1px solid #ddd; border-radius: 6px; padding: 16px; white-space: pre-wrap; font-family: var(--font-body);"><span class="muted" style="color: #888;">Your generated content will appear here...</span></div>
          </div>
        </article>
      </section>
    </section>
  `;

  const genForm = root.querySelector("#generate-form");
  const streamBox = root.querySelector("#gen-stream");
  const profileSelect = root.querySelector("#gen-profile");

  try {
    const res = await apiGet("/api/profiles");
    const profiles = Array.isArray(res) ? res : (res.data || []);
    
    profiles.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.name ? `${p.name} (${formatProfileDate(p.created_at)})` : `Profile ${p.id}`;
      profileSelect.appendChild(opt);
    });
  } catch (err) {
    console.error("Failed to load profiles:", err);
  }

  genForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    streamBox.innerHTML = "";

    const data = new FormData(genForm);
    const payload = {
      topic: String(data.get("topic") || "").trim(),
      profileId: String(data.get("profileId") || "").trim() || null,
      length: parseInt(data.get("length") || "300", 10),
      options: {
        contentType: String(data.get("contentType") || "article"),
        tone: String(data.get("tone") || "neutral"),
        context: String(data.get("context") || "").trim()
      }
    };

    mountLoader(streamBox, "Generating content...");

    try {
      streamBox.textContent = "";
      await streamGenerate(payload, {
        onToken: (token) => {
          streamBox.textContent += token;
        },
      });
      streamBox.innerHTML += "\n\n<strong style='color:green'>âœ” Generation Complete</strong>";
    } catch (error) {
      streamBox.innerHTML = `<div class='toast err'>${error.message || 'Generation failed'}</div>`;
    }
  });

}












