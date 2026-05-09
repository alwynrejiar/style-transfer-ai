import { apiDelete, apiGet } from "../api.js?v=20260324-google-auth-v14";
import { mountLoader } from "../components/loader.js";
import { profileCardMarkup } from "../components/profileCard.js";

function savedProfileDetailMarkup(data) {
  const confidence = Number(data?.confidence_report?.overall_profile_confidence || 0);
  const readability = data?.readability_metrics || {};
  const traits = data?.most_distinctive_traits || [];
  const humanSummary = data?.style_fingerprint_summary || "No human-readable summary is available for this profile yet.";

  return `
    <section class="card result-panel">
      <header class="result-head">
        <h3>Saved Profile Summary</h3>
        <span class="score-badge">Confidence ${(confidence * 100).toFixed(1)}%</span>
      </header>

      <div>
        <h4 class="section-title" style="margin-top:8px">Human-readable Profile</h4>
        <p class="muted" style="line-height:1.7">${humanSummary}</p>
      </div>

      <div class="metric-grid">
        <div class="metric-row"><span>Flesch Reading Ease</span><strong>${Number(readability.flesch_reading_ease || 0).toFixed(1)}</strong></div>
        <div class="metric-row"><span>Grade Level</span><strong>${Number(readability.flesch_kincaid_grade || 0).toFixed(1)}</strong></div>
        <div class="metric-row"><span>Complex Words %</span><strong>${Number(readability.complex_word_ratio || 0).toFixed(3)}</strong></div>
      </div>

      <div>
        <h4 class="section-title" style="margin-top:8px">Distinctive Traits</h4>
        <ul>
          ${traits.length ? traits.map((t) => `<li>${t}</li>`).join("") : "<li>No traits available.</li>"}
        </ul>
      </div>
    </section>
  `;
}

export async function mountProfilesPage(root) {
  root.innerHTML = `
    <section class="container page-enter">
      <header class="page-head">
        <h1 class="page-title">Saved Profiles</h1>
      </header>

      <section id="profiles-grid" class="profile-grid"></section>
      <section id="profile-detail"></section>
    </section>
  `;

  const grid = root.querySelector("#profiles-grid");
  const detail = root.querySelector("#profile-detail");

  detail.innerHTML = "<section class='card'><p class='muted'>Select a profile and click Open to view the human-readable profile summary.</p></section>";

  mountLoader(grid, "Loading profiles...");

  async function refresh() {
    try {
      const items = await apiGet("/api/profiles");
      if (!items.length) {
        grid.innerHTML = "<div class='card'><p class='muted'>No profiles yet. Analyze text to create one.</p></div>";
        detail.innerHTML = "";
        return;
      }

      grid.innerHTML = items.map(profileCardMarkup).join("");

      grid.querySelectorAll("button[data-action='open']").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const id = btn.dataset.id;
          const data = await apiGet(`/api/profiles/${id}`);
          detail.innerHTML = savedProfileDetailMarkup(data);
          detail.scrollIntoView({ behavior: "smooth", block: "start" });
        });
      });

      grid.querySelectorAll("button[data-action='delete']").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const id = btn.dataset.id;
          const confirmed = window.confirm("Delete this saved profile? This cannot be undone.");
          if (!confirmed) return;

          try {
            await apiDelete(`/api/profiles/${id}`);
            detail.innerHTML = "<div class='toast ok'>Profile deleted successfully.</div>";
            await refresh();
          } catch (error) {
            detail.innerHTML = `<div class='toast err'>${error.message}</div>`;
          }
        });
      });
    } catch (error) {
      grid.innerHTML = `<div class='toast err'>${error.message}</div>`;
    }
  }

  await refresh();
}











