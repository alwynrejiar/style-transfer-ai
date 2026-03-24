function confidenceClass(score) {
  if (score >= 80) return "high";
  if (score >= 60) return "mid";
  return "low";
}

function traitPills(profile) {
  if (!profile) return "";
  const traits = [];
  if (profile?.writing_style_analysis?.tone_assessment) {
    traits.push(`Tone: ${profile.writing_style_analysis.tone_assessment}`);
  }
  if (profile?.cognitive_fingerprint?.conceptual_depth) {
    traits.push(`Depth: ${profile.cognitive_fingerprint.conceptual_depth}`);
  }
  if (profile?.readability_metrics?.flesch_reading_ease != null) {
    traits.push(`Flesch: ${Number(profile.readability_metrics.flesch_reading_ease).toFixed(1)}`);
  }

  return traits.slice(0, 4).map((t) => `<span class="pill">${t}</span>`).join("");
}

export function profileCardMarkup(item) {
  const confidence = Number(item?.confidence_score || 75);
  const ringClass = confidenceClass(confidence);
  const createdAt = item?.created_at ? new Date(item.created_at).toLocaleString() : "Unknown";
  const title = item?.analysis_name || item?.name || "Untitled Profile";
  const mode = item?.processing_mode || "enhanced";
  const model = item?.model_used || "gemma3:1b";

  return `
    <article class="profile-card card" data-id="${item.id}">
      <header class="profile-head">
        <h3>${title}</h3>
        <div class="confidence-ring ${ringClass}"><span>${confidence}%</span></div>
      </header>

      <p class="muted">Created ${createdAt}</p>
      <div class="pills">
        <span class="pill">Mode: ${mode}</span>
        <span class="pill">Model: ${model}</span>
      </div>

      <footer class="profile-actions">
        <button class="btn" data-action="open" data-id="${item.id}">Open</button>
        <button class="btn" data-action="delete" data-id="${item.id}">Delete</button>
      </footer>
    </article>
  `;
}









