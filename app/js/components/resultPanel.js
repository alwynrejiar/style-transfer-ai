function toSafeText(value) {
  return String(value ?? "").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function metricBar(label, value, max = 100) {
  const number = Number(value || 0);
  const pct = Math.max(0, Math.min(100, (number / max) * 100));
  return `
    <div class="metric-row">
      <span>${toSafeText(label)}</span>
      <div class="meter" role="progressbar" aria-valuenow="${number}" aria-valuemin="0" aria-valuemax="${max}">
        <i style="width:${pct}%"></i>
      </div>
      <strong>${number.toFixed(1)}</strong>
    </div>
  `;
}

export function passProgressMarkup(passEvents) {
  if (!passEvents?.length) return "";
  return `
    <section class="card pass-progress">
      <h3>Analysis Passes</h3>
      <ul>
        ${passEvents.map((event) => `<li><span>${event.pass || "pass"}</span> <b>${event.status || "running"}</b></li>`).join("")}
      </ul>
    </section>
  `;
}

export function analysisResultMarkup(result) {
  const confidence = Number(result?.analysis_confidence_score || 0);
  const readability = result?.readability_metrics || {};
  const report = result?.human_readable_report || "No report available.";

  return `
    <section class="card result-panel">
      <header class="result-head">
        <h3>Analysis Result</h3>
        <span class="score-badge">Confidence ${confidence}%</span>
      </header>

      <div class="metric-grid">
        ${metricBar("Flesch Reading Ease", readability.flesch_reading_ease || 0, 120)}
        ${metricBar("Grade Level", readability.flesch_kincaid_grade || 0, 18)}
        ${metricBar("Complex Words %", readability.complex_words_percentage || 0, 100)}
      </div>

      <details open>
        <summary>Human-readable report</summary>
        <pre class="report-pre">${toSafeText(report)}</pre>
      </details>

      <details>
        <summary>Raw JSON</summary>
        <pre class="report-pre">${toSafeText(JSON.stringify(result, null, 2))}</pre>
      </details>
    </section>
  `;
}









