import { apiGet, apiPost, streamAnalyze } from "../api.js";
import { mountLoader } from "../components/loader.js";

function itemOptions(items) {
  return items.map((item) => `<option value="${item.id}">${item.analysis_name || item.name || `Profile ${item.id}`}</option>`).join("");
}

function compareBars(result) {
  const score = Number(result?.similarity_score || 0);
  const overlap = Number(result?.feature_overlap || 0);

  return `
    <div class="compare-bars card">
      <h3 class="section-title">Comparison Result</h3>
      <div class="metric-row">
        <span>Similarity Score</span>
        <div class="meter"><i style="width:${Math.max(0, Math.min(100, score))}%"></i></div>
        <strong>${score.toFixed(1)}%</strong>
      </div>
      <div class="metric-row">
        <span>Feature Overlap</span>
        <div class="meter"><i style="width:${Math.max(0, Math.min(100, overlap))}%"></i></div>
        <strong>${overlap.toFixed(1)}%</strong>
      </div>
    </div>
  `;
}

export async function mountComparePage(root) {
  root.innerHTML = `
    <section class="container page-enter">
      <header class="page-head">
        <h1 class="page-title">Style Comparison</h1>
        <p class="page-subtitle">Compare two saved profiles or analyze new text samples.</p>
      </header>

      <section class="card">
        <form id="compare-form" class="stack-form">
          <label>Profile A</label>
          <div style="display:flex; gap:15px; margin-bottom: 8px;">
            <label style="cursor:pointer;"><input type="radio" name="mode_a" value="saved" checked> Saved Profile</label>
            <label style="cursor:pointer;"><input type="radio" name="mode_a" value="text"> New Text Data</label>
          </div>
          <select id="profile-a"></select>
          <textarea id="text-a" rows="6" style="display:none;" placeholder="Paste First Text Sample Here..."></textarea>

          <label style="margin-top:20px;">Profile B</label>
          <div style="display:flex; gap:15px; margin-bottom: 8px;">
            <label style="cursor:pointer;"><input type="radio" name="mode_b" value="saved" checked> Saved Profile</label>
            <label style="cursor:pointer;"><input type="radio" name="mode_b" value="text"> New Text Data</label>
          </div>
          <select id="profile-b"></select>
          <textarea id="text-b" rows="6" style="display:none;" placeholder="Paste Second Text Sample Here..."></textarea>

          <button style="margin-top:25px;" type="submit" class="btn btn-dark" id="compare-btn">Compare Styles</button>  
        </form>
      </section>

      <section id="compare-result"></section>
    </section>
  `;

  const form = root.querySelector("#compare-form");
  const selectA = root.querySelector("#profile-a");
  const selectB = root.querySelector("#profile-b");
  const textA = root.querySelector("#text-a");
  const textB = root.querySelector("#text-b");
  const resultEl = root.querySelector("#compare-result");
  const btn = root.querySelector("#compare-btn");

  const toggleVisibility = (modeName, selectEl, textEl) => {
    const isText = root.querySelector(`input[name="${modeName}"]:checked`).value === "text";
    selectEl.style.display = isText ? "none" : "block";
    textEl.style.display = isText ? "block" : "none";
  };

  root.querySelectorAll('input[name="mode_a"]').forEach(el => el.addEventListener("change", () => toggleVisibility("mode_a", selectA, textA)));
  root.querySelectorAll('input[name="mode_b"]').forEach(el => el.addEventListener("change", () => toggleVisibility("mode_b", selectB, textB)));

  mountLoader(resultEl, "Loading profiles...");

  try {
    const profiles = await apiGet("/api/profiles");
    if (!profiles.length) {
      resultEl.innerHTML = "<div class='card'><p class='muted'>You have no saved profiles. Generate them from text inputs.</p></div>";
    } else {
      const options = itemOptions(profiles);
      selectA.innerHTML = options;
      selectB.innerHTML = options;
      if (profiles.length > 1) selectB.selectedIndex = 1;
      resultEl.innerHTML = "";
    }
  } catch (error) {
    resultEl.innerHTML = `<div class="toast err">${error.message}</div>`;
  }

  async function getProfilePayload(modeVal, selectVal, textVal, nameFallback) {
    if (modeVal === "saved") {
      if (!selectVal) throw new Error(`Please select a saved profile for ${nameFallback}`);
      return { id: String(selectVal), data: null };
    }

    if (!textVal.trim()) throw new Error(`Please enter text data for ${nameFallback}`);
    
    let finalProfile = null;
    await streamAnalyze({ text: textVal.trim(), author_name: nameFallback }, {
        onPass: () => {},
        onProgress: (evt) => {
           resultEl.innerHTML = `<div class="card"><div class="loader"></div><p class="muted">Analyzing ${nameFallback}... ${Number(evt.elapsed_seconds||0)}s elapsed.</p></div>`;
        },
        onResult: (res) => { finalProfile = res; },
        onError: (err) => { throw new Error(err); }
    });

    if(!finalProfile) throw new Error(`Failed to generate profile for ${nameFallback}`);
    
    // Return NO ID, just raw data so it doesn't save to DB!
    return { id: null, data: finalProfile };
  }

  form?.addEventListener("submit", async (event) => {
    event.preventDefault();

    const modeA = root.querySelector('input[name="mode_a"]:checked').value;
    const modeB = root.querySelector('input[name="mode_b"]:checked').value;
    
    const valA = selectA.value;
    const valB = selectB.value;
    
    const txtA = textA.value;
    const txtB = textB.value;

    btn.disabled = true;

    try {
      const targetA = await getProfilePayload(modeA, valA, txtA, "Compare Input A");
      const targetB = await getProfilePayload(modeB, valB, txtB, "Compare Input B");

      if (targetA.id && targetB.id && targetA.id === targetB.id) {
        resultEl.innerHTML = "<div class='toast err'>Compared profiles must be distinct. Make sure valid inputs are provided.</div>";
        btn.disabled = false;
        return;
      }

      mountLoader(resultEl, "Comparing profiles...");

      const payload = {
        profile_a_id: targetA.id,
        profile_b_id: targetB.id,
        profile_a_data: targetA.data,
        profile_b_data: targetB.data
      };

      const result = await apiPost("/api/comparisons", payload);
      resultEl.innerHTML = compareBars(result);

    } catch (error) {
      resultEl.innerHTML = `<div class="toast err">${error.message}</div>`;
    } finally {
      btn.disabled = false;
    }
  });
}

