import { getSession } from "../auth.js?v=20260504-logo-v2";
import { apiGet, apiPatch, apiPost } from "../api.js?v=20260324-google-auth-v14";

function getAvatarUrl(email) {
  return localStorage.getItem(`stylomex.avatar.${email}`);
}

export function showProfileModal() {
  const session = getSession() || {};
  if (!session.access_token && !session.token) {
    return;
  }

  const email = session.email || "";
  const username = email.split('@')[0];
  const initials = email ? email.substring(0, 2).toUpperCase() : "G";
  const avatarUrl = getAvatarUrl(email);
  const joinedKey = email ? `stylomex.joined.${email}` : "";
  const joinedISO = joinedKey ? (localStorage.getItem(joinedKey) || new Date().toISOString()) : new Date().toISOString();
  if (joinedKey && !localStorage.getItem(joinedKey)) {
    localStorage.setItem(joinedKey, joinedISO);
  }
  const joinedDate = new Date(joinedISO).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  const roleOptions = ["Writer", "Researcher", "Student", "Creator"];
  const toneOptions = ["Formal", "Persuasive", "Analytical", "Creative"];
  const roleKey = email ? `stylomex.role.${email}` : "";
  const toneKey = email ? `stylomex.tone.${email}` : "";
  const defaultRole = roleOptions[(username.length || 0) % roleOptions.length];
  const defaultTone = toneOptions[(email.length || 0) % toneOptions.length];
  let role = roleKey ? (localStorage.getItem(roleKey) || defaultRole) : defaultRole;
  let dominantTone = toneKey ? (localStorage.getItem(toneKey) || defaultTone) : defaultTone;

  const scoreFromString = (value) =>
    (Array.from(value || "").reduce((acc, ch) => acc + ch.charCodeAt(0), 0) % 31) + 70;

  const fingerprintScore = scoreFromString(email || username);
  let savedProfilesCount = 0;
  apiGet("/api/profiles")
    .then((profiles) => {
      const items = Array.isArray(profiles) ? profiles : profiles?.data || [];
      savedProfilesCount = items.length;
      const countNode = document.querySelector("#profile-modal-saved-count");
      if (countNode) countNode.textContent = String(savedProfilesCount);
    })
    .catch(() => {});

  let modal = document.getElementById("profile-modal");
  if (modal) { modal.remove(); modal = null; } /* Force recreate to prevent weird states */
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "profile-modal";
    modal.className = "profile-modal-overlay";
    document.body.appendChild(modal);
  }

  modal.innerHTML = `
    <div class="profile-modal-card">
      <button class="profile-modal-close" aria-label="Close">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
      <div class="profile-modal-header">
        <div class="profile-modal-avatar-container">
          ${avatarUrl 
            ? `<img class="profile-modal-avatar" src="${avatarUrl}" alt="Avatar">`
            : `<div class="profile-modal-avatar text-fallback">${initials}</div>`
          }
          <label class="profile-modal-avatar-edit" title="Change Profile Picture">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <input type="file" id="profile-avatar-upload" accept="image/*" style="display: none;">
          </label>
        </div>
        <div class="profile-modal-info">
          <span class="profile-modal-email-label">Profile Overview</span>
          <h2 class="profile-modal-username">${username}</h2>
        </div>
      </div>
      <div class="profile-modal-body">
        <section class="profile-modal-section">
          <div class="profile-modal-row">
            <span class="profile-modal-key">Username</span>
            <span class="profile-modal-value">${username}</span>
          </div>
          <div class="profile-modal-row">
            <span class="profile-modal-key">Email</span>
            <span class="profile-modal-value">${email}</span>
          </div>
          <div class="profile-modal-row">
            <span class="profile-modal-key">Joined Date</span>
            <span class="profile-modal-value">${joinedDate}</span>
          </div>
        </section>

        <section class="profile-modal-section">
          <div class="profile-modal-row">
            <span class="profile-modal-key">Role</span>
            <span class="profile-modal-value" id="profile-modal-role">${role}</span>
          </div>
          <div class="profile-modal-row">
            <span class="profile-modal-key">Writing Fingerprint Score</span>
            <span class="profile-modal-value">${fingerprintScore}</span>
          </div>
          <div class="profile-modal-row">
            <span class="profile-modal-key">Dominant Tone</span>
            <span class="profile-modal-value" id="profile-modal-tone">${dominantTone}</span>
          </div>
        </section>

        <section class="profile-modal-section">
          <div class="profile-modal-row">
            <span class="profile-modal-key">Number of Saved Profiles</span>
            <span class="profile-modal-value" id="profile-modal-saved-count">${savedProfilesCount}</span>
          </div>
        </section>

        <section class="profile-modal-section">
          <div class="profile-modal-actions">
            <button type="button" id="profile-edit-btn" class="profile-modal-btn">Edit Profile</button>
            <button type="button" id="profile-save-btn" class="profile-modal-btn profile-modal-btn-primary hidden">Save</button>
          </div>
          <div id="profile-edit-fields" class="profile-modal-edit-grid hidden">
            <label class="profile-modal-input-wrap" for="profile-role-select">
              <span class="profile-modal-key">Role</span>
              <select id="profile-role-select" class="profile-modal-select">
                ${roleOptions.map((option) => `<option value="${option}" ${option === role ? "selected" : ""}>${option}</option>`).join("")}
              </select>
            </label>
            <label class="profile-modal-input-wrap" for="profile-tone-select">
              <span class="profile-modal-key">Dominant Tone</span>
              <select id="profile-tone-select" class="profile-modal-select">
                ${toneOptions.map((option) => `<option value="${option}" ${option === dominantTone ? "selected" : ""}>${option}</option>`).join("")}
              </select>
            </label>
          </div>
        </section>
      </div>
    </div>
  `;

  // Start animation loop
  requestAnimationFrame(() => {
    modal.classList.add("open");
  });

  // Events
  modal.querySelector(".profile-modal-close").addEventListener("click", () => {
    modal.classList.remove("open");
    setTimeout(() => modal.remove(), 300);
  });

  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.classList.remove("open");
      setTimeout(() => modal.remove(), 300);
    }
  });

  // File Upload
  const uploader = modal.querySelector("#profile-avatar-upload");
  uploader.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = async (event) => {
        const base64Data = event.target.result;
        let avatarSrc = base64Data;
        try {
          const saved = await apiPost("/api/user-profile/avatar", {
            image_base64: base64Data,
            filename: file.name,
            content_type: file.type,
          });
          avatarSrc = saved?.avatar_url || base64Data;
        } catch {
          // Keep the UI responsive even if the server-side local save fails.
        }

        localStorage.setItem(`stylomex.avatar.${email}`, avatarSrc);
        
        // Update local modal view
        const avatarDiv = modal.querySelector(".profile-modal-avatar");
        if (avatarDiv.tagName === "IMG") {
          avatarDiv.src = avatarSrc;
        } else {
          const img = document.createElement("img");
          img.className = "profile-modal-avatar";
          img.src = avatarSrc;
          img.alt = "Avatar";
          avatarDiv.replaceWith(img);
        }

        // Trigger refetch in navbar
        window.dispatchEvent(new Event("auth:change"));
      };
      reader.readAsDataURL(file);
    }
  });

  const editBtn = modal.querySelector("#profile-edit-btn");
  const saveBtn = modal.querySelector("#profile-save-btn");
  const editFields = modal.querySelector("#profile-edit-fields");
  const roleSelect = modal.querySelector("#profile-role-select");
  const toneSelect = modal.querySelector("#profile-tone-select");
  const roleValue = modal.querySelector("#profile-modal-role");
  const toneValue = modal.querySelector("#profile-modal-tone");

  apiGet("/api/user-profile")
    .then((profile) => {
      if (profile?.role) {
        role = profile.role;
        if (roleValue) roleValue.textContent = role;
        if (roleSelect) roleSelect.value = role;
        if (roleKey) localStorage.setItem(roleKey, role);
      }
      if (profile?.dominant_tone) {
        dominantTone = profile.dominant_tone;
        if (toneValue) toneValue.textContent = dominantTone;
        if (toneSelect) toneSelect.value = dominantTone;
        if (toneKey) localStorage.setItem(toneKey, dominantTone);
      }
      if (profile?.avatar_url) {
        localStorage.setItem(`stylomex.avatar.${email}`, profile.avatar_url);
        const avatarDiv = modal.querySelector(".profile-modal-avatar");
        if (avatarDiv?.tagName === "IMG") {
          avatarDiv.src = profile.avatar_url;
        } else if (avatarDiv) {
          const img = document.createElement("img");
          img.className = "profile-modal-avatar";
          img.src = profile.avatar_url;
          img.alt = "Avatar";
          avatarDiv.replaceWith(img);
        }
      }
    })
    .catch(() => {});

  editBtn?.addEventListener("click", () => {
    editFields?.classList.remove("hidden");
    saveBtn?.classList.remove("hidden");
    editBtn.classList.add("hidden");
  });

  saveBtn?.addEventListener("click", async () => {
    role = roleSelect?.value || role;
    dominantTone = toneSelect?.value || dominantTone;

    if (roleKey) localStorage.setItem(roleKey, role);
    if (toneKey) localStorage.setItem(toneKey, dominantTone);

    try {
      await apiPatch("/api/user-profile", {
        username,
        role,
        dominant_tone: dominantTone,
        writing_fingerprint_score: fingerprintScore,
        number_of_saved_profiles: savedProfilesCount,
      });
    } catch {
      // Browser localStorage still mirrors these preferences for the current session.
    }

    if (roleValue) roleValue.textContent = role;
    if (toneValue) toneValue.textContent = dominantTone;

    editFields?.classList.add("hidden");
    saveBtn.classList.add("hidden");
    editBtn?.classList.remove("hidden");
  });
}




