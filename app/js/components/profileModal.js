import { getSession } from "../auth.js?v=20260324-google-auth-v14";

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
          <span class="profile-modal-email-label">${email}</span>
          <h2 class="profile-modal-username">${username}</h2>
        </div>
      </div>
      <div class="profile-modal-body">
        <p class="profile-modal-bio">
          Stylomex.AI User<br/>
          Building stylistic fingerprints and generating content through advanced AI profiling.
        </p>
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
      reader.onload = (event) => {
        const base64Data = event.target.result;
        localStorage.setItem(`stylomex.avatar.${email}`, base64Data);
        
        // Update local modal view
        const avatarDiv = modal.querySelector(".profile-modal-avatar");
        if (avatarDiv.tagName === "IMG") {
          avatarDiv.src = base64Data;
        } else {
          const img = document.createElement("img");
          img.className = "profile-modal-avatar";
          img.src = base64Data;
          img.alt = "Avatar";
          avatarDiv.replaceWith(img);
        }

        // Trigger refetch in navbar
        window.dispatchEvent(new Event("auth:change"));
      };
      reader.readAsDataURL(file);
    }
  });
}








