import { isAuthenticated, showAuthModal, getSession } from "../auth.js?v=20260504-logo-v2";
import { signOutSession } from "../api.js?v=20260324-google-auth-v14";
import { showProfileModal } from "./profileModal.js?v=20260324-google-auth-v14";

const links = [
  { hash: "#/analyze", label: "Analyze" },
  { hash: "#/generate", label: "Generate" },
  { hash: "#/compare", label: "Compare" },
  { hash: "#/student-analogy", label: "Student Analogy" },
  { hash: "#/profiles", label: "Profiles" },
  { hash: "#/settings", label: "Settings" },
];

const THEME_KEY = "stylomex.theme";
const THEME_DARK = "dark";
const THEME_LIGHT = "light";

function getStoredTheme() {
  const value = localStorage.getItem(THEME_KEY);
  return value === THEME_LIGHT ? THEME_LIGHT : THEME_DARK;
}

function applyTheme(theme, nav) {
  const isDark = theme === THEME_DARK;
  document.body.classList.toggle("dark", isDark);
  document.body.classList.toggle("light", !isDark);
  localStorage.setItem(THEME_KEY, isDark ? THEME_DARK : THEME_LIGHT);

  if (nav) {
    const toggleBtn = nav.querySelector("#theme-toggle");
    if (toggleBtn) {
      toggleBtn.textContent = isDark ? "Light mode" : "Dark mode";
    }
  }
}

function getInitials(email) {
  if (!email) return "G";
  return email.substring(0, 2).toUpperCase();
}

function getAvatarUrl(email) {
  if (!email) return null;
  return localStorage.getItem(`stylomex.avatar.${email}`);
}

function updateActive(nav) {
  const hash = window.location.hash || "#/analyze";
  nav.querySelectorAll("a[data-hash]").forEach((anchor) => {
    anchor.classList.toggle("active", anchor.dataset.hash === hash);
  });

  const isAuth = isAuthenticated();

  const authBtn = nav.querySelector("#auth-btn");
  if (authBtn) {
    authBtn.textContent = isAuth ? "Sign out" : "Sign in";
  }

  const profileCardContainer = nav.querySelector("#sidebar-profile-card");
  if (profileCardContainer) {
    const session = getSession() || {};
    const email = session.email || "Guest User";
    const initials = isAuth ? getInitials(email) : "G";
    const status = isAuth ? "Verified Identity" : "Explore features";
    const shortName = isAuth ? email.split('@')[0] : "Guest User";
    const avatarUrl = isAuth ? getAvatarUrl(email) : null;

    // Base Layer Updates
    nav.querySelector(".profile-base-name").textContent = shortName;
    nav.querySelector(".profile-base-status").textContent = status;
    const baseAvatar = nav.querySelector(".profile-base-initials");
    
    // Overlay Layer Updates
    nav.querySelector(".profile-overlay-name").textContent = shortName;
    nav.querySelector(".profile-overlay-status").textContent = status;
    const overlayAvatar = nav.querySelector(".profile-overlay-initials");

    // Apply avatar or initials to base
    if (avatarUrl) {
        baseAvatar.textContent = "";
        baseAvatar.style.backgroundImage = `url(${avatarUrl})`;
        baseAvatar.style.backgroundSize = "cover";
        baseAvatar.style.backgroundPosition = "center";

        overlayAvatar.textContent = "";
        overlayAvatar.style.backgroundImage = `url(${avatarUrl})`;
        overlayAvatar.style.backgroundSize = "cover";
        overlayAvatar.style.backgroundPosition = "center";
    } else {
        baseAvatar.textContent = initials;
        baseAvatar.style.backgroundImage = "none";

        overlayAvatar.textContent = initials;
        overlayAvatar.style.backgroundImage = "none";
    }
  }
}

export function renderNavbar(mountNode) {
  if (!mountNode) return;

  mountNode.innerHTML = `
    <aside class="app-sidebar" aria-label="Primary Sidebar">
      <a class="sidebar-logo" href="/docs/index.html">
        <img class="sidebar-logo-mark" src="assets/logo.png" alt="Stylomex logo" width="38" height="38" />
        <span class="logo-text">Stylomex.AI</span>
      </a>

      <div class="sidebar-nav" role="navigation" aria-label="Feature Navigation">
        ${links.map((l) => `<a class="sidebar-link" data-hash="${l.hash}" href="${l.hash}">${l.label}</a>`).join("")}
      </div>

      <div class="sidebar-bottom-actions" style="margin-top: auto; display: flex; flex-direction: column; gap: 12px; width: 100%;">
        <button id="theme-toggle" class="settings-btn theme-toggle-btn" type="button">Light mode</button>
        <a id="nav-contact" href="#/contact" class="sidebar-link" data-hash="#/contact" style="display: flex; align-items: center; justify-content: center; gap: 8px; width: 100%; border: 1px solid var(--border-color, rgba(128,128,128,0.2)); border-radius: 12px; font-weight: 600;">
          <svg viewBox="0 0 24 24" fill="none" class="nav-icon" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 18px; height: 18px;">
            <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
          </svg>
          <span>Contact</span>
        </a>
        <button id="auth-btn" class="settings-btn settings-btn-primary" type="button" style="width: 100%; border-radius: 12px; padding: 10px; font-weight: 600; cursor: pointer; transition: all 0.2s;">Sign in</button>
        
        <div id="sidebar-profile-card" class="animated-profile-card" role="button" tabindex="0" style="margin-top: 0;" title="Click to view identity card">
          <div class="profile-card-base">
            <div class="profile-header">
              <div class="profile-avatar profile-base-initials" style="transition: none;">G</div>
              <div class="profile-info">
                <span class="profile-name profile-base-name">Guest User</span>
                <span class="profile-role profile-base-status">Explore features</span>
              </div>
            </div>
          </div>
          <div class="profile-card-overlay">
            <div class="profile-header">
              <div class="profile-avatar profile-overlay-initials">G</div>
              <div class="profile-info">
                <span class="profile-name profile-overlay-name">Guest User</span>
                <span class="profile-role profile-overlay-status">Explore features</span>
              </div>
            </div>
            <div class="profile-action">
              <span class="profile-overlay-action">View Identity</span>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="profile-action-icon">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline>
              </svg>
            </div>
          </div>
        </div>
      </div>
    </aside>
  `;

  const nav = mountNode.querySelector(".app-sidebar");
  const authBtn = mountNode.querySelector("#auth-btn");
  const profileCard = mountNode.querySelector("#sidebar-profile-card");
  const themeToggle = mountNode.querySelector("#theme-toggle");

  applyTheme(getStoredTheme(), nav);

  themeToggle?.addEventListener("click", () => {
    const nextTheme = document.body.classList.contains("dark") ? THEME_LIGHT : THEME_DARK;
    applyTheme(nextTheme, nav);
  });

  authBtn?.addEventListener("click", async () => {
    if (!isAuthenticated()) {
      showAuthModal();
      return;
    }
    await signOutSession();
  });

  profileCard?.addEventListener("click", () => {
      if (isAuthenticated()) {
          showProfileModal();
      } else {
          showAuthModal();
      }
  });

  window.addEventListener("hashchange", () => updateActive(nav));
  window.addEventListener("auth:change", () => updateActive(nav));
  updateActive(nav);
}








