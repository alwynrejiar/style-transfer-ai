import { isAuthenticated, showAuthModal, getSession } from "../auth.js?v=20260504-logo-v2";
import { showProfileModal } from "./profileModal.js?v=20260324-google-auth-v14";

const links = [
  { hash: "#/analyze", label: "Analyze", icon: "analyze" },
  { hash: "#/generate", label: "Generate", icon: "generate" },
  { hash: "#/compare", label: "Compare", icon: "compare" },
  { hash: "#/student-analogy", label: "Student Analogy", icon: "analogy" },
  { hash: "#/profiles", label: "Saved Profiles", icon: "profiles" },
];

const THEME_KEY = "stylomex.theme";
const THEME_DARK = "dark";
const THEME_LIGHT = "light";
let currentTheme = null;
const BRAND_LOGO_LIGHT = "assets/logo.png";
const BRAND_LOGO_DARK = "assets/logo-orange-transparent.png";
const ANALYZE_ICON_LIGHT = "assets/analyze black.png";
const ANALYZE_ICON_DARK = "assets/analyze dark.png";
const GENERATE_ICON_LIGHT = "assets/generation black.png";
const GENERATE_ICON_DARK = "assets/generation orange.png";
const COMPARE_ICON_LIGHT = "assets/comparing black.png";
const COMPARE_ICON_DARK = "assets/comparing orange.png";
const ANALOGY_ICON_LIGHT = "assets/graduated black.png";
const ANALOGY_ICON_DARK = "assets/graduated orange.png";
const PROFILES_ICON_LIGHT = "assets/download black.png";
const PROFILES_ICON_DARK = "assets/download orange.png";

export function getStoredTheme() {
  const value = localStorage.getItem(THEME_KEY);
  return value === THEME_LIGHT ? THEME_LIGHT : THEME_DARK;
}

export function applyTheme(theme) {
  const nextTheme = theme === THEME_LIGHT ? THEME_LIGHT : THEME_DARK;
  if (currentTheme === nextTheme) return;

  const root = document.documentElement;
  root.classList.add("theme-switching");

  currentTheme = nextTheme;
  const isDark = nextTheme === THEME_DARK;
  document.body.classList.toggle("dark", isDark);
  document.body.classList.toggle("light", !isDark);
  localStorage.setItem(THEME_KEY, nextTheme);
  document.body.dispatchEvent(new CustomEvent("theme:change", { detail: { theme: nextTheme } }));

  requestAnimationFrame(() => {
    root.classList.remove("theme-switching");
  });
}

function navIcon(type) {
  const base = `viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="sidebar-item-icon"`;
  if (type === "analyze") return `
    <span class="sidebar-item-icon sidebar-item-logo analyze-icon-wrap" aria-hidden="true">
      <img class="analyze-icon analyze-icon-light" src="${ANALYZE_ICON_LIGHT}" alt="" />
      <img class="analyze-icon analyze-icon-dark" src="${ANALYZE_ICON_DARK}" alt="" />
    </span>
  `;
  if (type === "generate") return `
    <span class="sidebar-item-icon sidebar-item-logo generate-icon-wrap" aria-hidden="true">
      <img class="generate-icon generate-icon-light" src="${GENERATE_ICON_LIGHT}" alt="" />
      <img class="generate-icon generate-icon-dark" src="${GENERATE_ICON_DARK}" alt="" />
    </span>
  `;
  if (type === "compare") return `
    <span class="sidebar-item-icon sidebar-item-logo compare-icon-wrap" aria-hidden="true">
      <img class="compare-icon compare-icon-light" src="${COMPARE_ICON_LIGHT}" alt="" />
      <img class="compare-icon compare-icon-dark" src="${COMPARE_ICON_DARK}" alt="" />
    </span>
  `;
  if (type === "analogy") return `
    <span class="sidebar-item-icon sidebar-item-logo analogy-icon-wrap" aria-hidden="true">
      <img class="analogy-icon analogy-icon-light" src="${ANALOGY_ICON_LIGHT}" alt="" />
      <img class="analogy-icon analogy-icon-dark" src="${ANALOGY_ICON_DARK}" alt="" />
    </span>
  `;
  if (type === "profiles") return `
    <span class="sidebar-item-icon sidebar-item-logo profiles-icon-wrap" aria-hidden="true">
      <img class="profiles-icon profiles-icon-light" src="${PROFILES_ICON_LIGHT}" alt="" />
      <img class="profiles-icon profiles-icon-dark" src="${PROFILES_ICON_DARK}" alt="" />
    </span>
  `;
  if (type === "settings") return `<svg ${base}><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 0 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.2a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 0 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.2a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3h.1a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.2a1.7 1.7 0 0 0 1 1.5h.1a1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8v.1a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.2a1.7 1.7 0 0 0-1.5 1z"></path></svg>`;
  return `<svg ${base}><circle cx="12" cy="12" r="9"></circle></svg>`;
}

function contactIcon() {
  return `<svg viewBox="0 0 24 24" fill="none" class="sidebar-item-icon" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path></svg>`;
}

function setSidebarLogo(nav) {
  const logo = nav?.querySelector(".sidebar-logo-mark");
  if (!logo) return;
  const dark = document.body.classList.contains("dark");
  logo.src = dark ? BRAND_LOGO_DARK : BRAND_LOGO_LIGHT;
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
        <img class="sidebar-logo-mark" src="${BRAND_LOGO_LIGHT}" alt="Stylomex logo" />
        <span class="logo-text sidebar-label">Stylomex.AI</span>
      </a>

      <div class="sidebar-nav" role="navigation" aria-label="Feature Navigation">
        ${links.map((l) => `<a class="sidebar-link sidebar-item" data-hash="${l.hash}" href="${l.hash}">${navIcon(l.icon)}<span class="sidebar-label">${l.label}</span></a>`).join("")}
      </div>

      <div class="sidebar-bottom-actions">
        <a href="#/settings" class="sidebar-link sidebar-item sidebar-bottom-link" data-hash="#/settings">
          ${navIcon("settings")}
          <span class="sidebar-label">Settings</span>
        </a>
        <a id="nav-contact" href="#/contact" class="sidebar-link sidebar-item sidebar-bottom-link" data-hash="#/contact">
          ${contactIcon()}
          <span class="sidebar-label">Contact</span>
        </a>
        <div id="sidebar-profile-card" class="animated-profile-card" role="button" tabindex="0" title="Click to view identity card">
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
  const profileCard = mountNode.querySelector("#sidebar-profile-card");

  applyTheme(getStoredTheme());
  setSidebarLogo(nav);
  document.body.addEventListener("theme:change", () => setSidebarLogo(nav));

  const setExpanded = (expand) => {
    document.body.classList.toggle("sidebar-expanded", expand);
  };
  nav.addEventListener("mouseenter", () => setExpanded(true));
  nav.addEventListener("mouseleave", () => setExpanded(false));
  nav.addEventListener("focusin", () => setExpanded(true));
  nav.addEventListener("focusout", (event) => {
    if (!nav.contains(event.relatedTarget)) setExpanded(false);
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
