import { isAuthenticated, showAuthModal } from "./auth.js?v=20260504-logo-v2";
import { renderNavbar, applyTheme, getStoredTheme } from "./components/navbar.js?v=20260429-theme-v2";
import { signOutSession } from "./api.js?v=20260511-gemini-rest-v1";
import { mountAnalyzePage } from "./pages/analyze.js?v=20260511-gemini-rest-v1";
import { mountStudentAnalogyPage } from "./pages/studentAnalogy.js?v=20260511-gemini-rest-v1";
import { mountProfilesPage } from "./pages/profiles.js?v=20260511-gemini-rest-v1";
import { mountGeneratePage } from "./pages/generate.js?v=20260511-generate-output-v1";
import { mountComparePage } from "./pages/compare.js?v=20260511-gemini-rest-v1";
import { mountContactPage } from "./pages/contact.js?v=14";

import { mountSettingsPage } from "./pages/settings.js?v=20260511-gemini-rest-v1";

const APP_ROOT = document.getElementById("app");
let navbarMounted = false;
let renderRunId = 0;
const TOP_TOGGLE_ROUTES = new Set([
  "#/analyze",
  "#/generate",
  "#/compare",
  "#/student-analogy",
  "#/profiles",
  "#/settings",
]);
const TOP_AUTH_ROUTES = new Set([
  "#/analyze",
  "#/generate",
  "#/compare",
  "#/student-analogy",
  "#/profiles",
  "#/settings",
]);

const routes = {
  "": mountAnalyzePage,
  "#/": mountAnalyzePage,
  "#/analyze": mountAnalyzePage,
  "#/student-analogy": mountStudentAnalogyPage,
  "#/profiles": mountProfilesPage,
  "#/generate": mountGeneratePage,
  "#/compare": mountComparePage,
  "#/settings": mountSettingsPage,
  "#/contact": mountContactPage,
};

function normalizeHash(hash) {
  if (!hash || hash === "#") return "#/analyze";
  return hash.startsWith("#/") ? hash : `#/${hash.replace(/^#/, "")}`;
}

function protectedRoute(hash) {
  return hash !== "#/settings" && hash !== "#/contact";
}

function renderPageThemeToggle(hash) {
  const existing = APP_ROOT.querySelector(".page-top-actions");
  if (existing) existing.remove();
  if (!TOP_TOGGLE_ROUTES.has(hash)) return;

  const scope = APP_ROOT.firstElementChild;
  if (!scope) return;
  scope.classList.add("page-theme-scope");

  const isDark = document.body.classList.contains("dark");
  const actions = document.createElement("div");
  actions.className = "page-top-actions";

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "page-theme-toggle";
  toggle.setAttribute("role", "switch");
  toggle.setAttribute("aria-checked", String(isDark));
  toggle.setAttribute("aria-label", "Toggle light and dark mode");
  toggle.innerHTML = `
    <span class="page-theme-track">
      <span class="page-theme-thumb"></span>
    </span>
  `;
  if (isDark) toggle.classList.add("is-dark");

  toggle.addEventListener("click", () => {
    const nextTheme = document.body.classList.contains("dark") ? "light" : "dark";
    applyTheme(nextTheme);
    const nowDark = document.body.classList.contains("dark");
    toggle.classList.toggle("is-dark", nowDark);
    toggle.setAttribute("aria-checked", String(nowDark));
  });

  actions.appendChild(toggle);

  if (TOP_AUTH_ROUTES.has(hash) && isAuthenticated()) {
    const authBtn = document.createElement("button");
    authBtn.type = "button";
    authBtn.className = "page-auth-btn";
    authBtn.setAttribute("aria-label", "Sign out");
    authBtn.innerHTML = `
      <img class="page-auth-icon page-auth-icon-light" src="assets/signout-transparent.png" alt="" aria-hidden="true" />
      <img class="page-auth-icon page-auth-icon-dark" src="assets/signout-orange-transparent.png" alt="" aria-hidden="true" />
    `;
    authBtn.addEventListener("click", async () => {
      authBtn.disabled = true;
      try {
        await signOutSession();
      } finally {
        authBtn.disabled = false;
      }
    });
    actions.appendChild(authBtn);
  }

  scope.appendChild(actions);
}

async function renderCurrentRoute() {
  const runId = ++renderRunId;
  const hash = normalizeHash(window.location.hash);
  if (window.location.hash !== hash) {
    window.location.hash = hash;
    return;
  }

  const mount = routes[hash] || mountAnalyzePage;

  if (protectedRoute(hash) && !isAuthenticated()) {
    showAuthModal();
    APP_ROOT.innerHTML = `
      <section class="card" style="padding:20px">
        <h2 class="section-title">Sign in required</h2>
        <p class="muted">You need an account to use analysis and generation tools.</p>
      </section>
    `;
    return;
  }

  await mount(APP_ROOT);
  if (runId !== renderRunId) return;
  renderPageThemeToggle(hash);
}

function mountNavbarOnce() {
  if (navbarMounted) return;
  const mountNode = document.getElementById("navbar-mount");
  renderNavbar(mountNode);
  navbarMounted = true;
}

window.addEventListener("hashchange", () => {
  renderCurrentRoute().catch((error) => {
    APP_ROOT.innerHTML = `<div class="toast err">${error.message}</div>`;
  });
});

window.addEventListener("auth:change", () => {
  renderCurrentRoute().catch((error) => {
    APP_ROOT.innerHTML = `<div class="toast err">${error.message}</div>`;
  });
});

mountNavbarOnce();
applyTheme(getStoredTheme());
renderCurrentRoute().catch((error) => {
  APP_ROOT.innerHTML = `<div class="toast err">${error.message}</div>`;
});














