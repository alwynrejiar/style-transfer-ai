import { isAuthenticated, showAuthModal } from "./auth.js?v=20260504-logo-v2";
import { renderNavbar } from "./components/navbar.js?v=20260429-theme-v2";
import { mountAnalyzePage } from "./pages/analyze.js?v=20260325-analogy-fix-v19";
import { mountStudentAnalogyPage } from "./pages/studentAnalogy.js?v=20260325-analogy-fix-v19";
import { mountProfilesPage } from "./pages/profiles.js?v=20260325-analogy-fix-v19";
import { mountGeneratePage } from "./pages/generate.js?v=20260325-analogy-fix-v19";
import { mountComparePage } from "./pages/compare.js?v=20260325-analogy-fix-v19";
import { mountContactPage } from "./pages/contact.js?v=14";

import { mountSettingsPage } from "./pages/settings.js?v=20260325-analogy-fix-v19";

const APP_ROOT = document.getElementById("app");
let navbarMounted = false;

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

async function renderCurrentRoute() {
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
renderCurrentRoute().catch((error) => {
  APP_ROOT.innerHTML = `<div class="toast err">${error.message}</div>`;
});














