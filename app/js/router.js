import { isAuthenticated, showAuthModal } from "./auth.js";
import { renderNavbar } from "./components/navbar.js";
import { mountAnalyzePage } from "./pages/analyze.js";
import { mountStudentAnalogyPage } from "./pages/studentAnalogy.js";
import { mountProfilesPage } from "./pages/profiles.js";
import { mountGeneratePage } from "./pages/generate.js";
import { mountComparePage } from "./pages/compare.js";
import { mountSettingsPage } from "./pages/settings.js";

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
};

function normalizeHash(hash) {
  if (!hash || hash === "#") return "#/analyze";
  return hash.startsWith("#/") ? hash : `#/${hash.replace(/^#/, "")}`;
}

function protectedRoute(hash) {
  return hash !== "#/settings";
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
