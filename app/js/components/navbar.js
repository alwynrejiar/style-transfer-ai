import { isAuthenticated, showAuthModal } from "../auth.js";
import { signOutSession } from "../api.js";

const links = [
  { hash: "#/analyze", label: "Analyze" },
  { hash: "#/generate", label: "Generate" },
  { hash: "#/compare", label: "Compare" },
  { hash: "#/student-analogy", label: "Student Analogy" },
  { hash: "#/profiles", label: "Profiles" },
  { hash: "#/settings", label: "Settings" },
];

function updateActive(nav) {
  const hash = window.location.hash || "#/analyze";
  nav.querySelectorAll("a[data-hash]").forEach((anchor) => {
    anchor.classList.toggle("active", anchor.dataset.hash === hash);
  });

  const authBtn = nav.querySelector("#auth-btn");
  if (authBtn) {
    authBtn.textContent = isAuthenticated() ? "Sign out" : "Sign in";
  }
}

export function renderNavbar(mountNode) {
  if (!mountNode) return;

  mountNode.innerHTML = `
    <aside class="app-sidebar" aria-label="Primary Sidebar">
      <a class="sidebar-logo" href="/docs/index.html">
        <span class="logo-text">Stylomex.AI</span>
      </a>

      <div class="sidebar-nav" role="navigation" aria-label="Feature Navigation">
        ${links.map((l) => `<a class="sidebar-link" data-hash="${l.hash}" href="${l.hash}">${l.label}</a>`).join("")}
      </div>

      <button id="auth-btn" class="btn btn-dark sidebar-auth-btn" type="button">Sign in</button>
    </aside>
  `;

  const nav = mountNode.querySelector(".app-sidebar");
  const authBtn = mountNode.querySelector("#auth-btn");

  authBtn?.addEventListener("click", async () => {
    if (!isAuthenticated()) {
      showAuthModal();
      return;
    }
    await signOutSession();
  });

  window.addEventListener("hashchange", () => updateActive(nav));
  window.addEventListener("auth:change", () => updateActive(nav));
  updateActive(nav);
}
