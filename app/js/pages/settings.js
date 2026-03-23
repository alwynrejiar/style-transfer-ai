export async function mountSettingsPage(root) {
  root.innerHTML = `
    <section class="container page-enter">
      <header class="page-head">
        <h1 class="page-title">Settings</h1>
        <p class="page-subtitle">Configure your application preferences.</p>
      </header>

      <section class="card stack-form">
        <p class="muted">No settings available at this time.</p>
      </section>
    </section>
  `;
}

