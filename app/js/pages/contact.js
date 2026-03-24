export async function mountContactPage(root) {
  root.innerHTML = `
    <section class="page-enter contact-page">
      <div class="contact-bg-glow">
        <div class="glow-1"></div>
        <div class="glow-2"></div>
        <div class="glow-3"></div>
      </div>
      
      <div class="contact-header">
        <h1 class="contact-title">Contact Us</h1>
        <p class="contact-subtitle">Contact the support team at Stylomex.AI</p>
      </div>

      <div class="contact-grid">
        <!-- Email Box -->
        <div class="contact-box">
          <div class="contact-box-header">
            <svg class="contact-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <rect width="20" height="16" x="2" y="4" rx="2"></rect>
              <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"></path>
            </svg>
            <h2>Email</h2>
          </div>
          <div class="contact-box-body">
            <div class="contact-copy-row">
              <a href="mailto:stylomex.ai@gmail.com" class="contact-link">stylomex.ai@gmail.com</a>
              <button class="contact-copy-btn" data-copy="stylomex.ai@gmail.com" title="Copy">
                <svg class="icon-copy" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path></svg>
                <svg class="icon-check hidden" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"></path></svg>
              </button>
            </div>
          </div>
          <div class="contact-box-footer">
            <p>We respond to all emails within 24 hours.</p>
          </div>
        </div>

        <!-- GitHub Box -->
        <div class="contact-box">
          <div class="contact-box-header">
            <svg class="contact-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M15 22v-4a4.7 4.7 0 0 0-1-3.4c3 0 6-2 6-5.6a6 6 0 0 0-1-3.6c.3-1.2.3-2.4 0-3.6 0 0-1 0-3 1.4-2.7-.5-5.4-.5-8.1 0C6 2 5 2 5 2c-.3 1.2-.3 2.4 0 3.6A5.5 5.5 0 0 0 4 9c0 3.6 3 5.6 6 5.6-.4.5-.7 1.1-.9 1.7v3.7M9 18c-4.5 2-5-2-7-2"></path>
            </svg>
            <h2>GitHub</h2>
          </div>
          <div class="contact-box-body">
            <div class="contact-copy-row">
              <a href="https://github.com/alwynrejiar/style-transfer-ai" target="_blank" class="contact-link truncate">github.com/alwynrejiar/style-transfer-ai</a>
              <button class="contact-copy-btn" data-copy="https://github.com/alwynrejiar/style-transfer-ai" title="Copy">
                <svg class="icon-copy" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path></svg>
                <svg class="icon-check hidden" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"></path></svg>
              </button>
            </div>
          </div>
          <div class="contact-box-footer">
            <p>Check out our latest open-source projects.</p>
          </div>
        </div>

        <!-- Phone Box -->
        <div class="contact-box">
          <div class="contact-box-header">
            <svg class="contact-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
            </svg>
            <h2>Phone</h2>
          </div>
          <div class="contact-box-body">
            <div class="contact-copy-col">
              <div class="contact-copy-row">
                <a href="tel:+916238952923" class="contact-link">+91 62389 52923</a>
                <button class="contact-copy-btn" data-copy="+91 62389 52923" title="Copy">
                  <svg class="icon-copy" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path></svg>
                  <svg class="icon-check hidden" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"></path></svg>
                </button>
              </div>
              <div class="contact-copy-row mt-2">
                <a href="tel:+919037482216" class="contact-link">+91 90374 82216</a>
                <button class="contact-copy-btn" data-copy="+91 90374 82216" title="Copy">
                  <svg class="icon-copy" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path></svg>
                  <svg class="icon-check hidden" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"></path></svg>
                </button>
              </div>
            </div>
          </div>
          <div class="contact-box-footer">
            <p>We're available Mon-Fri, 9am-5pm.</p>
          </div>
        </div>
      </div>

      <div class="contact-online">
        <div class="contact-dotted-bg"></div>
        <div class="contact-online-content">
          <h2 class="contact-online-title">Find us online</h2>
          <div class="contact-socials">
            <a href="https://github.com/alwynrejiar/style-transfer-ai" target="_blank" class="contact-social-btn">
              <svg viewBox="0 0 24 24" fill="none" class="size-4" stroke="currentColor" stroke-width="2"><path d="M15 22v-4a4.7 4.7 0 0 0-1-3.4c3 0 6-2 6-5.6a6 6 0 0 0-1-3.6c.3-1.2.3-2.4 0-3.6 0 0-1 0-3 1.4-2.7-.5-5.4-.5-8.1 0C6 2 5 2 5 2c-.3 1.2-.3 2.4 0 3.6A5.5 5.5 0 0 0 4 9c0 3.6 3 5.6 6 5.6-.4.5-.7 1.1-.9 1.7v3.7M9 18c-4.5 2-5-2-7-2"></path></svg>
              <span>GitHub</span>
            </a>
            <a href="https://x.com/Stylomex_AI" target="_blank" class="contact-social-btn">
              <svg viewBox="0 0 24 24" fill="none" class="size-4" stroke="currentColor" stroke-width="2"><path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z"></path></svg>
              <span>Twitter</span>
            </a>
            <a href="https://www.linkedin.com/in/stylomex-ai-b0a3493b9/" target="_blank" class="contact-social-btn">
              <svg viewBox="0 0 24 24" fill="none" class="size-4" stroke="currentColor" stroke-width="2"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect width="4" height="12" x="2" y="9"></rect><circle cx="4" cy="4" r="2"></circle></svg>
              <span>LinkedIn</span>
            </a>
            <a href="https://www.instagram.com/stylomex.ai/" target="_blank" class="contact-social-btn">
              <svg viewBox="0 0 24 24" fill="none" class="size-4" stroke="currentColor" stroke-width="2"><rect width="20" height="20" x="2" y="2" rx="5" ry="5"></rect><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path><line x1="17.5" x2="17.51" y1="6.5" y2="6.5"></line></svg>
              <span>Instagram</span>
            </a>
          </div>
        </div>
      </div>
    </section>
  `;

  // Apply copy listeners
  root.querySelectorAll(".contact-copy-btn").forEach(btn => {
    btn.addEventListener("click", async () => {
      const text = btn.getAttribute("data-copy");
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        const iconCopy = btn.querySelector(".icon-copy");
        const iconCheck = btn.querySelector(".icon-check");
        if (iconCopy && iconCheck) {
          iconCopy.classList.add("hidden");
          iconCheck.classList.remove("hidden");
          setTimeout(() => {
            iconCopy.classList.remove("hidden");
            iconCheck.classList.add("hidden");
          }, 1500);
        }
      } catch (err) {
        console.error("Failed to copy text", err);
      }
    });
  });
}
