export function loaderMarkup(text = "Loading...") {
  return `
    <div class="loader-wrap" role="status" aria-live="polite">
      <div class="loader" aria-hidden="true"></div>
      <p class="muted">${text}</p>
    </div>
  `;
}

export function mountLoader(target, text) {
  if (!target) return;
  target.innerHTML = loaderMarkup(text);
}
