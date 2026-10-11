/* ==========================================================================
   zoom.js: click any image in a chapter to see it full screen.
   For Jupyter Book 2 (MyST book-theme). Styles live in custom.css, section 4.

   Close it with the x button, a click outside the image, or the Esc key.
   To keep one image from zooming, give it (or its figure) :class: no-zoom
   ========================================================================== */
(() => {
  // Take this <script> tag out of the page before the theme starts up,
  // so the theme sees exactly the HTML it rendered.
  if (document.currentScript) document.currentScript.remove();
  if (window.__mystZoom) return;
  window.__mystZoom = true;

  const ZOOMABLE = ':is(article, .article) img';
  const SKIP = 'a img, img.no-zoom, .no-zoom img, .zoom-dialog img';

  let dialog, big, caption;

  // Built on the first click, after the theme has finished loading.
  function build() {
    dialog = document.createElement('dialog');
    dialog.className = 'zoom-dialog';
    dialog.setAttribute('aria-label', 'Enlarged image');
    dialog.innerHTML =
      '<button class="zoom-close" type="button" aria-label="Close image">&times;</button>' +
      '<img class="zoom-img" alt="">' +
      '<p class="zoom-caption"></p>';
    big = dialog.querySelector('.zoom-img');
    caption = dialog.querySelector('.zoom-caption');

    dialog.addEventListener('click', (e) => {
      if (e.target.closest('.zoom-close')) return dialog.close();
      if (e.target === caption) return;
      if (e.target === big && onPicture(e)) return;
      dialog.close(); // anywhere else counts as "outside"
    });
    dialog.addEventListener('close', () => {
      document.documentElement.classList.remove('zoom-open');
      big.removeAttribute('src');
    });
    document.body.appendChild(dialog);
  }

  // The enlarged image is letterboxed (object-fit: contain). A click on the
  // empty letterbox area should close it, so check against the drawn picture.
  function onPicture(e) {
    const box = big.getBoundingClientRect();
    const w0 = big.naturalWidth, h0 = big.naturalHeight;
    if (!w0 || !h0) return true;
    const scale = Math.min(box.width / w0, box.height / h0);
    const w = w0 * scale, h = h0 * scale;
    const left = box.left + (box.width - w) / 2;
    const top = box.top + (box.height - h) / 2;
    return e.clientX >= left && e.clientX <= left + w &&
           e.clientY >= top && e.clientY <= top + h;
  }

  function open(img) {
    if (!dialog || !dialog.isConnected) build();
    big.src = img.currentSrc || img.src;
    big.alt = img.alt || '';
    // Carry over effects like the invert-on-dark filter.
    big.style.filter = getComputedStyle(img).filter;

    const fig = img.closest('figure');
    const text = fig && fig.querySelector('figcaption')
      ? fig.querySelector('figcaption').textContent.trim()
      : '';
    caption.textContent = text;
    caption.hidden = !text;

    document.documentElement.classList.add('zoom-open');
    dialog.showModal();
  }

  // One listener on the whole document. It keeps working when the theme
  // swaps pages without a full reload.
  document.addEventListener('click', (e) => {
    if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const img = e.target;
    if (!(img instanceof HTMLImageElement)) return;
    if (!img.matches(ZOOMABLE) || img.matches(SKIP)) return;
    e.preventDefault();
    open(img);
  });
})();