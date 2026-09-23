/* imajev playground — vanilla JS, no build step, no external dependencies.
 *
 * Talks to the local FastAPI server described in docs/playground-spec.md:
 *   GET  /v1/models
 *   GET  /examples            (+ /examples/{n}/image/{i})
 *   POST /v1/systemone        multipart: request=<json string>, image=<file> x0-2 (none = text-only)
 *
 * index.html?mock=1  renders mock.json instead, so the UI can be reviewed with no server.
 * index.html?api=http://127.0.0.1:8765  points the page at a server on another origin.
 */
'use strict';

/* ───────────────────────────── config & tiny helpers ───────────────────────────── */

const PARAMS = new URLSearchParams(location.search);
const MOCK = PARAMS.get('mock') === '1';
const API = (PARAMS.get('api') || '').replace(/\/+$/, '');
const HISTORY_LIMIT = 20;
const MAX_IMAGES = 2;
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
const IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const ROLES = ['reference', 'target'];

const LS = { theme: 'imajev.pg.theme', history: 'imajev.pg.history', draft: 'imajev.pg.draft', ui: 'imajev.pg.ui' };

const apiUrl = path => (API ? API + path : path);
const curlBase = () =>
  API || (location.protocol.startsWith('http') ? location.origin : 'http://127.0.0.1:8765');

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = x => `${Math.round((Number(x) || 0) * 100)}%`;
const pct1 = x => `${((Number(x) || 0) * 100).toFixed(1)}%`;
const clampPct = x => Math.max(0, Math.min(100, (Number(x) || 0) * 100));
const kb = n => (n < 1024 * 1024 ? `${Math.round(n / 1024)} KB` : `${(n / 1048576).toFixed(1)} MB`);
const readJSON = (key, fallback) => {
  try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
  catch { return fallback; }
};
const writeJSON = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* quota / private mode */ } };

/* in-memory app state */
const S = {
  images: [],          // [{ file, url, name, size }]
  response: null,      // last response object
  request: null,       // the request object that produced it ({state, questions})
  history: [],         // [{ id, at, ... }] — JSON parts also persisted to localStorage
  imagesByRun: new Map(), // runId -> [{file, name}] (memory only, never persisted)
  examples: [],
  running: false,
  rawOpen: false,
};

/* ───────────────────────────── theme ───────────────────────────── */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $('theme-toggle').textContent = theme === 'dark' ? 'Light' : 'Dark';
}

function initTheme() {
  applyTheme(localStorage.getItem(LS.theme) || 'dark');
  $('theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem(LS.theme, next); } catch { /* ignore */ }
  });
}

/* ───────────────────────────── toast ───────────────────────────── */

let toastTimer = null;
function toast(message) {
  let node = document.querySelector('.toast');
  if (!node) { node = document.createElement('div'); node.className = 'toast'; document.body.appendChild(node); }
  node.textContent = message;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.remove(), 1800);
}

async function copyText(text, what) {
  try {
    await navigator.clipboard.writeText(text);
    toast(`${what} copied`);
  } catch {
    // Clipboard API needs a secure context; fall back to a selectable textarea.
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); toast(`${what} copied`); }
    catch { toast('Copy failed — see console'); console.log(text); }
    ta.remove();
  }
}

/* ───────────────────────────── images ───────────────────────────── */

function addFiles(files) {
  const rejected = [];
  for (const file of files) {
    if (S.images.length >= MAX_IMAGES) { rejected.push(`${file.name}: at most ${MAX_IMAGES} images`); continue; }
    if (file.type && !IMAGE_TYPES.includes(file.type)) { rejected.push(`${file.name}: ${file.type} is not JPEG/PNG/WebP`); continue; }
    if (file.size > MAX_IMAGE_BYTES) { rejected.push(`${file.name}: ${kb(file.size)} exceeds 20 MB`); continue; }
    S.images.push({ file, url: URL.createObjectURL(file), name: file.name, size: file.size });
  }
  renderImages();
  if (rejected.length) showError(rejected.join('\n'));
}

function removeImage(index) {
  const [gone] = S.images.splice(index, 1);
  if (gone) URL.revokeObjectURL(gone.url);
  renderImages();
}

function renderImages() {
  const box = $('thumbs');
  box.innerHTML = '';
  S.images.forEach((img, i) => {
    const el = document.createElement('figure');
    el.className = 'shot';
    el.innerHTML = `
      <div class="shot-img">
        <img src="${esc(img.url)}" alt="${esc(ROLES[i])} image" title="${esc(img.name)} — click to view full size">
        <button class="remove" type="button" aria-label="Remove ${esc(img.name)}">×</button>
        <button class="expand" type="button" aria-label="View ${esc(img.name)} full size" title="View full size">⤢</button>
      </div>
      <figcaption>
        <span class="role">${esc(ROLES[i])}</span>
        <span class="dims">—</span>
        <span class="name" title="${esc(img.name)} · ${kb(img.size)}">${esc(img.name)}</span>
      </figcaption>`;
    const preview = el.querySelector('img');
    // The pixel size matters for an image model, so show it as soon as the browser knows it.
    const showDims = () => { el.querySelector('.dims').textContent = `${preview.naturalWidth}×${preview.naturalHeight}`; };
    if (preview.complete && preview.naturalWidth) showDims(); else preview.addEventListener('load', showDims);
    preview.addEventListener('click', () => openLightbox(i));
    el.querySelector('.expand').addEventListener('click', () => openLightbox(i));
    el.querySelector('.remove').addEventListener('click', () => removeImage(i));
    box.appendChild(el);
  });
  $('image-count').textContent = `${S.images.length} / ${MAX_IMAGES}`;
  $('dropzone').classList.toggle('full', S.images.length >= MAX_IMAGES);
  $('images-strip').classList.toggle('empty', !S.images.length);
}

/** After a run, mark each tile with what the server actually decoded and scored. */
function annotateImages(usageImages) {
  const tiles = $('thumbs').querySelectorAll('.shot');
  (usageImages || []).forEach((meta, i) => {
    const tile = tiles[i];
    if (!tile || !meta) return;
    tile.classList.add('scored');
    const sha = String(meta.sha256 || '').slice(0, 10);
    tile.querySelector('.shot-img').title =
      `scored by the model as ${meta.width}×${meta.height}${sha ? ` · sha256 ${sha}…` : ''}`;
  });
}

function initDropzone() {
  const zone = $('dropzone');
  const input = $('file-input');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => { addFiles([...input.files]); input.value = ''; });

  for (const type of ['dragenter', 'dragover']) {
    zone.addEventListener(type, e => { e.preventDefault(); zone.classList.add('over'); });
  }
  for (const type of ['dragleave', 'drop']) {
    zone.addEventListener(type, () => zone.classList.remove('over'));
  }
  zone.addEventListener('drop', e => {
    e.preventDefault();
    addFiles([...(e.dataTransfer?.files || [])]);
  });

  // Images are the point of this playground: a drop anywhere in the request pane counts.
  const pane = document.querySelector('.pane-left');
  const strip = $('images-strip');
  for (const type of ['dragenter', 'dragover']) {
    pane.addEventListener(type, e => {
      if (!e.dataTransfer?.types?.includes('Files')) return;
      e.preventDefault();
      strip.classList.add('over');
    });
  }
  pane.addEventListener('dragleave', e => { if (e.target === pane) strip.classList.remove('over'); });
  pane.addEventListener('drop', e => {
    if (!e.dataTransfer?.types?.includes('Files')) return;
    e.preventDefault();
    strip.classList.remove('over');
    if (e.target !== zone && !zone.contains(e.target)) addFiles([...(e.dataTransfer.files || [])]);
  });
  // Dropping anywhere on the left column works too, but never hijack drops elsewhere.
  document.addEventListener('dragover', e => { if (e.dataTransfer?.types?.includes('Files')) e.preventDefault(); });
}

/* ───────────────────────────── JSON editors & validation ───────────────────────────── */

const DEFAULT_STATE = {
  listing: { title: 'Striped cotton shirt', color: 'blue', product_type: 'shirt' },
  channel: 'marketplace',
};

const DEFAULT_QUESTIONS = {
  matches_color: { type: 'noul', instructions: 'The garment in the photo matches `listing.color`.' },
  product_type: {
    type: 'choice',
    instructions: 'What kind of product is shown?',
    criteria: { shirt: 'A shirt or T-shirt', shoe: 'Footwear', bag: 'A handbag or backpack', other: null },
  },
  photo_quality: {
    type: 'score',
    instructions: 'How usable is this photo for a product listing?',
    criteria: ['Unusable: blurry, dark or badly cropped', 'Usable with flaws', 'Clear and well framed'],
  },
};

const TEMPLATES = {
  noul: { type: 'noul', instructions: 'The garment in the photo matches `listing.color`.', criteria: { true: 'It matches', false: 'It does not match' } },
  choice: { type: 'choice', instructions: 'What kind of product is shown?', criteria: { shirt: 'A shirt or T-shirt', shoe: 'Footwear', other: null } },
  score: { type: 'score', instructions: 'How usable is this photo for a product listing?', criteria: ['Unusable', 'Usable with flaws', 'Clear and well framed'] },
};

const ID_RE = /^[A-Za-z][A-Za-z0-9_]{0,63}$/;

/** Validate the State editor. Returns { ok, value, message }. */
function validateState(text) {
  if (!text.trim()) return { ok: true, value: {}, message: 'empty state' };
  let value;
  try { value = JSON.parse(text); }
  catch (e) { return { ok: false, message: `invalid JSON — ${e.message}` }; }
  if (value === null || !['object', 'string'].includes(typeof value) || Array.isArray(value)) {
    return { ok: false, message: 'state must be a JSON object or string' };
  }
  const bytes = new TextEncoder().encode(JSON.stringify(value)).length;
  if (bytes > 32768) return { ok: false, message: `state is ${bytes} bytes; the limit is 32768` };
  return { ok: true, value, message: `${typeof value === 'string' ? 'text' : Object.keys(value).length + ' key(s)'} · ${bytes} bytes` };
}

/** Validate the Questions editor against the contracts.py limits. Returns { ok, value, message }. */
function validateQuestions(text) {
  let value;
  try { value = JSON.parse(text); }
  catch (e) { return { ok: false, message: `invalid JSON — ${e.message}` }; }
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return { ok: false, message: 'questions must be a JSON object of id -> question' };
  }
  const keys = Object.keys(value);
  if (keys.length < 1 || keys.length > 8) return { ok: false, message: `${keys.length} questions; 1–8 allowed` };

  for (const key of keys) {
    const q = value[key];
    const where = `"${key}"`;
    if (!ID_RE.test(key)) return { ok: false, message: `${where}: id must match [A-Za-z][A-Za-z0-9_]{0,63}` };
    if (!q || typeof q !== 'object' || Array.isArray(q)) return { ok: false, message: `${where}: must be an object` };
    if (!q.instructions) return { ok: false, message: `${where}: needs "instructions"` };
    if (q.type === 'noul') {
      if (q.criteria != null && (typeof q.criteria !== 'object' || Array.isArray(q.criteria))) {
        return { ok: false, message: `${where}: noul criteria must be {"true": ..., "false": ...}` };
      }
    } else if (q.type === 'choice') {
      if (!q.criteria || typeof q.criteria !== 'object' || Array.isArray(q.criteria)) {
        return { ok: false, message: `${where}: choice needs a criteria object of option -> description` };
      }
      const n = Object.keys(q.criteria).length;
      if (n < 2 || n > 254) return { ok: false, message: `${where}: ${n} options; 2–254 allowed` };
    } else if (q.type === 'score') {
      if (!Array.isArray(q.criteria)) return { ok: false, message: `${where}: score needs an ordered criteria array` };
      if (q.criteria.length < 2 || q.criteria.length > 10) {
        return { ok: false, message: `${where}: ${q.criteria.length} levels; 2–10 allowed` };
      }
      if (q.criteria.some(level => typeof level !== 'string' || !level.trim())) {
        return { ok: false, message: `${where}: every score level needs a non-empty description` };
      }
    } else {
      return { ok: false, message: `${where}: unsupported type ${JSON.stringify(q.type)}; use noul, choice or score` };
    }
  }
  const counts = keys.reduce((acc, k) => (acc[value[k].type] = (acc[value[k].type] || 0) + 1, acc), {});
  const summary = Object.entries(counts).map(([t, n]) => `${n} ${t}`).join(' · ');
  return { ok: true, value, message: `${keys.length} question(s) · ${summary}` };
}

function refreshValidation() {
  const state = validateState($('state').value);
  const questions = validateQuestions($('questions').value);
  for (const [field, result] of [['state', state], ['questions', questions]]) {
    $(field).closest('.ed').classList.toggle('invalid', !result.ok);
    decorateEditor($(field).closest('.ed'));
    const msg = $(`${field}-msg`);
    msg.textContent = result.message;
    msg.className = `validation ${result.ok ? 'ok' : 'bad'}`;
  }
  $('run').disabled = S.running || !state.ok || !questions.ok;
  writeJSON(LS.draft, { state: $('state').value, questions: $('questions').value });
  return { state, questions };
}

function setEditors(state, questions) {
  $('state').value = JSON.stringify(state ?? {}, null, 2);
  $('questions').value = JSON.stringify(questions ?? {}, null, 2);
  refreshValidation();
}

function initEditors() {
  const draft = readJSON(LS.draft, null);
  if (draft && draft.questions) {
    $('state').value = draft.state || '{}';
    $('questions').value = draft.questions;
  } else {
    setEditors(DEFAULT_STATE, DEFAULT_QUESTIONS);
  }
  for (const id of ['state', 'questions']) {
    $(id).addEventListener('input', refreshValidation);
  }
  $('format-state').addEventListener('click', () => formatEditor('state'));
  $('format-questions').addEventListener('click', () => formatEditor('questions'));

  $('insert-question').addEventListener('change', e => {
    const kind = e.target.value;
    e.target.value = '';
    if (kind) insertQuestion(kind);
  });
  refreshValidation();
}

function formatEditor(id) {
  try { $(id).value = JSON.stringify(JSON.parse($(id).value || '{}'), null, 2); }
  catch { toast('Cannot format invalid JSON'); }
  refreshValidation();
}

function insertQuestion(kind) {
  const field = $('questions');
  const template = structuredClone(TEMPLATES[kind]);
  let parsed;
  try { parsed = JSON.parse(field.value || '{}'); } catch { parsed = null; }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    toast('Fix the Questions JSON first');
    return;
  }
  let name = `${kind}_question`;
  for (let n = 2; name in parsed; n++) name = `${kind}_question_${n}`;
  parsed[name] = template;
  field.value = JSON.stringify(parsed, null, 2);
  refreshValidation();
  field.focus();
  const at = field.value.indexOf(`"${name}"`);
  if (at >= 0) field.setSelectionRange(at, at + name.length + 2);
}

/* ───────────────────────────── layout: draggable splits and folding sections ─────────────────────────────
 * The panes and the two editors are resizable, every section folds, and both survive a reload.
 * All of the motion lives in CSS; this only moves numbers and toggles classes.
 */

const UI = { left: null, stateHeight: null, folded: {} };

function saveUI() { writeJSON(LS.ui, UI); }

function applyLeftWidth(percent) {
  UI.left = Math.max(24, Math.min(76, percent));
  document.querySelector('.split').style.setProperty('--left', `${UI.left}%`);
}

function applyStateHeight(px) {
  const pane = document.querySelector('.pane-left');
  const max = Math.max(120, pane.clientHeight - 260);
  UI.stateHeight = Math.max(52, Math.min(max, px));
  $('ed-state').style.height = `${UI.stateHeight}px`;
}

/** Wire one separator. `axis` is 'x' for the pane split, 'y' for the editor split. */
function initSplit(el, axis, { onMove, onReset, step }) {
  const bodyClass = axis === 'x' ? 'resizing' : 'resizing-v';
  let dragging = false;
  let frame = null;

  // Move and release are watched on the window: pointer capture can be refused, and the pointer
  // regularly leaves a 1px-wide rule mid-drag.
  let pending = null;
  const move = e => {
    if (!dragging) return;
    pending = e;                             // keep the newest position
    if (frame) return;                       // one update per frame is plenty
    frame = requestAnimationFrame(() => {
      frame = null;
      onMove(pending);
      decorateEditors();                     // the wrap points moved, so the mirrors must follow
    });
  };

  const end = () => {
    if (!dragging) return;
    dragging = false;
    if (frame) {                             // flush the last move before saving
      cancelAnimationFrame(frame);
      frame = null;
      onMove(pending);
    }
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', end);
    window.removeEventListener('pointercancel', end);
    el.classList.remove('dragging');
    document.body.classList.remove(bodyClass);
    decorateEditors();
    saveUI();
  };

  el.addEventListener('pointerdown', e => {
    e.preventDefault();
    dragging = true;
    try { el.setPointerCapture(e.pointerId); } catch { /* synthetic or stale pointer */ }
    el.classList.add('dragging');
    document.body.classList.add(bodyClass);
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', end);
    window.addEventListener('pointercancel', end);
  });

  el.addEventListener('dblclick', () => { onReset(); decorateEditors(); saveUI(); });

  el.addEventListener('keydown', e => {
    const back = axis === 'x' ? 'ArrowLeft' : 'ArrowUp';
    const forward = axis === 'x' ? 'ArrowRight' : 'ArrowDown';
    if (e.key !== back && e.key !== forward) return;
    e.preventDefault();
    step(e.key === back ? -1 : 1);
    decorateEditors();
    saveUI();
  });
}

function initLayout() {
  const saved = readJSON(LS.ui, null);
  if (saved) Object.assign(UI, saved);

  const split = document.querySelector('.split');
  applyLeftWidth(UI.left ?? 50);
  if (UI.stateHeight) applyStateHeight(UI.stateHeight);

  initSplit($('vsplit'), 'x', {
    onMove: e => {
      const box = split.getBoundingClientRect();
      applyLeftWidth(((e.clientX - box.left) / box.width) * 100);
    },
    onReset: () => applyLeftWidth(50),
    step: direction => applyLeftWidth((UI.left ?? 50) + direction * 2),
  });

  initSplit($('hsplit'), 'y', {
    onMove: e => applyStateHeight(e.clientY - $('ed-state').getBoundingClientRect().top),
    onReset: () => applyStateHeight(132),
    step: direction => applyStateHeight(($('ed-state').offsetHeight) + direction * 16),
  });

  for (const button of document.querySelectorAll('.fold')) {
    const target = $(button.dataset.fold);
    const setFolded = folded => {
      target.classList.toggle('folded', folded);
      button.setAttribute('aria-expanded', String(!folded));
      button.setAttribute('aria-label', `${folded ? 'Expand' : 'Collapse'} ${button.dataset.fold.replace(/^ed-/, '')}`);
      UI.folded[button.dataset.fold] = folded;
    };
    setFolded(!!UI.folded[button.dataset.fold]);
    button.addEventListener('click', () => {
      setFolded(!target.classList.contains('folded'));
      saveUI();
      // the editor changed size, so its mirrors need measuring again once the transition lands
      setTimeout(decorateEditors, 220);
    });
  }
}

/* ───────────────────────────── code editors: gutter + JSON highlighting ─────────────────────────────
 * A textarea with transparent text sits over a <pre> that renders the same text, highlighted, and a
 * gutter whose numbers are sized to each line's rendered height so wrapped lines stay aligned.
 * No editor library: the page has to work offline from a file:// checkout.
 */

const JSON_TOKEN = /("(?:[^"\\]|\\.)*")(\s*:)?|(-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|\b(true|false|null)\b|([{}[\],:])/g;

/** Highlight one line of JSON. Runs on the raw text and escapes each piece itself. */
function highlightJSON(line) {
  let out = '', last = 0;
  for (const match of line.matchAll(JSON_TOKEN)) {
    const [full, str, colon, num, lit, punct] = match;
    out += esc(line.slice(last, match.index));
    if (str) {
      out += colon
        ? `<span class="tok-key">${esc(str)}</span><span class="tok-punct">${esc(colon)}</span>`
        : `<span class="tok-str">${esc(str)}</span>`;
    } else if (num) out += `<span class="tok-num">${esc(num)}</span>`;
    else if (lit) out += `<span class="tok-lit">${lit}</span>`;
    else out += `<span class="tok-punct">${esc(punct)}</span>`;
    last = match.index + full.length;
  }
  return out + esc(line.slice(last));
}

function syncEditorScroll(ed) {
  const input = ed.querySelector('.ed-input');
  const offset = `translateY(${-input.scrollTop}px)`;
  // Move the CONTENTS of each mirror, never the mirror itself: translating .ed-hl would drag its
  // clipping box up with it and blank out the bottom of the editor.
  for (const inner of ed.querySelectorAll('.ed-code, .ed-nums')) inner.style.transform = offset;
}

function decorateEditor(ed) {
  const input = ed.querySelector('.ed-input');
  const hl = ed.querySelector('.ed-hl');
  const gutter = ed.querySelector('.ed-gutter');
  const lines = input.value.split('\n');

  // A classic scrollbar steals width from the textarea; match it so both wrap identically.
  const bar = input.offsetWidth - input.clientWidth;
  hl.style.paddingRight = `${12 + bar}px`;

  hl.innerHTML = `<div class="ed-code">${lines
    .map(line => `<span class="ln">${highlightJSON(line) || '​'}</span>`).join('')}</div>`;
  const rendered = hl.querySelectorAll('.ln');
  gutter.innerHTML = `<div class="ed-nums">${lines.map((_, i) =>
    `<i style="height:${rendered[i] ? rendered[i].offsetHeight : 0}px">${i + 1}</i>`).join('')}</div>`;
  syncEditorScroll(ed);
}

function decorateEditors() {
  for (const ed of document.querySelectorAll('.ed')) decorateEditor(ed);
}

function initEditorChrome() {
  for (const ed of document.querySelectorAll('.ed')) {
    const input = ed.querySelector('.ed-input');
    input.addEventListener('scroll', () => syncEditorScroll(ed));
  }
  // Wrap points move with the pane width, so the gutter has to be measured again.
  let resizeTimer = null;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(decorateEditors, 80);
  });
  decorateEditors();
}

/* ───────────────────────────── examples ───────────────────────────── */

/** The server's /examples shape is not pinned down, so accept the plausible variants. */
function normaliseExamples(payload) {
  const list = Array.isArray(payload) ? payload
    : Array.isArray(payload?.examples) ? payload.examples
      : [];
  return list.map((item, index) => {
    const body = item.request && typeof item.request === 'object' ? item.request : item;
    const imageCount = Array.isArray(item.images) ? item.images.length
      : Number.isInteger(item.images) ? item.images
        : Number.isInteger(item.image_count) ? item.image_count
          : null; // null = probe /examples/{n}/image/{i} until it 404s
    return {
      index: Number.isInteger(item.id) ? item.id : index,
      name: item.name || item.title || item.label || `Example ${index + 1}`,
      state: body.state ?? {},
      questions: body.questions ?? {},
      imageCount,
    };
  });
}

async function loadExamples() {
  try {
    const payload = MOCK ? (await loadMock()).examples || [] : await (await fetch(apiUrl('/examples'))).json();
    S.examples = normaliseExamples(payload);
  } catch {
    S.examples = [];
  }
  const select = $('load-example');
  select.innerHTML = '<option value="">Load example…</option>';
  if (!S.examples.length) {
    select.disabled = true;
    select.title = 'No /examples available';
    return;
  }
  select.disabled = false;
  S.examples.forEach((ex, i) => {
    const option = document.createElement('option');
    option.value = String(i);
    option.textContent = ex.name;
    select.appendChild(option);
  });
  select.addEventListener('change', async e => {
    const i = e.target.value;
    e.target.value = '';
    if (i !== '') await applyExample(S.examples[Number(i)]);
  });
}

async function applyExample(example) {
  if (!example) return;
  hideError();
  setEditors(example.state, example.questions);
  S.images.splice(0).forEach(img => URL.revokeObjectURL(img.url));
  renderImages();
  if (MOCK) { toast(`Loaded "${example.name}" (images skipped in mock mode)`); return; }

  const wanted = example.imageCount === null ? MAX_IMAGES : Math.min(example.imageCount, MAX_IMAGES);
  for (let i = 0; i < wanted; i++) {
    try {
      const listed = example.images && example.images[i] && example.images[i].url;  // carries a cache-buster when the file changes
      const res = await fetch(apiUrl(listed || `/examples/${example.index}/image/${i}`), { cache: 'no-store' });
      if (!res.ok) break;
      const blob = await res.blob();
      const ext = (blob.type.split('/')[1] || 'jpg').replace('jpeg', 'jpg');
      addFiles([new File([blob], `example${example.index}-${ROLES[i]}.${ext}`, { type: blob.type || 'image/jpeg' })]);
    } catch { break; }
  }
  toast(`Loaded "${example.name}"`);
}

/* ───────────────────────────── running a request ───────────────────────────── */

let mockCache = null;
async function loadMock() {
  if (!mockCache) mockCache = await (await fetch('mock.json')).json();
  return mockCache;
}

function showError(message) {
  const box = $('error');
  box.textContent = message;
  box.hidden = false;
}
function hideError() { $('error').hidden = true; }

/* how long the last request took, and a live clock while one is in flight */
let elapsedTimer = null;

function showElapsed(text, kind = '') {
  const el = $('elapsed');
  el.textContent = text;
  el.className = `elapsed${kind ? ' ' + kind : ''}`;
}

function startElapsedClock(startedAt) {
  clearInterval(elapsedTimer);
  const tick = () => showElapsed(`${((performance.now() - startedAt) / 1000).toFixed(1)}s`, 'live');
  tick();
  elapsedTimer = setInterval(tick, 100);
}

function stopElapsedClock({ serverMs, roundTripMs, failed = false }) {
  clearInterval(elapsedTimer);
  const round = Math.round(roundTripMs);
  if (failed) { showElapsed(`failed after ${round} ms`, 'bad'); return; }
  const total = serverMs == null ? round : Math.round(serverMs);
  showElapsed(`${total} ms`);
  $('elapsed').title = serverMs == null
    ? 'round trip measured in the browser'
    : `server total_ms ${Math.round(serverMs)} · round trip ${round} ms`;
}

function setRunning(running) {
  S.running = running;
  $('spinner').hidden = !running;
  $('run-label').textContent = running ? 'Running…' : 'Run request';
  refreshValidation();
  $('run').disabled = running || $('run').disabled;
}

async function run() {
  if (S.running) return;
  const { state, questions } = refreshValidation();
  if (!state.ok || !questions.ok) { showError('Fix the JSON errors above before running.'); return; }
  // Images are optional: with none the model answers from the state and the questions alone.
  const request = { state: state.value, questions: questions.value };
  hideError();
  setRunning(true);
  const started = performance.now();
  startElapsedClock(started);
  try {
    const response = MOCK ? await runMock() : await postSystemOne(request);
    S.request = request;
    S.response = response;
    renderResponse(request, response);
    annotateImages(response.usage?.images);
    stopElapsedClock({ serverMs: response.usage?.total_ms, roundTripMs: performance.now() - started });
    recordRun({ request, response, elapsed: performance.now() - started });
  } catch (err) {
    showError(String(err.message || err));
    stopElapsedClock({ roundTripMs: performance.now() - started, failed: true });
    recordRun({ request, error: String(err.message || err), elapsed: performance.now() - started });
  } finally {
    setRunning(false);
  }
}

async function runMock() {
  const mock = await loadMock();
  await new Promise(r => setTimeout(r, 350)); // make the spinner visible
  return structuredClone(mock.response);
}

async function postSystemOne(request) {
  const body = new FormData();
  body.append('request', JSON.stringify(request));
  for (const img of S.images) body.append('image', img.file, img.name);

  let res;
  try {
    res = await fetch(apiUrl('/v1/systemone'), { method: 'POST', body });
  } catch (err) {
    throw new Error(`Cannot reach ${curlBase()}/v1/systemone — is the server running?\n${err.message}`);
  }
  const text = await res.text();
  let payload = null;
  try { payload = JSON.parse(text); } catch { /* non-JSON error body */ }
  if (!res.ok) {
    const detail = payload
      ? [payload.error, payload.detail].filter(Boolean).join(' — ') || JSON.stringify(payload)
      : text.slice(0, 500);
    throw new Error(`HTTP ${res.status} ${res.statusText}\n${detail}`);
  }
  if (!payload || typeof payload.answers !== 'object') throw new Error('Response has no "answers" object.');
  return payload;
}

async function loadModelInfo() {
  if (MOCK) {
    $('mock-chip').hidden = false;
    $('model-chip').textContent = 'mock.json';
    $('endpoint-chip').textContent = 'no server — ?mock=1';
    return;
  }
  $('endpoint-chip').textContent = `POST ${curlBase()}/v1/systemone`;
  try {
    const info = await (await fetch(apiUrl('/v1/models'))).json();
    const bits = [info.model || 'model', info.backend, info.adapter ? 'adapter' : null].filter(Boolean);
    $('model-chip').textContent = bits.join(' · ');
  } catch {
    $('model-chip').textContent = 'server not reachable';
  }
}

/* ───────────────────────────── rendering: the answer table ─────────────────────────────
 * One grid row per answer: twist | key + instructions | value | primitive type. Columns line up
 * down the pane so several answers can be read as a table rather than as a stack of cards.
 */

const TYPE_LABEL = { score: 'Score', noul: 'Noul', choice: 'Choice' };
const HEAD_OPTIONS = 3;   // option lines shown before the row has to be expanded

function sortedProbabilities(probabilities) {
  return Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1]);
}

function instructionsOf(request, key) {
  const raw = request?.questions?.[key]?.instructions;
  if (typeof raw === 'string') return raw;
  if (Array.isArray(raw)) return raw.join('\n');
  if (raw && typeof raw === 'object') return Object.entries(raw).map(([k, v]) => `${k}: ${v}`).join('\n');
  return '';
}

const plain = value => ({ html: esc(value), text: String(value) });

/** One `option ..... 12%` line; `bar` adds the probability bar used in the expanded detail. */
function optionLine(label, probability, { top = false, bar = false } = {}) {
  return `<div class="opt${top ? ' top' : ''}">
    <span class="name" title="${esc(label.text)}">${label.html}</span>
    ${bar ? `<span class="bar"><i style="width:0" data-grow="${clampPct(probability).toFixed(1)}"></i></span>` : ''}
    <span class="pv">${pct(probability)}</span>
  </div>`;
}

/** The one quiet meta line under a value: type-specific parts, then the unknown mass. */
function footLine(parts, unknown) {
  const mass = Number(unknown) || 0;
  const bits = [...parts,
    `<span class="unknown${mass >= 0.25 ? ' high' : ''}" title="probability mass on &quot;cannot tell from this evidence&quot; (${pct1(mass)})">unknown ${pct(mass)}</span>`];
  return `<div class="foot">${bits.join('<span class="sep">·</span>')}</div>`;
}

/* one cell builder per primitive type -> { value, cardinality, detail } */

function noulCell(answer) {
  const p = Number(answer.noul) || 0;
  const at = clampPct(p).toFixed(1);
  return {
    value: `<div class="big${p < 0.5 ? ' low' : ''}">${pct(p)}<span class="unit">true</span></div>
      <div class="track" role="img" aria-label="noul ${pct1(p)} true">
        <div class="mid"></div><div class="dot" style="left:0" data-slide="${at}"></div>
      </div>
      ${footLine([`noul ${p.toFixed(3)}`], answer.unknown_probability)}`,
    cardinality: 'true / false',
    detail: '',
  };
}

function choiceCell(answer) {
  const entries = sortedProbabilities(answer.probabilities);
  const topKey = answer.choice ?? (entries[0] && entries[0][0]);
  const topProbability = answer.probabilities?.[topKey];
  return {
    value: `<div class="opts lead">${entries.slice(0, HEAD_OPTIONS)
        .map(([key, p]) => optionLine(plain(key), p, { top: key === topKey })).join('')}</div>
      ${footLine([`confidence ${pct(answer.confidence)}`], answer.unknown_probability)}`,
    cardinality: `${entries.length} options`,
    detail: entries.length > HEAD_OPTIONS
      ? `<div class="opts">${entries.map(([key, p]) =>
          optionLine(plain(key), p, { top: key === topKey, bar: true })).join('')}</div>`
      : '',
  };
}

function scoreCell(answer) {
  const legend = answer.legend || {};
  const probabilities = answer.probabilities || {};
  const levels = Object.keys(probabilities).length ? Object.keys(probabilities) : Object.keys(legend);
  const numeric = levels.map(Number).filter(n => Number.isFinite(n));
  const max = numeric.length ? Math.max(...numeric) : 0;
  const min = numeric.length ? Math.min(...numeric) : 0;
  const entries = levels.map(k => [k, probabilities[k] ?? 0]).sort((a, b) => Number(a[0]) - Number(b[0]));
  const top = entries.reduce((best, e) => (e[1] > best[1] ? e : best), entries[0] || ['', 0])[0];

  const label = key => {
    const text = legend[key] || `level ${key}`;
    return { html: `<span class="lvl">${esc(key)}</span>${esc(text)}`, text: `${key} · ${text}` };
  };
  return {
    value: `<div class="big">${Number(answer.score).toFixed(2)}<span class="unit">of ${max}</span></div>
      ${footLine([`confidence ${pct(answer.confidence)}`], answer.unknown_probability)}`,
    cardinality: `${entries.length} levels · ${min}–${max}`,
    detail: `<div class="opts">${entries.map(([key, p]) =>
      optionLine(label(key), p, { top: key === top, bar: true })).join('')}</div>`,
  };
}

function buildRow(key, answer, request) {
  const type = answer.type || 'noul';
  const cell = type === 'score' ? scoreCell(answer)
    : type === 'choice' ? choiceCell(answer)
      : type === 'noul' ? noulCell(answer)
        : { value: `<pre class="raw">${esc(JSON.stringify(answer, null, 2))}</pre>`, cardinality: '', detail: '' };
  const instructions = instructionsOf(request, key);
  const abstained = answer.abstained === true;

  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `
    <span class="c-twist">${cell.detail
      ? `<button class="twist" type="button" aria-expanded="false" aria-label="Show every outcome for ${esc(key)}"></button>`
      : ''}</span>
    <div class="c-key">
      <div class="key">${esc(key)}</div>
      ${instructions ? `<div class="instructions">${esc(instructions)}</div>` : ''}
    </div>
    <div class="value">
      ${abstained ? '<span class="abst" title="Unknown was the top outcome: insufficient evidence, a false premise, or the right answer not being listed">abstained</span>' : ''}
      <div class="${abstained ? 'dim' : ''}">${cell.value}</div>
    </div>
    <div class="ptype">
      <span class="badge">${esc(TYPE_LABEL[type] || type)}</span>
      ${cell.cardinality ? `<span class="card">${esc(cell.cardinality)}</span>` : ''}
    </div>`;

  if (cell.detail) {
    const wrap = document.createElement('div');
    wrap.className = 'detail-wrap';
    wrap.innerHTML = `<div class="detail">${cell.detail}</div>`;
    row.appendChild(wrap);
    const twist = row.querySelector('.twist');
    twist.addEventListener('click', () => {
      const open = twist.getAttribute('aria-expanded') === 'true';
      twist.setAttribute('aria-expanded', String(!open));
      wrap.classList.toggle('open', !open);
      if (!open) growBars(wrap);           // the bars inside were never released
    });
  }
  return row;
}

/** Bars and markers render at zero and are released after paint, so they glide to their value. */
function growBars(scope) {
  requestAnimationFrame(() => {
    for (const el of scope.querySelectorAll('[data-grow]')) el.style.width = `${el.dataset.grow}%`;
    for (const el of scope.querySelectorAll('[data-slide]')) el.style.left = `${el.dataset.slide}%`;
  });
}

/* "Ran just now" in the Response header, refreshed while the page sits open. */
let ranAt = null;
function renderRan() {
  if (!ranAt) { $('ran-at').textContent = ''; return; }
  const seconds = Math.round((Date.now() - ranAt) / 1000);
  $('ran-at').textContent = seconds < 45 ? 'Ran just now'
    : seconds < 5400 ? `Ran ${Math.round(seconds / 60)}m ago`
      : `Ran ${Math.round(seconds / 3600)}h ago`;
}

function renderResponse(request, response) {
  const usage = response.usage || {};
  const model = response.model || 'model';
  $('latency').innerHTML = usage.prefill_ms != null
    ? `<b>${esc(model)}</b> <i>${Math.round(usage.prefill_ms)}ms + ${Math.round(usage.questions_ms || 0)}ms</i>`
    : `<b>${esc(model)}</b>`;
  ranAt = Date.now();
  renderRan();

  const rows = $('cards');
  rows.innerHTML = '';
  const answers = response.answers || {};
  // Keep the order of the questions as written, then anything the server added.
  const order = [...Object.keys(request?.questions || {}).filter(k => k in answers),
    ...Object.keys(answers).filter(k => !(k in (request?.questions || {})))];
  if (!order.length) {
    rows.innerHTML = '<p class="empty">The response contained no answers.</p>';
  } else {
    order.forEach((key, i) => {
      const row = buildRow(key, answers[key], request);
      row.style.setProperty('--i', i);      // a small stagger as the answers arrive
      rows.appendChild(row);
    });
    growBars(rows);
  }
  $('raw').textContent = JSON.stringify(response, null, 2);
  syncRawToggle();
}

/** The Raw JSON button swaps the table for the raw payload, rather than stacking both. */
function syncRawToggle() {
  const showRaw = S.rawOpen && !!S.response;
  $('raw').hidden = !showRaw;
  $('cards').hidden = showRaw;
  $('toggle-raw').setAttribute('aria-pressed', String(S.rawOpen));
  $('toggle-raw').textContent = showRaw ? 'Table' : 'Raw JSON';
}

/* ───────────────────────────── copy as curl / vd decide ───────────────────────────── */

const shellQuote = s => `'${String(s).replace(/'/g, `'\\''`)}'`;

function currentRequestJSON() {
  const { state, questions } = refreshValidation();
  if (!state.ok || !questions.ok) return null;
  return { state: state.value, questions: questions.value };
}

function curlCommand() {
  const request = S.request || currentRequestJSON();
  if (!request) { toast('Fix the JSON errors first'); return null; }
  const names = S.images.map(i => i.name);
  if (!names.length) {  // text-only: JSON is the shorter encoding when there is nothing to attach
    return [`curl -sS ${curlBase()}/v1/systemone \\`,
      `  -H 'content-type: application/json' \\`,
      `  -d ${shellQuote(JSON.stringify(request))}`].join('\n');
  }
  const lines = [
    `curl -sS ${curlBase()}/v1/systemone \\`,
    `  -F request=${shellQuote(JSON.stringify(request))} \\`,
    ...names.map((name, i) => `  -F image=@${shellQuote(name)}${i === names.length - 1 ? '' : ' \\'}`),
  ];
  return lines.join('\n');
}

function vdCommand() {
  const request = S.request || currentRequestJSON();
  if (!request) { toast('Fix the JSON errors first'); return null; }
  const names = S.images.map(i => i.name);
  return [
    `cat > request.json <<'JSON'`,
    JSON.stringify(request, null, 2),
    `JSON`,
    ``,
    `# from the repo root, with the venv active`,
    `vd decide \\`,
    ...names.map(name => `  --image ${shellQuote(name)} \\`),
    `  --request request.json`,
    `# add:  --adapter <mlx adapter dir>   to run the fine-tuned adapter`,
  ].join('\n');
}

/* ───────────────────────────── history ───────────────────────────── */

function recordRun({ request, response, error, elapsed }) {
  const answers = response?.answers || {};
  const entry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    at: new Date().toISOString(),
    model: response?.model || null,
    questionCount: Object.keys(request.questions || {}).length,
    imageCount: S.images.length,
    imageNames: S.images.map(i => i.name),
    totalMs: Math.round(response?.usage?.total_ms ?? elapsed),
    abstained: Object.values(answers).filter(a => a.abstained).length,
    error: error || null,
    state: request.state,
    questions: request.questions,
    response: response || null,
  };
  S.imagesByRun.set(entry.id, S.images.map(i => ({ file: i.file, name: i.name })));
  S.history.unshift(entry);
  S.history = S.history.slice(0, HISTORY_LIMIT);
  writeJSON(LS.history, S.history);
  renderHistory();
}

function renderHistory() {
  const box = $('history');
  box.innerHTML = '';
  $('history-count').textContent = S.history.length ? `(${S.history.length})` : '';
  if (!S.history.length) {
    box.innerHTML = '<p class="empty">Runs from this browser show up here — the last 20.</p>';
    return;
  }
  for (const entry of S.history) {
    const time = new Date(entry.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const button = document.createElement('button');
    button.className = `hitem${entry.error ? ' err' : ''}`;
    button.type = 'button';
    button.innerHTML = `
      <span class="what">${entry.error ? 'failed — ' + esc(entry.error.split('\n')[0].slice(0, 60))
        : `${entry.questionCount} question(s) · ${entry.imageCount} image(s)`
        + (entry.abstained ? ` · <span class="flag">${entry.abstained} abstained</span>` : '')}</span>
      <span class="ms">${entry.error ? '' : entry.totalMs + 'ms'}</span>
      <span class="when">${esc(time)}${entry.model ? ' · ' + esc(entry.model) : ''}</span>
      <span class="ms">${entry.imageNames.length ? esc(entry.imageNames.join(', ').slice(0, 30)) : ''}</span>`;
    button.addEventListener('click', () => restoreRun(entry));
    box.appendChild(button);
  }
}

function restoreRun(entry) {
  setEditors(entry.state, entry.questions);
  const kept = S.imagesByRun.get(entry.id);
  S.images.splice(0).forEach(img => URL.revokeObjectURL(img.url));
  if (kept) {
    for (const { file, name } of kept) {
      S.images.push({ file, url: URL.createObjectURL(file), name, size: file.size });
    }
  }
  renderImages();
  if (entry.response) {
    S.request = { state: entry.state, questions: entry.questions };
    S.response = entry.response;
    renderResponse(S.request, entry.response);
    annotateImages(entry.response.usage?.images);
    showElapsed(`${entry.totalMs} ms`);
    hideError();
  } else if (entry.error) {
    showError(entry.error);
  }
  if (!kept && entry.imageNames.length) {
    toast('Images are kept in memory only — re-add them to run again');
  }
}

/* ───────────────────────────── wiring ───────────────────────────── */

function initButtons() {
  $('run').addEventListener('click', run);

  $('toggle-raw').addEventListener('click', () => {
    S.rawOpen = !S.rawOpen;
    const swap = () => syncRawToggle();
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (document.startViewTransition && !reduced) {
      const transition = document.startViewTransition(swap);
      // Toggling again mid-flight aborts the previous transition; that is not an error worth raising.
      transition.ready.catch(() => {});
      transition.finished.catch(() => {});
    } else {
      swap();
    }
    if (S.rawOpen && !S.response) toast('No response yet');
  });

  $('copy-curl').addEventListener('click', () => {
    const command = curlCommand();
    if (command) copyText(command, 'curl command');
  });

  $('copy-vd').addEventListener('click', () => {
    const command = vdCommand();
    if (command) copyText(command, 'vd decide command');
  });

  $('clear-history').addEventListener('click', () => {
    S.history = [];
    S.imagesByRun.clear();
    writeJSON(LS.history, S.history);
    renderHistory();
  });

  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); run(); }
  });
}

async function init() {
  initTheme();
  initDropzone();
  initEditors();
  initEditorChrome();
  initLayout();
  initButtons();
  setInterval(renderRan, 30000);
  S.history = readJSON(LS.history, []).slice(0, HISTORY_LIMIT);
  renderHistory();
  renderImages();
  await loadModelInfo();
  await loadExamples();
  if (MOCK) {
    const mock = await loadMock();
    setEditors(mock.state, mock.questions);
    S.request = { state: mock.state, questions: mock.questions };
    S.response = structuredClone(mock.response);
    renderResponse(S.request, S.response);
  }
}

init();


// ---- image lightbox: full-size view of the uploaded reference / target images ----
let lbIndex = 0;
function openLightbox(i) {
  if (!S.images.length) return;
  lbIndex = Math.max(0, Math.min(i, S.images.length - 1));
  const img = S.images[lbIndex];
  const box = $('lightbox'), el = $('lb-img');
  el.src = img.url; el.alt = `${ROLES[lbIndex]} image, full size`;
  el.onload = () => {
    // Small source images (dataset thumbnails) are upscaled so their content is visible; the caption says so.
    const target = Math.min(window.innerWidth * 0.8, window.innerHeight * 0.8);
    const longest = Math.max(el.naturalWidth, el.naturalHeight);
    const factor = longest < 480 ? Math.min(8, Math.max(1, Math.floor(target / longest))) : 1;
    el.style.width = factor > 1 ? `${el.naturalWidth * factor}px` : '';
    el.style.imageRendering = factor >= 4 ? 'pixelated' : 'auto';
    $('lb-caption').innerHTML = `<b>${esc(ROLES[lbIndex])}</b> · ${esc(img.name)} · ${el.naturalWidth}×${el.naturalHeight} px · ${kb(img.size)}` + (factor > 1 ? ` · shown at ${factor}× (source is small)` : '');
  };
  const multi = S.images.length > 1;
  box.querySelector('.lb-prev').hidden = !multi; box.querySelector('.lb-next').hidden = !multi;
  box.hidden = false; document.body.style.overflow = 'hidden'; box.querySelector('.lb-close').focus();
}
function closeLightbox() { $('lightbox').hidden = true; document.body.style.overflow = ''; }
function stepLightbox(d) { if (S.images.length > 1) openLightbox((lbIndex + d + S.images.length) % S.images.length); }
function wireLightbox() {
  const box = $('lightbox'); if (!box) return;
  box.querySelector('.lb-close').addEventListener('click', closeLightbox);
  box.querySelector('.lb-prev').addEventListener('click', e => { e.stopPropagation(); stepLightbox(-1); });
  box.querySelector('.lb-next').addEventListener('click', e => { e.stopPropagation(); stepLightbox(1); });
  box.addEventListener('click', e => { if (e.target === box) closeLightbox(); });
  // Capture phase on window so no other handler can swallow the keys while the viewer is open.
  window.addEventListener('keydown', e => {
    if (box.hidden) return;
    if (e.key === 'Escape' || e.key === 'Esc') { e.preventDefault(); e.stopPropagation(); closeLightbox(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); stepLightbox(-1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); stepLightbox(1); }
  }, true);
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wireLightbox); else wireLightbox();
