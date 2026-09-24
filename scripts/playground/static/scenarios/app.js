/* imajev scenarios page — no build step. Scenario data and the shared request builder live in
   scenarios.js; this file renders one scenario, sends it to POST /v1/systemone and shows what an
   application would do with the answers. */
'use strict';

const API = (new URLSearchParams(location.search).get('api') || '').replace(/\/+$/, '');
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = x => `${Math.round((Number(x) || 0) * 100)}%`;
const LS_THEME = 'imajev.pg.theme';   // shared with the playground
const CUSTOM = '__your_photo';

const S = {
  scenario: null,
  flips: {},          // path -> value (as sent)
  variant: null,      // label of the chosen photo variant
  edits: {},          // path -> free-text value typed into the record
  threshold: window.DEFAULT_THRESHOLD,
  answers: null,
  lastBody: null,
  lastRequest: null,
  seq: 0,
  inFlight: false,
  pending: false,
  blobs: new Map(),
  custom: null,       // { slot, src } — the viewer's own photo (object URL); never part of the check run
  attribution: {},
};

/* ------------------------------------------------------------------ helpers */

function toast(message) {
  const el = $('toast');
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 1800);
}

function leaves(obj, prefix = '') {
  return Object.entries(obj).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return value && typeof value === 'object' && !Array.isArray(value) ? leaves(value, path) : [[path, value]];
  });
}

function currentCase() {
  const built = window.buildCase(S.scenario, { flips: S.flips, variant: S.variant });
  if (S.custom && S.variant === CUSTOM) built.images[S.custom.slot] = S.custom.src;
  for (const [path, value] of Object.entries(S.edits)) {
    const keys = path.split('.');
    let node = built.state;
    for (const key of keys.slice(0, -1)) node = node[key];
    node[keys[keys.length - 1]] = value;
  }
  return built;
}

async function blobFor(src) {
  if (!S.blobs.has(src)) S.blobs.set(src, fetch(src).then(r => { if (!r.ok) throw new Error(`missing ${src}`); return r.blob(); }));
  return S.blobs.get(src);
}

/* ------------------------------------------------------------------ theme */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $('theme-toggle').textContent = theme === 'dark' ? 'Light' : 'Dark';
  try { localStorage.setItem(LS_THEME, theme); } catch { /* private mode */ }
}

/* ------------------------------------------------------------------ navigation */

function renderList() {
  $('scenario-list').innerHTML = window.SCENARIOS.map((s, i) => `
    <li><a href="#${s.id}" data-id="${s.id}">
      <span class="num">${String(i + 1).padStart(2, '0')}</span>
      <span class="name">${esc(s.title)}</span>
      <span class="sub">${esc(s.kicker)} · ${s.images.length ? `${s.images.length} photo${s.images.length > 1 ? 's' : ''}` : 'no photo'}</span>
    </a></li>`).join('');
}

function readHash() {
  const [id, query] = location.hash.slice(1).split('?');
  const scenario = window.SCENARIOS.find(s => s.id === id) || window.SCENARIOS[0];
  const params = new URLSearchParams(query || '');
  const flips = {};
  let variant = null;
  for (const [key, value] of params) {
    if (key === 'photo') variant = value;
    else if ((scenario.flips || []).some(f => f.path === key)) flips[key] = value;
  }
  return { scenario, flips, variant };
}

function writeHash() {
  const params = new URLSearchParams();
  for (const [path, value] of Object.entries(S.flips)) {
    const flip = S.scenario.flips.find(f => f.path === path);
    const index = flip.values.indexOf(value);
    if (index > 0) params.set(path, flip.names ? flip.names[index] : value);
  }
  if (S.variant && S.variant !== CUSTOM && S.variant !== S.scenario.variants.options[0].label) params.set('photo', S.variant);
  const query = params.toString();
  history.replaceState(null, '', `#${S.scenario.id}${query ? `?${query}` : ''}`);
}

function openScenario({ scenario, flips, variant }) {
  S.scenario = scenario;
  S.flips = {};
  S.edits = {};
  for (const flip of scenario.flips || []) {
    const wanted = flips[flip.path];
    const byName = flip.names ? flip.names.indexOf(wanted) : -1;
    S.flips[flip.path] = byName >= 0 ? flip.values[byName] : flip.values.includes(wanted) ? wanted : flip.values[0];
  }
  S.variant = scenario.variants
    ? (scenario.variants.options.find(o => o.label === variant) || scenario.variants.options[0]).label
    : null;
  S.answers = null;
  if (S.custom) URL.revokeObjectURL(S.custom.src);
  S.custom = null;

  document.querySelectorAll('#scenario-list a').forEach(a => a.setAttribute('aria-current', a.dataset.id === scenario.id ? 'page' : 'false'));
  $('kicker').textContent = scenario.kicker;
  $('title').textContent = scenario.title;
  $('problem').textContent = scenario.problem;
  $('solves').textContent = scenario.solves;
  $('rule').textContent = `// ${scenario.id}: plain application code over the model's probabilities\n// sure(a) = not abstained and top probability ≥ threshold\n${scenario.route.toString()}`;
  renderEvidence();
  renderRecord();
  renderVerdict(null);
  $('cards').innerHTML = '';
  writeHash();
  run();
}

/* ------------------------------------------------------------------ evidence */

const dirOf = src => src.slice(0, src.lastIndexOf('/'));

/* Photo credits come from attribution.json next to each photo, so a page can reuse another page's assets. */
async function loadAttribution() {
  const dirs = new Set();
  for (const s of window.SCENARIOS) {
    for (const image of s.images) dirs.add(dirOf(image.src));
    for (const option of s.variants?.options || []) dirs.add(dirOf(option.src));
  }
  await Promise.all([...dirs].map(async dir => {
    try {
      const response = await fetch(`${dir}/attribution.json`);
      const rows = response.ok ? await response.json() : [];
      S.attribution[dir] = Array.isArray(rows) ? rows : [];
    } catch { S.attribution[dir] = []; }
  }));
}

function creditFor(src) {
  const file = src.split('/').pop();
  return (S.attribution[dirOf(src)] || []).find(a => a.file === file);
}

function renderEvidence() {
  const built = currentCase();
  const scenario = S.scenario;
  $('shots').className = `shots n${built.images.length}`;
  $('shots').innerHTML = built.images.length ? built.images.map((src, i) => {
    const credit = creditFor(src);
    const badge = src.startsWith('blob:') ? '<span class="badge dim" title="Your upload is sent only to this local server and is not part of the check run">your photo · not checked</span>'
      : credit ? (credit.synthetic ? '<span class="badge warn" title="Generated catalogue image, not a photograph">AI-generated</span>'
      : credit.edit ? '<span class="badge warn" title="' + esc(credit.edit) + '">composite</span>'
      : credit.held_out ? '<span class="badge" title="This photo is in no training row of the released model">unseen in training</span>'
      : '<span class="badge dim" title="This photo appears in training rows">seen in training</span>') : '';
    return `<figure class="shot">
      <div class="frame"><img src="${esc(src)}" alt="${esc(scenario.images[i].slot)}"></div>
      <figcaption><span class="slot">${i + 1} · ${esc(scenario.images[i].slot)}</span>${badge}</figcaption>
    </figure>`;
  }).join('') : `<div class="no-photo"><strong>No photo.</strong> The same endpoint answers from the record alone.</div>`;

  const textOnly = !built.images.length;
  document.querySelector('.evidence').hidden = textOnly;
  document.querySelector('.bench').classList.toggle('text-only', textOnly);
  $('record-note').textContent = textOnly ? 'no photo: answered from the record alone' : 'click a value to change the record';
  $('evidence-note').textContent = built.images.length === 2 ? 'two photos, one request' : built.images.length ? 'one photo' : 'text only';

  const variants = scenario.variants;
  const slot = uploadSlot(scenario);
  $('variants').hidden = !variants && slot == null;
  const options = variants ? variants.options.map(o =>
    `<button type="button" class="seg${o.label === S.variant ? ' on' : ''}" data-variant="${esc(o.label)}" aria-pressed="${o.label === S.variant}">${esc(o.label)}</button>`).join('') : '';
  const mine = slot == null ? '' : `<label class="seg upload${S.variant === CUSTOM ? ' on' : ''}">+ your photo<input type="file" id="own-photo" accept="image/jpeg,image/png,image/webp" hidden></label>`;
  $('variants').innerHTML = `<span class="label">Photo ${(variants ? variants.slot : slot) + 1}:</span>${options}${mine}`;

  const seen = new Map();
  for (const src of built.images) {
    const credit = creditFor(src);
    if (credit) seen.set(`${credit.credit}|${credit.edit || ''}`, credit);
  }
  $('credits').innerHTML = [...seen.values()].map(c =>
    `Photo: ${esc(c.credit)}${c.edit ? ` · ${esc(c.edit)}` : ''}`).join('<br>');
}

/* The viewer's own photo replaces the variant slot, or the only / last photo of a scenario without variants. */
function uploadSlot(scenario) {
  if (scenario.variants) return scenario.variants.slot;
  return scenario.images.length ? scenario.images.length - 1 : null;
}

/* Phone photos are large: scale to a 1024-px edge in the browser (the model has an image-size limit). */
async function shrink(file) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 1024 / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
}

async function useOwnPhoto(file) {
  if (!file) return;
  try {
    const blob = await shrink(file);
    if (S.custom) URL.revokeObjectURL(S.custom.src);
    const src = URL.createObjectURL(blob);
    S.blobs.set(src, Promise.resolve(blob));
    S.custom = { slot: uploadSlot(S.scenario), src };
    S.variant = CUSTOM;
    renderEvidence();
    changed({ immediate: true });
  } catch {
    toast('Could not read that image');
  }
}

/* ------------------------------------------------------------------ record */

function renderRecord() {
  const built = currentCase();
  const flips = S.scenario.flips || [];
  const fields = leaves(built.state);
  if (!fields.length) {
    $('record-fields').innerHTML = '<p class="empty-record">No record for this one: imajev answers from the photo alone.</p>';
    return;
  }
  $('record-fields').innerHTML = fields.map(([path, value]) => {
    const flip = flips.find(f => f.path === path);
    const long = String(value).length > 60;
    let control;
    if (flip) {
      control = `<div class="segs">${flip.values.map((v, i) => {
        const label = flip.names ? flip.names[i] : v;
        const on = v === value;
        return `<button type="button" class="seg${on ? ' on' : ''}${i === 0 ? ' orig' : ''}" data-flip="${esc(path)}" data-value="${esc(v)}" aria-pressed="${on}">${esc(label)}</button>`;
      }).join('')}</div>${long ? `<p class="long">${esc(value)}</p>` : ''}`;
    } else if (long) {
      control = `<textarea data-edit="${esc(path)}" rows="3" spellcheck="false">${esc(value)}</textarea>`;
    } else {
      control = `<input data-edit="${esc(path)}" value="${esc(value)}" spellcheck="false">`;
    }
    return `<div class="field${flip ? ' flippable' : ''}"><code class="path">${esc(path)}</code>${control}</div>`;
  }).join('');
}

/* ------------------------------------------------------------------ answers */

function bar(p, cls = '') {
  return `<span class="bar ${cls}"><i style="width:${(Math.max(0, Math.min(1, p)) * 100).toFixed(1)}%"></i></span>`;
}

function card(key, answer, question) {
  const sure = window.sureAt(S.threshold)(answer);
  let body;
  if (answer.type === 'noul') {
    const p = answer.noul;
    body = `<div class="big ${!sure ? 'unsure' : p >= 0.5 ? 'yes' : 'no'}">${pct(p)}<span class="unit">true</span></div>
      <div class="track"><span class="mid"></span><span class="dot" style="left:${(p * 100).toFixed(1)}%"></span></div>`;
  } else if (answer.type === 'choice') {
    const entries = Object.entries(answer.probabilities || {}).sort((a, b) => b[1] - a[1]).slice(0, 3);
    body = `<div class="opts">${entries.map(([name, p], i) =>
      `<div class="opt${i === 0 ? ' top' : ''}"><span class="name">${esc(name)}</span>${bar(p)}<span class="pv">${pct(p)}</span></div>`).join('')}</div>`;
  } else {
    const legend = answer.legend || {};
    const entries = Object.entries(answer.probabilities || {}).sort((a, b) => Number(a[0]) - Number(b[0]));
    const top = entries.reduce((best, e) => (e[1] > best[1] ? e : best), entries[0]);
    body = `<div class="big">${Number(answer.score).toFixed(2)}<span class="unit">of ${entries.length - 1}</span></div>
      <div class="opts">${entries.map(([level, p]) =>
        `<div class="opt${level === top[0] ? ' top' : ''}"><span class="name"><b>${esc(level)}</b> ${esc(legend[level] || '')}</span>${bar(p)}<span class="pv">${pct(p)}</span></div>`).join('')}</div>`;
  }
  const instructions = typeof question.instructions === 'string' ? question.instructions : JSON.stringify(question.instructions);
  return `<article class="card${sure ? '' : ' unsure'}">
    <header><code>${esc(key)}</code><span class="type">${esc(answer.type)}</span></header>
    <p class="instr">${esc(instructions)}</p>
    ${body}
    <footer>
      <span class="unknown${answer.unknown_probability >= 0.25 ? ' high' : ''}">unknown ${pct(answer.unknown_probability)}</span>
      ${answer.abstained ? '<span class="abstained">abstained</span>' : sure ? '<span class="sure">sure enough</span>' : '<span class="notsure">below threshold</span>'}
    </footer>
  </article>`;
}

function renderAnswers() {
  if (!S.answers) return;
  const questions = S.scenario.questions;
  $('cards').innerHTML = Object.entries(S.answers).map(([key, answer]) => card(key, answer, questions[key])).join('');
  renderVerdict(S.scenario.route(S.answers, window.sureAt(S.threshold)));
}

const ICONS = { ok: '✓', flag: '!', human: '?' };
function renderVerdict(route) {
  const el = $('verdict');
  if (!route) {
    el.dataset.kind = 'idle';
    $('verdict-icon').textContent = '·';
    $('verdict-title').textContent = 'Asking imajev…';
    $('verdict-detail').textContent = '';
    return;
  }
  const changed = el.dataset.kind !== route.kind || $('verdict-title').textContent !== route.title;
  el.dataset.kind = route.kind;
  $('verdict-icon').textContent = ICONS[route.kind];
  $('verdict-title').textContent = route.title;
  $('verdict-detail').textContent = route.detail;
  if (changed) { el.classList.remove('pulse'); void el.offsetWidth; el.classList.add('pulse'); }
}

/* ------------------------------------------------------------------ requests */

async function run() {
  if (S.inFlight) { S.pending = true; return; }
  S.inFlight = true;
  S.pending = false;
  const seq = ++S.seq;
  const built = currentCase();
  const scenarioId = S.scenario.id;
  $('main').classList.add('busy');
  $('error').hidden = true;
  try {
    const form = new FormData();
    const request = { state: built.state, questions: built.questions };
    form.append('request', JSON.stringify(request));
    for (const src of built.images) form.append('image', await blobFor(src), src.startsWith('blob:') ? 'your-photo.jpg' : src.split('/').pop());
    const started = performance.now();
    const response = await fetch(`${API}/v1/systemone`, { method: 'POST', body: form });
    const roundTrip = performance.now() - started;
    const body = await response.json().catch(() => ({ error: 'bad_response', detail: `HTTP ${response.status}` }));
    if (!response.ok) throw new Error(`${body.error}: ${body.detail}`);
    if (seq === S.seq && scenarioId === S.scenario.id) {
      S.answers = body.answers;
      S.lastBody = body;
      S.lastRequest = { request, images: built.images };
      renderAnswers();
      const ms = body.usage?.total_ms;
      $('latency').textContent = ms != null ? `${Math.round(ms)} ms` : '';
      $('latency').title = `server ${ms} ms · round trip ${Math.round(roundTrip)} ms`;
      $('meta-line').textContent = `${body.model || ''} · ${Object.keys(body.answers).length} questions · ${built.images.length ? `${built.images.length} photo${built.images.length === 1 ? '' : 's'}` : 'no photo'}`;
      $('json').textContent = JSON.stringify({ request: { ...request, images: built.images.map(s => (s.startsWith('blob:') ? 'your-photo.jpg' : s.split('/').pop())) }, response: body }, null, 2);
    }
  } catch (err) {
    $('error').textContent = `Request failed. ${err.message}. Is the server running? (scripts/playground/README.md)`;
    $('error').hidden = false;
    renderVerdict({ kind: 'human', title: 'No answer', detail: 'The model server did not respond.' });
  } finally {
    S.inFlight = false;
    $('main').classList.remove('busy');
    if (S.pending) run();
  }
}

let debounce = null;
function changed({ immediate = false } = {}) {
  writeHash();
  if (!$('live').checked) return;
  clearTimeout(debounce);
  debounce = setTimeout(run, immediate ? 0 : 250);
}

function curl() {
  if (!S.lastRequest) return '';
  const base = API || location.origin;
  const page = location.pathname.split('/').filter(Boolean).pop() || 'scenarios';
  const images = S.lastRequest.images.map(src => ` \\\n  -F image=@${src.startsWith('blob:') ? 'your-photo.jpg' : `scripts/playground/static/${page}/${src}`}`).join('');
  const json = JSON.stringify(S.lastRequest.request).replace(/'/g, `'\\''`);
  return `curl -s ${base}/v1/systemone \\\n  -F 'request=${json}'${images}`;
}

/* ------------------------------------------------------------------ verification dialog */

async function showVerification() {
  const dialog = $('verification');
  try {
    const { model } = await (await fetch(`${API}/v1/models`)).json();
    const data = await (await fetch(`verification-${model}.json`, { cache: 'no-cache' })).json();
    $('verification-body').innerHTML = `<p class="dim">${data.passed}/${data.total} checks pass on <b>${esc(data.model.model)}</b> at threshold ${pct(data.threshold)} · ${esc(data.run_at.slice(0, 16).replace('T', ' '))} UTC</p>
      <table><thead><tr><th></th><th>Scenario</th><th>Case</th><th>App does</th><th>ms</th></tr></thead><tbody>
      ${data.rows.map(r => `<tr class="${r.pass ? '' : 'fail'}"><td>${r.pass ? '✓' : '✗'}</td><td>${esc(r.scenario)}</td><td>${esc(r.case)}</td><td>${esc(r.route.title)}</td><td>${Math.round(r.server_ms)}</td></tr>`).join('')}
      </tbody></table>`;
  } catch {
    $('verification-body').innerHTML = '<p>No check run found. Run <code>node scripts/playground/verify_scenarios.mjs</code>.</p>';
  }
  dialog.showModal();
}

/* ------------------------------------------------------------------ wiring */

function wire() {
  $('theme-toggle').addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));

  $('scenario-list').addEventListener('click', e => {
    const a = e.target.closest('a[data-id]');
    if (!a) return;
    e.preventDefault();
    openScenario({ scenario: window.SCENARIOS.find(s => s.id === a.dataset.id), flips: {}, variant: null });
    $('main').focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  });
  window.addEventListener('hashchange', () => {
    const next = readHash();
    if (next.scenario !== S.scenario) openScenario(next);
  });

  $('record-fields').addEventListener('click', e => {
    const b = e.target.closest('button[data-flip]');
    if (!b || S.flips[b.dataset.flip] === b.dataset.value) return;
    S.flips[b.dataset.flip] = b.dataset.value;
    renderRecord();
    changed({ immediate: true });
  });
  $('record-fields').addEventListener('input', e => {
    const el = e.target.closest('[data-edit]');
    if (!el) return;
    S.edits[el.dataset.edit] = el.value;
    changed();
  });
  $('variants').addEventListener('change', e => {
    if (e.target.id === 'own-photo') useOwnPhoto(e.target.files[0]);
  });
  $('variants').addEventListener('click', e => {
    const b = e.target.closest('button[data-variant]');
    if (!b || S.variant === b.dataset.variant) return;
    S.variant = b.dataset.variant;
    renderEvidence();
    changed({ immediate: true });
  });

  $('threshold').addEventListener('input', e => {
    S.threshold = Number(e.target.value);
    $('threshold-out').textContent = pct(S.threshold);
    renderAnswers();
  });
  $('run').addEventListener('click', () => run());
  $('live').addEventListener('change', e => { $('run').hidden = e.target.checked; });
  $('run').hidden = $('live').checked;

  for (const [button, pane] of [['toggle-rule', 'rule'], ['toggle-json', 'json']]) {
    $(button).addEventListener('click', () => {
      const open = $(pane).hidden;
      $(pane).hidden = !open;
      $(button).setAttribute('aria-expanded', String(open));
    });
  }
  $('copy-curl').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(curl()); toast('curl command copied'); } catch { toast('Copy failed'); }
  });
  $('show-verification').addEventListener('click', e => { e.preventDefault(); showVerification(); });

  document.addEventListener('keydown', e => {
    if (e.target.closest('input, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
    const i = window.SCENARIOS.indexOf(S.scenario);
    if (e.key === 'j') openScenario({ scenario: window.SCENARIOS[(i + 1) % window.SCENARIOS.length], flips: {}, variant: null });
    if (e.key === 'k') openScenario({ scenario: window.SCENARIOS[(i - 1 + window.SCENARIOS.length) % window.SCENARIOS.length], flips: {}, variant: null });
  });
}

async function init() {
  let theme = 'dark';
  try { theme = localStorage.getItem(LS_THEME) || 'dark'; } catch { /* private mode */ }
  applyTheme(theme);
  renderList();
  wire();
  await loadAttribution();
  fetch(`${API}/v1/models`).then(r => r.json()).then(m => fetch(`verification-${m.model}.json`, { cache: 'no-cache' })).then(r => r.json()).then(v => {
    $('check-summary').textContent = ` (${v.passed}/${v.total} pass on ${v.model.model})`;
  }).catch(() => {});
  fetch(`${API}/v1/models`).then(r => r.json()).then(m => {
    $('model-chip').textContent = `${m.model} · ${m.backend}`;
    $('model-chip').classList.add('ok');
  }).catch(() => { $('model-chip').textContent = 'server offline'; $('model-chip').classList.add('bad'); });
  openScenario(readHash());
}

init();
