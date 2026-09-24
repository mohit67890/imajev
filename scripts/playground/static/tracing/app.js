/* imajev tracing pad — the app. Request, feedback rule and checks live in scenarios.js (shared with the checker). */
'use strict';

const API = (new URLSearchParams(location.search).get('api') || '').replace(/\/+$/, '');
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = x => `${Math.round((Number(x) || 0) * 100)}%`;
const TR = window.TRACING;
const SIZE = 512;
const PEN = '#25388c';        // same pen as build_tracing_samples.py
const PEN_WIDTH = 22;
const REACH = 0.07 * SIZE;    // a guide point counts as covered if ink is this close
const LS_STARS = 'imajev.tracing.stars';

const S = { mode: 'letters', char: 'A', letters: null, drawing: false, last: null, sample: null, checks: null, busy: false, stars: {} };
try { S.stars = JSON.parse(localStorage.getItem(LS_STARS)) || {}; } catch { S.stars = {}; }

const guide = $('guide').getContext('2d');
const ink = $('ink').getContext('2d', { willReadFrequently: true });

/* ------------------------------------------------------------------ geometry */

function strokePoints(stroke, step = 0.01) {
  if (stroke.arc) {
    const [cx, cy, rx, ry, a0, a1] = stroke.arc;
    const n = Math.ceil(Math.abs(a1 - a0) / 4);
    return Array.from({ length: n + 1 }, (_, i) => {
      const a = (a0 + ((a1 - a0) * i) / n) * Math.PI / 180;
      return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)];
    });
  }
  const out = [];
  stroke.line.slice(1).forEach(([x1, y1], i) => {
    const [x0, y0] = stroke.line[i];
    const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / step));
    for (let k = 0; k < n; k++) out.push([x0 + ((x1 - x0) * k) / n, y0 + ((y1 - y0) * k) / n]);
  });
  out.push(stroke.line[stroke.line.length - 1]);
  return out;
}

/* How much of each guide line the child's ink passes over (app geometry, not the model). A short line such as
   the crossbar of an A barely moves an average, so the star uses the weakest line: every line must be traced. */
function coverage() {
  const data = ink.getImageData(0, 0, SIZE, SIZE).data;
  const inked = (x, y) => x >= 0 && y >= 0 && x < SIZE && y < SIZE && data[(Math.round(y) * SIZE + Math.round(x)) * 4 + 3] > 40;
  const perLine = [];
  for (const stroke of S.letters[S.char]) {
    let total = 0, hit = 0;
    for (const [gx, gy] of strokePoints(stroke, 0.02)) {
      total++;
      const x = gx * SIZE, y = gy * SIZE;
      search: for (let r = 0; r <= REACH; r += 6) {
        for (let a = 0; a < 360; a += r ? 30 : 360) {
          if (inked(x + r * Math.cos(a * Math.PI / 180), y + r * Math.sin(a * Math.PI / 180))) { hit++; break search; }
        }
      }
    }
    perLine.push(total ? hit / total : 0);
  }
  return { weakest: Math.min(...perLine), lines: perLine };
}

function inkAmount() {
  const data = ink.getImageData(0, 0, SIZE, SIZE).data;
  let n = 0;
  for (let i = 3; i < data.length; i += 16) if (data[i] > 40) n++;
  return n;
}

/* ------------------------------------------------------------------ drawing */

function drawGuide() {
  guide.clearRect(0, 0, SIZE, SIZE);
  guide.lineCap = 'round';
  guide.lineJoin = 'round';
  for (const stroke of S.letters[S.char]) {
    const pts = strokePoints(stroke);
    guide.strokeStyle = '#e9e2cf';
    guide.lineWidth = 46;
    guide.setLineDash([]);
    guide.beginPath();
    pts.forEach(([x, y], i) => (i ? guide.lineTo(x * SIZE, y * SIZE) : guide.moveTo(x * SIZE, y * SIZE)));
    guide.stroke();
    guide.strokeStyle = '#b9ad8c';
    guide.lineWidth = 6;
    guide.setLineDash([2, 16]);
    guide.stroke();
  }
  guide.setLineDash([]);
  S.letters[S.char].forEach((stroke, n) => {       // numbered start dots, like a workbook
    const [x, y] = strokePoints(stroke)[0];
    guide.fillStyle = '#3fb37f';
    guide.beginPath();
    guide.arc(x * SIZE, y * SIZE, 17, 0, Math.PI * 2);
    guide.fill();
    guide.fillStyle = '#fff';
    guide.font = 'bold 20px ui-rounded, system-ui, sans-serif';
    guide.textAlign = 'center';
    guide.textBaseline = 'middle';
    guide.fillText(String(n + 1), x * SIZE, y * SIZE + 1);
  });
}

function clearInk() {
  ink.clearRect(0, 0, SIZE, SIZE);
  S.sample = null;
  S.last = null;
  $('feedback').hidden = true;
  $('readout').innerHTML = '';
  renderBadge();
}

function pos(e) {
  const r = $('ink').getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * SIZE, ((e.clientY - r.top) / r.height) * SIZE];
}

function wirePad() {
  const canvas = $('ink');
  ink.lineCap = 'round';
  ink.lineJoin = 'round';
  canvas.addEventListener('pointerdown', e => {
    if (S.busy) return;
    if (S.sample || !$('feedback').hidden) clearInk();
    canvas.setPointerCapture(e.pointerId);
    S.drawing = true;
    S.last = pos(e);
    ink.fillStyle = PEN;
    ink.beginPath();
    ink.arc(S.last[0], S.last[1], PEN_WIDTH / 2, 0, Math.PI * 2);
    ink.fill();
    e.preventDefault();
  });
  canvas.addEventListener('pointermove', e => {
    if (!S.drawing) return;
    const p = pos(e);
    ink.strokeStyle = PEN;
    ink.lineWidth = PEN_WIDTH;
    ink.beginPath();
    ink.moveTo(S.last[0], S.last[1]);
    ink.lineTo(p[0], p[1]);
    ink.stroke();
    S.last = p;
    e.preventDefault();
  });
  const end = () => { S.drawing = false; };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
}

/* A sample PNG is ink on white: make white transparent so coverage and the upload match a real drawing. */
async function loadSample(kind) {
  // fetch + createImageBitmap rather than Image.decode(), which can stall in a background tab.
  const img = await createImageBitmap(await (await fetch(`samples/${S.char}-${kind}.png`)).blob());
  clearInk();
  ink.drawImage(img, 0, 0, SIZE, SIZE);
  const frame = ink.getImageData(0, 0, SIZE, SIZE);
  for (let i = 0; i < frame.data.length; i += 4) {
    const light = (frame.data[i] + frame.data[i + 1] + frame.data[i + 2]) / 3;
    frame.data[i + 3] = light > 235 ? 0 : 255;
  }
  ink.putImageData(frame, 0, 0);
  S.sample = kind;
  renderBadge();
}

/* ------------------------------------------------------------------ check */

function inkOnWhite() {
  const canvas = document.createElement('canvas');
  canvas.width = SIZE;
  canvas.height = SIZE;
  const c = canvas.getContext('2d');
  c.fillStyle = '#fff';
  c.fillRect(0, 0, SIZE, SIZE);
  c.drawImage($('ink'), 0, 0);
  return new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
}

async function check() {
  if (S.busy) return;
  if (inkAmount() < 30) { showFeedback({ kind: 'human', title: 'Draw first!', detail: 'Trace the dotted lines with your finger.' }); return; }
  S.busy = true;
  $('check').disabled = true;
  $('check').textContent = 'Looking…';
  try {
    const covered = coverage();
    // A checked sample sends its original file, byte for byte what the check run sent; a drawing sends the canvas.
    const blob = S.sample ? await (await fetch(`samples/${S.char}-${S.sample}.png`)).blob() : await inkOnWhite();
    const form = new FormData();
    form.append('request', JSON.stringify({ state: TR.state(), questions: TR.questions() }));
    form.append('image', blob, 'tracing.png');
    const response = await fetch(`${API}/v1/systemone`, { method: 'POST', body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || response.status);
    const verdict = TR.route(body.answers, window.sureAt(window.DEFAULT_THRESHOLD), S.char, covered.weakest);
    showFeedback(verdict);
    const drawn = body.answers.drawn;
    $('readout').innerHTML = `<span>imajev read <b>${esc(drawn.choice === TR.SCRIBBLE ? 'a scribble' : drawn.choice)}</b> ${pct(drawn.probabilities[drawn.choice])}</span>
      <span>lines traced <b>${covered.lines.filter(x => x >= TR.COVERED).length} of ${covered.lines.length}</b> · weakest ${pct(covered.weakest)}<i class="meter"><i style="width:${(covered.weakest * 100).toFixed(0)}%"></i></i></span>`;
    $('latency').textContent = `${Math.round(body.usage?.total_ms ?? 0)} ms · ${body.model} · star needs imajev to read ${S.char} (≥ ${pct(window.DEFAULT_THRESHOLD)}) and every line ≥ ${pct(TR.COVERED)} traced`;
    if (verdict.kind === 'ok' && !S.sample) {
      S.stars[S.char] = (S.stars[S.char] || 0) + 1;
      try { localStorage.setItem(LS_STARS, JSON.stringify(S.stars)); } catch { /* private mode */ }
      renderPicker();
    }
  } catch (err) {
    showFeedback({ kind: 'human', title: 'Oops', detail: `The checker is not answering (${err.message}).` });
  } finally {
    S.busy = false;
    $('check').disabled = false;
    $('check').textContent = 'Check ✓';
  }
}

function showFeedback(v) {
  const el = $('feedback');
  el.dataset.kind = v.kind;
  $('fb-title').textContent = v.title;
  $('fb-detail').textContent = v.detail;
  el.hidden = false;
  el.classList.remove('pop');
  void el.offsetWidth;
  el.classList.add('pop');
}

/* ------------------------------------------------------------------ chrome */

function renderPicker() {
  const chars = TR.CHARS.filter(c => (S.mode === 'numbers') === TR.isDigit(c));
  $('picker').innerHTML = chars.map(c => `<button role="tab" data-char="${c}" aria-selected="${c === S.char}">
    ${c}${S.stars[c] ? `<span class="got">★${S.stars[c] > 1 ? S.stars[c] : ''}</span>` : ''}</button>`).join('');
  const total = Object.values(S.stars).reduce((a, b) => a + b, 0);
  $('stars').textContent = `★ ${total}`;
  $('mode-title').textContent = S.mode === 'numbers' ? 'Numbers' : 'Letters';
  document.querySelectorAll('.modes button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.mode === S.mode)));
  $('samples').innerHTML = ['good', 'half', 'wrong', 'scribble'].map(k => `<button data-sample="${k}">${k}</button>`).join('');
}

function choose(char) {
  S.char = char;
  clearInk();
  drawGuide();
  renderPicker();
}

function renderBadge() {
  const el = $('check-badge');
  if (!S.sample) { el.textContent = ''; return; }
  const row = S.checks?.rows.find(r => r.scenario === `trace-${S.char}` && r.case === S.sample);
  el.textContent = row ? (row.pass ? `sample “${S.sample}” · checked ✓` : `sample “${S.sample}” · checked ✗`) : `sample “${S.sample}” · not in check run`;
  el.className = `badge ${row ? (row.pass ? 'ok' : 'bad') : ''}`;
}

async function loadChecks() {
  try {
    const { model } = await (await fetch(`${API}/v1/models`)).json();
    S.checks = await (await fetch(`verification-${model}.json`, { cache: 'no-cache' })).json();
    $('check-line').textContent = ` (${S.checks.passed}/${S.checks.total} pass on ${S.checks.model.model})`;
  } catch { S.checks = null; }
}

function showChecks() {
  const c = S.checks;
  $('checks-body').innerHTML = c ? `<p class="small">${c.passed}/${c.total} pass on <b>${esc(c.model.model)}</b>. Each generated tracing (good, the wrong character, a scribble) is sent exactly as the page sends it; a check passes when imajev reads the right character and the feedback is the expected kind. Half-finished tracings are judged by the app's coverage check, so they are not in this run.</p>
    <table><thead><tr><th></th><th>Task</th><th>Sample</th><th>imajev read</th><th>Feedback</th></tr></thead><tbody>${c.rows.map(r => `
      <tr class="${r.pass ? '' : 'fail'}"><td>${r.pass ? '✓' : '✗'}</td><td>${esc(r.scenario.replace('trace-', ''))}</td><td>${esc(r.case)}</td>
      <td>${esc(r.answers.drawn.top === TR.SCRIBBLE ? 'scribble' : r.answers.drawn.top)} ${pct(Math.max(...Object.values(r.answers.drawn.probabilities)))}</td><td>${esc(r.route.title)}</td></tr>`).join('')}
    </tbody></table>` : '<p>No check run found. Run <code>node scripts/playground/verify_scenarios.mjs --pack tracing</code>.</p>';
  $('checks-dialog').showModal();
}

function wire() {
  $('picker').addEventListener('click', e => { const b = e.target.closest('button[data-char]'); if (b) choose(b.dataset.char); });
  document.querySelector('.modes').addEventListener('click', e => {
    const b = e.target.closest('button[data-mode]');
    if (!b || b.dataset.mode === S.mode) return;
    S.mode = b.dataset.mode;
    choose(S.mode === 'numbers' ? '1' : 'A');
  });
  $('clear').addEventListener('click', clearInk);
  $('check').addEventListener('click', check);
  $('samples').addEventListener('click', e => { const b = e.target.closest('button[data-sample]'); if (b) loadSample(b.dataset.sample); });
  $('about-checks').addEventListener('click', e => { e.preventDefault(); showChecks(); });
}

async function init() {
  S.letters = await (await fetch('letters.json')).json();
  wirePad();
  wire();
  // Deep link: #sample=A-good opens that character, loads the sample and checks it (used for screenshots and links).
  const [char, kind] = (new URLSearchParams(location.hash.slice(1)).get('sample') || '').split('-');
  choose(char && S.letters[char] ? char : 'A');
  await loadChecks();
  if (kind) { await loadSample(kind); await check(); }
}

init();
