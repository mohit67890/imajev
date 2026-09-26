// Render synthetic screens / geometry figures (HTML written by scripts/p3/gen_screens.py and gen_geometry.py) to PNG,
// report every labelled element's box, and optionally draw an overlay (marked boxes with letter badges, grid lines with
// cell labels) whose geometry the Python generator computed from a first measuring pass.
//
//   node scripts/p3/screen_templates/render.mjs <jobs.jsonl> <results.jsonl> [concurrency]
//
// jobs rows:    {"id", "html", "out" | null, "width", "height", "dpr"?: 1, "overlay"?: [shape, ...]}
//               out = null -> measure only (no screenshot)
//               shape: {"t": "box", "x", "y", "w", "h", "c"} | {"t": "badge", "x", "y", "w", "h", "text", "c"}
//                      | {"t": "hline" | "vline", "p", "c"}      (CSS px; the generator stores image px = CSS px * dpr)
// results rows: {"id", "ok", "elements": [{"el", "text", "x", "y", "w", "h", "visible", "disabled"}]}  (CSS px, measured
//               BEFORE the overlay is drawn, so the overlay never hides an element from the check)
//
// puppeteer-core from site/node_modules and a local headless Chromium, as scripts/p3/gui_templates/render.mjs.
import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, existsSync, readdirSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';

const here = path.dirname(decodeURIComponent(new URL(import.meta.url).pathname));
const root = path.resolve(here, '../../..');
const require = createRequire(path.join(root, 'site', 'noop.js'));
const puppeteer = require('puppeteer-core');

function findChrome() {
  if (process.env.CHROME) return { exe: process.env.CHROME, headless: 'new' };
  const cache = path.join(homedir(), 'Library/Caches/ms-playwright');
  if (existsSync(cache)) {
    const dirs = readdirSync(cache).filter(d => d.startsWith('chromium_headless_shell-')).sort().reverse();
    for (const d of dirs) {
      const exe = path.join(cache, d, 'chrome-headless-shell-mac-arm64', 'chrome-headless-shell');
      if (existsSync(exe)) return { exe, headless: 'shell' };
    }
  }
  return { exe: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new' };
}

const [jobsPath, outPath, concArg] = process.argv.slice(2);
const jobs = readFileSync(jobsPath, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const conc = Number(concArg || 4);
const { exe, headless } = findChrome();
const browser = await puppeteer.launch({ executablePath: exe, headless, args: ['--no-sandbox', '--font-render-hinting=none'] });
const results = new Array(jobs.length);
let next = 0;

async function worker() {
  const page = await browser.newPage();
  while (next < jobs.length) {
    const i = next++;
    const j = jobs[i];
    try {
      await page.setViewport({ width: j.width, height: j.height, deviceScaleFactor: j.dpr || 1 });
      await page.goto('file://' + j.html, { waitUntil: 'load' });
      const elements = await page.evaluate(() => {
        const W = window.innerWidth, H = window.innerHeight;
        return [...document.querySelectorAll('[data-el]')].map(e => {
          const b = e.getBoundingClientRect();
          const cx = b.left + b.width / 2, cy = b.top + b.height / 2;
          const inView = b.width > 2 && b.height > 2 && b.left >= 0 && b.top >= 0 && b.right <= W && b.bottom <= H;
          let onTop = false;
          if (inView) { const hit = document.elementFromPoint(cx, cy); onTop = !!hit && (hit === e || e.contains(hit) || hit.contains(e)); }
          const st = getComputedStyle(e);
          return { el: e.dataset.el, text: (e.innerText || e.value || '').trim().slice(0, 120), x: Math.round(b.left * 10) / 10,
                   y: Math.round(b.top * 10) / 10, w: Math.round(b.width * 10) / 10, h: Math.round(b.height * 10) / 10,
                   visible: inView && onTop && st.visibility !== 'hidden' && st.opacity !== '0',
                   disabled: !!(e.disabled || e.getAttribute('aria-disabled') === 'true') };
        });
      });
      if (j.out) {
        if (j.overlay && j.overlay.length) {
          await page.evaluate((shapes) => {
            const layer = document.createElement('div');
            layer.style.cssText = 'position:fixed;left:0;top:0;right:0;bottom:0;pointer-events:none;z-index:2147483647';
            for (const s of shapes) {
              const d = document.createElement('div');
              if (s.t === 'box') {
                d.style.cssText = `position:absolute;left:${s.x}px;top:${s.y}px;width:${s.w}px;height:${s.h}px;border:2px solid ${s.c};box-sizing:border-box;border-radius:3px`;
              } else if (s.t === 'badge') {
                d.style.cssText = `position:absolute;left:${s.x}px;top:${s.y}px;width:${s.w}px;height:${s.h}px;background:${s.c};color:#fff;` +
                  `font:700 ${Math.round(s.h * 0.72)}px Helvetica,Arial,sans-serif;display:flex;align-items:center;justify-content:center;` +
                  `border-radius:3px;box-shadow:0 0 0 1px #fff`;
                d.textContent = s.text;
              } else if (s.t === 'hline') {
                d.style.cssText = `position:absolute;left:0;right:0;top:${s.p - 1}px;height:2px;background:${s.c}`;
              } else if (s.t === 'vline') {
                d.style.cssText = `position:absolute;top:0;bottom:0;left:${s.p - 1}px;width:2px;background:${s.c}`;
              }
              layer.appendChild(d);
            }
            document.body.appendChild(layer);
          }, j.overlay);
        }
        await page.screenshot({ path: j.out, type: 'png' });
      }
      results[i] = { id: j.id, ok: true, elements };
    } catch (err) {
      results[i] = { id: j.id, ok: false, error: String(err).slice(0, 300) };
    }
  }
  await page.close();
}
await Promise.all(Array.from({ length: conc }, worker));
await browser.close();
writeFileSync(outPath, results.map(r => JSON.stringify(r)).join('\n') + '\n');
console.log(`rendered ${results.filter(r => r.ok).length}/${jobs.length} with ${exe}`);
