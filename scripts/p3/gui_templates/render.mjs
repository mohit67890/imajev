// Render synthetic GUI screens (HTML written by scripts/p3/gen_gui.py) to PNG, and report where every labelled element
// ended up, so the generator only asks about elements that are really visible in the screenshot.
//
//   node scripts/p3/gui_templates/render.mjs <jobs.jsonl> <results.jsonl> [concurrency]
//
// jobs.jsonl rows: {"id", "html", "out", "width", "height"}  (html/out absolute paths)
// results rows:    {"id", "ok", "elements": [{"el", "text", "x", "y", "w", "h", "visible", "disabled"}]}
//
// Uses puppeteer-core from site/node_modules (already in the repo) and a local headless Chromium: $CHROME, else the
// Playwright headless shell cache, else Google Chrome. Nothing is installed.
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
      await page.setViewport({ width: j.width, height: j.height, deviceScaleFactor: 1 });
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
          return { el: e.dataset.el, text: (e.innerText || e.value || '').trim().slice(0, 120), x: Math.round(b.left), y: Math.round(b.top),
                   w: Math.round(b.width), h: Math.round(b.height), visible: inView && onTop && st.visibility !== 'hidden' && st.opacity !== '0',
                   disabled: !!(e.disabled || e.getAttribute('aria-disabled') === 'true') };
        });
      });
      await page.screenshot({ path: j.out, type: 'png' });
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
