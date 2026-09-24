// Render the landing page's key visuals as light and dark PNGs for the GitHub README and the Hugging Face cards,
// which cannot run the page's scripts. Run after build_site.py, with the site served locally:
//
//   (cd site && python3 -m http.server 8811) &   node site/readme_shots.mjs   # needs puppeteer-core and Chrome
//
// Every image is a picture of real, checked output already on the page; nothing is drawn here.
import puppeteer from 'puppeteer-core';
import { mkdirSync } from 'node:fs';

const OUT = new URL('./assets/readme/', import.meta.url).pathname;
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
mkdirSync(OUT, { recursive: true });
const browser = await puppeteer.launch({ executablePath: CHROME, headless: 'new' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
for (const theme of ['light', 'dark']) {
  const p = await browser.newPage();
  await p.setViewport({ width: 1280, height: 1000, deviceScaleFactor: 2 });
  await p.goto(process.env.SITE || 'http://127.0.0.1:8811/index.html', { waitUntil: 'networkidle0' });
  await p.evaluate(t => { document.documentElement.dataset.theme = t; }, theme);
  await sleep(500);
  async function shot(selector, name, pad = 20) {
    // document coordinates (the clip is relative to the page, not the viewport)
    const r = await p.$eval(selector, e => { const b = e.getBoundingClientRect(); return { x: b.left + scrollX, y: b.top + scrollY, width: b.width, height: b.height }; });
    await p.screenshot({ path: `${OUT}${name}-${theme}.png`, clip: { x: r.x - pad, y: r.y - pad, width: r.width + 2 * pad, height: r.height + 2 * pad } });
  }
  await shot('#sc-card', 'request-listing');
  await p.click('#sc-switch button[data-k="ticket"]'); await sleep(300);
  await shot('#sc-card', 'request-ticket');
  await shot('#apart .hl', 'highlights');
  await shot('#automation .auto', 'automation');
  await shot('#made .ledger', 'training');
  await p.close();
}
await browser.close();
console.log('wrote', OUT);
