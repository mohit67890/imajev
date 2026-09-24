// Run every scenario check of one page (static/<pack>/scenarios.js) against a live playground server.
//
//   node scripts/playground/verify_scenarios.mjs [--pack scenarios|wardrobe] [--api http://127.0.0.1:8765]
//        [--threshold 0.8] [--out reports/scenarios/verification-<pack>.json]
//
// A check passes when the scenario's own route() returns the expected kind and every named answer
// has the expected top choice (or one of a list). Only passing combinations belong in a demo or a GIF.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../..');
const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const API = arg('api', 'http://127.0.0.1:8765');
const PACK = arg('pack', 'scenarios');
const DATA = arg('data', 'scenarios.js');                       // e.g. --pack wardrobe --data stylist.js
const TAG = DATA === 'scenarios.js' ? PACK : `${PACK}-${DATA.replace(/\.js$/, '')}`;
const STATIC = join(HERE, 'static', PACK);
const model = await (await fetch(`${API}/v1/models`)).json();
// Results are per served model: the pages load the file for the model they are talking to.
const OUT = resolve(ROOT, arg('out', `reports/scenarios/${model.model}/verification-${TAG}.json`));
const PAGE_COPY = `${DATA === 'scenarios.js' ? 'verification' : `${DATA.replace(/\.js$/, '')}-verification`}-${model.model}.json`;

// Classic browser scripts: `window` is the global object, so data files can call YES_NO(...) directly.
const sandbox = {};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(readFileSync(join(HERE, 'static/scenarios/engine.js'), 'utf8'), sandbox);
vm.runInContext(readFileSync(join(STATIC, DATA), 'utf8'), sandbox);
const { SCENARIOS, buildCase, sureAt, checksOf, DEFAULT_THRESHOLD } = sandbox;
const threshold = Number(arg('threshold', DEFAULT_THRESHOLD));
const sure = sureAt(threshold);

const topOf = a => (a.type === 'noul' ? a.noul.toFixed(3) : a.type === 'score' ? a.score.toFixed(2) : a.choice);
const rows = [];
for (const scenario of SCENARIOS) {
  for (const check of checksOf(scenario)) {
    const built = buildCase(scenario, check);
    const form = new FormData();
    form.append('request', JSON.stringify({ state: built.state, questions: built.questions }));
    for (const src of built.images) {
      form.append('image', new Blob([readFileSync(join(STATIC, src))], { type: 'image/jpeg' }), src.split('/').pop());
    }
    const started = performance.now();
    const response = await fetch(`${API}/v1/systemone`, { method: 'POST', body: form });
    const roundTrip = performance.now() - started;
    const body = await response.json();
    if (!response.ok) throw new Error(`${scenario.id}: ${JSON.stringify(body)}`);
    const route = scenario.route(body.answers, sure);
    // A noul answer is compared as 'yes' / 'no' (P(yes) ≥ 0.5); choice answers by their top option.
    const gotOf = a => (a.type === 'noul' ? (a.noul >= 0.5 ? 'yes' : 'no') : a.choice);
    const wrong = Object.entries(check.answer || {})
      .filter(([key, value]) => ![].concat(value).includes(gotOf(body.answers[key])))
      .map(([key, value]) => `${key}=${gotOf(body.answers[key])} (want ${[].concat(value).join(' or ')})`);
    const pass = [].concat(check.expect).includes(route.kind) && wrong.length === 0;   // expect: a kind or a list of acceptable kinds
    const label = [check.variant, ...Object.entries(check.flips || {}).map(([k, v]) => `${k}=${v}`)].filter(Boolean).join(', ') || 'as shipped';
    rows.push({
      scenario: scenario.id, case: label, pass, expect: check.expect, got: route.kind, route, wrong,
      answers: Object.fromEntries(Object.entries(body.answers).map(([k, a]) => [k, {
        top: topOf(a), unknown: a.unknown_probability, abstained: a.abstained, probabilities: a.probabilities ?? null,
      }])),
      server_ms: body.usage?.total_ms, round_trip_ms: Math.round(roundTrip),
    });
    const tops = Object.entries(body.answers).map(([k, a]) => `${k}=${topOf(a)}`).join('  ');
    console.log(`${pass ? 'PASS' : 'FAIL'}  ${scenario.id.padEnd(8)} ${label.slice(0, 44).padEnd(44)} ${route.kind.padEnd(5)} ${String(body.usage?.total_ms).padStart(6)} ms  ${tops}${wrong.length ? '  <- ' + wrong.join('; ') : ''}`);
  }
}
const passed = rows.filter(r => r.pass).length;
console.log(`\n${passed}/${rows.length} checks pass at threshold ${threshold} on ${model.model} (${model.adapter})`);
mkdirSync(dirname(OUT), { recursive: true });
const report = JSON.stringify({ model, threshold, api: API, run_at: new Date().toISOString(), passed, total: rows.length, rows }, null, 2) + '\n';
writeFileSync(OUT, report);
writeFileSync(join(STATIC, PAGE_COPY), report);   // shown by the page's "See the check run"
console.log(`wrote ${OUT} and static/${PACK}/${PAGE_COPY}`);
