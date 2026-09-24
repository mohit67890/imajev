// Dump every checked demo case (playground packs + their verification runs) as JSON for site/build_site.py.
//
//   node site/dump_demos.mjs <model-name> > /tmp/demos.json
//
// For each check this re-builds the exact request the page and the checker send (engine.js buildCase) and joins it
// with the recorded answer from reports/scenarios/<model>/verification-<tag>.json, so the site shows only runs that happened.
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const STATIC = join(ROOT, 'scripts/playground/static');
const MODEL = process.argv[2] || 'imajev-4b-raw';
const PACKS = [['scenarios', 'scenarios.js', 'scenarios'], ['text', 'scenarios.js', 'text'], ['wardrobe', 'scenarios.js', 'wardrobe'],
  ['wardrobe', 'stylist.js', 'wardrobe-stylist'], ['tracing', 'scenarios.js', 'tracing']];

const labelOf = check => [check.variant, ...Object.entries(check.flips || {}).map(([k, v]) => `${k}=${v}`)].filter(Boolean).join(', ') || 'as shipped';
const out = { model: MODEL, packs: {} };
for (const [pack, data, tag] of PACKS) {
  const sandbox = {}; sandbox.window = sandbox; vm.createContext(sandbox);
  vm.runInContext(readFileSync(join(STATIC, 'scenarios/engine.js'), 'utf8'), sandbox);
  vm.runInContext(readFileSync(join(STATIC, pack, data), 'utf8'), sandbox);
  const { SCENARIOS, buildCase, checksOf } = sandbox;
  const ver = JSON.parse(readFileSync(join(ROOT, `reports/scenarios/${MODEL}/verification-${tag}.json`), 'utf8'));
  const rows = new Map(ver.rows.map(r => [`${r.scenario}|${r.case}`, r]));
  out.packs[tag] = {
    passed: ver.passed, total: ver.total, threshold: ver.threshold, run_at: ver.run_at,
    scenarios: SCENARIOS.map(s => ({
      id: s.id, title: s.title, kicker: s.kicker, problem: s.problem || '', solves: s.solves || '',
      slots: s.images.map(i => i.slot),
      flips: (s.flips || []).map(f => ({ path: f.path, values: (f.names || f.values).map(String), raw: f.values })),
      variant_slot: s.variants ? s.variants.slot : null,
      cases: checksOf(s).map(check => {
        const built = buildCase(s, check);
        const row = rows.get(`${s.id}|${labelOf(check)}`);
        return {
          label: labelOf(check), flips: Object.fromEntries(Object.entries(check.flips || {}).map(([k, v]) => [k, String(v)])),
          variant: check.variant ?? null, expect: check.expect, want: check.answer || {},
          state: built.state, images: built.images.map(src => `${pack}/${src}`),
          questions: Object.fromEntries(Object.entries(built.questions).map(([k, q]) => [k, {
            type: q.type, instructions: typeof q.instructions === 'string' ? q.instructions.replace(/`/g, '') : q.instructions,
            options: q.type === 'choice' ? Object.keys(q.criteria || {}) : q.type === 'score' ? q.criteria : null,
          }])),
          pass: row ? row.pass : null, wrong: row ? row.wrong : [], route: row ? row.route : null,
          answers: row ? row.answers : null, ms: row ? row.server_ms : null,
        };
      }),
    })),
  };
}
process.stdout.write(JSON.stringify(out));
