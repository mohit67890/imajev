# imajev playground

A local, Jev-style playground for the v1 decision model: state JSON + typed questions (noul / choice / score) + 0–2 images →
typed answers with probabilities, confidence and an explicit unknown.

## Start
```sh
cd ""
HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts .venv/bin/python scripts/playground/server.py --backend auto --port 8765
```
Then open http://127.0.0.1:8765/ . First request after start takes ~1 s longer (model warm-up). Stop with Ctrl-C.

- `--backend auto` uses the fast MLX path with the converted v1 adapter (`reports/decision-v1/runs/h100x4-full/best-mlx`);
  `--backend torch` uses PyTorch + the original PEFT adapter (about 5x slower); `--no-adapter` runs the untrained base model.
- API: `POST /v1/systemone` (multipart: `request` JSON + `image` files, or JSON with base64 `images`), `GET /v1/models`, `GET /examples`.
  Request/response shapes follow TypeSafe's Jev API (`docs/playground-spec.md`), plus `unknown_probability` / `abstained` per answer.
- Limits: 1–8 questions per request, choice ≤ 25 options, score 2–10 levels, state ≤ 8 KB, **0–2 images** (first = reference, second = target).
  Three or more images is a 422.
- The UI also runs without the model for layout work: `scripts/playground/static/index.html?mock=1`.
- Layout: drag the rule between the panes, or the one under the State editor, to resize (arrow keys work when
  focused, double-click resets); the chevron in each section header folds it. Both are remembered per browser.

## Text-only requests (no image)
Images are optional. A request with zero images renders the chat template with `num_images=0`, skips the vision tower entirely and
keeps the shared prefill over the state and the question header, so every question of the request still costs one forward pass on top
of one prefill. Same endpoint, same shapes, same `unknown` / `abstained` semantics — the model simply has only the state to go on.
Two shipped examples run this way ("Text-only: support ticket triage", "Text-only: resume screening"); the second also shows
object-form instructions (`{"question": ..., "today": ...}`) and a plain-string state.

On text-only work v1 is a decision model out of its training distribution: the decision-v1 adapter was trained on image decisions,
so treat text answers as a capability demo, not as a benchmarked mode. `reports/text-decisions-v1/report.md` measures exactly how far
off it is on a public text benchmark, against the base model and an external open-source baseline.

## Reading the answers
- noul: probability of "yes"; near 0.5 means the model cannot tell (Jev convention).
- choice: `choice` = top option, `probabilities` over the options you supplied, `confidence` = (n·p_max − 1)/(n − 1) as in Jev.
- score: probability-weighted level (0-based), with `legend`; `confidence` as above. Give a subjective
  photo-quality question at least 4 levels — on a 3-level scale v1 puts ~0.99 on unknown and abstains,
  because the training data only ever used a 5-point quality scale. Counting and legibility scales are fine at 3.
- With two images, keep to the question shapes v1 was trained on: booleans ("the second photo shows the listed
  item") and choices whose options are *values* ("what colour is the product in the second photo"). A choice whose
  options are *state field paths* — "which field does photo 2 contradict: `listing.color` / `listing.product_type`" —
  has no two-image training data at all (5,519 such records, every one single-image), so v1 stops resolving which
  photo is meant and answers from the single-image prior at ~1.00 confidence. Ask that one with a single image.
- `unknown_probability` is the mass the model put on "unknown"; `abstained` is true when unknown was the top outcome. It does not say why
  (insufficient evidence, false premise, or the right answer not being listed); add an "other / none of the above" option when that matters.

## Scenarios page (`/scenarios/`)
Seven real-world decisions, each a Jev request over 0–2 photos with one-click changes to the record ("flips") or
to a photo ("variants"), and the app rule that turns the answers into an action (publish / hold / send to a person).
Serve the release 2B and open http://127.0.0.1:8765/scenarios/ :
```sh
HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts .venv/bin/python scripts/playground/server.py --backend mlx \
  --adapter reports/decision-v2.1/runs/h100x4/last-step1361-mlx \
  --calibration reports/decision-v2.1/calibration-v2.1-final.json --model-name imajev-2b
```
- Scenarios, flips, variants, app rules and expected outcomes: `static/scenarios/scenarios.js`. Shared helpers
  (`buildCase`, `sureAt`, `allCombinations`) are in `static/scenarios/engine.js`, used by the pages and the checker,
  so a checked combination is exactly what the page sends.
- Check every default, flip and variant against the running server before a demo or a recording:
  `node scripts/playground/verify_scenarios.mjs --pack scenarios` → `reports/scenarios/verification-scenarios.json`
  (also copied to the page, "See the check run"). Only combinations that pass belong in a GIF.
- "+ your photo" lets a viewer try their own image (resized to 1024 px in the browser); it is labelled "not checked".
- Photos: `python scripts/playground/build_scenario_assets.py` copies them into `static/scenarios/assets/` and writes
  `attribution.json` with licence, source and whether the photo appears in any training row of the v1 → v2.1 manifests
  (the page shows "unseen in training", "seen in training" or "composite"). Takes ~5 minutes (greps the manifests).
- Deep links keep the state: `#listing?listing.color=red`, `#qc?photo=good part`. `j` / `k` switch scenarios.

## Wardrobe page (`/wardrobe/`)
Five everyday clothing decisions on the same engine (`static/wardrobe/scenarios.js`): ordered vs. arrived, dress code,
snap-and-tag, "do I already own this?", and which of my shoes match. Colour, pattern and category only; the model was
not checked on style. Check with `node scripts/playground/verify_scenarios.mjs --pack wardrobe` (every reachable
combination, 49 checks). Photos are 768-px demo copies in `static/wardrobe/assets/demo/`, credits built by
`build_scenario_assets.py --wardrobe` (ABO pieces are also matched against training rows by product ID). The earlier
closet browser and outfit builder are at `/wardrobe/closet.html`.

## Stylist app (`/wardrobe/stylist.html`)
A phone-app demo (phone frame on desktop, full screen on a phone): pick a piece from the closet, or snap one,
and get bottoms / shoes / bag (or top / layer) from the same closet in the colours you like. imajev reads the photo
(colour, pattern; category for a snapped photo) and ranks the closet items for each slot; the app applies the
styling rules and the "never" colours in code before asking, and writes the "why" line from the catalogue tags.
Logic and requests: `static/wardrobe/stylist.js` (shared with the checker). Check every closet piece x colour
preference with `node scripts/playground/verify_scenarios.mjs --pack wardrobe --data stylist.js`
(→ `static/wardrobe/stylist-verification.json`; the app shows "checked ✓ / ✗" on each exact request). The option
wording was revised on these same cases, so the pass rate is development evidence, not a held-out score.

## Tracing pad (`/tracing/`)
A kids' app: pick a letter or number (A C E H L O T, 1 4 7), trace the dotted guide with a finger, tap Check.
imajev reads which character the strokes show (it never sees the guide and is not told the answer: a task in the
record primed it to "see" the asked-for character); the app compares that with the task and measures, per guide
line, how much the strokes cover (every line ≥ 80% for a star). Shapes: `static/tracing/letters.json`; request and
feedback rule: `static/tracing/scenarios.js`; samples: `build_tracing_samples.py` (good / half / wrong / scribble).
Checks: `node scripts/playground/verify_scenarios.mjs --pack tracing` (imajev's reading, 30 checks); the app's
geometry on the same samples is in `reports/scenarios/tracing-geometry.md` (run in the browser console:
`for (const c of TRACING.CHARS) { choose(c); for (const k of ['good','half','wrong','scribble']) { await loadSample(k); console.log(c, k, coverage().weakest); } }`).
Only generated tracings are checked, not real children's handwriting.
