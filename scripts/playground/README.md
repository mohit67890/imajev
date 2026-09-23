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
- Scenarios, flips, variants, app rules and expected outcomes: `static/scenarios/scenarios.js` (`buildCase` is shared
  by the page and the checker, so a checked combination is exactly what the page sends).
- Check every default, flip and variant against the running server before a demo or a recording:
  `node scripts/playground/verify_scenarios.mjs` → `reports/scenarios/verification.json` (also copied to the page,
  "See the check run"). Only combinations that pass belong in a GIF.
- Photos: `python scripts/playground/build_scenario_assets.py` copies them into `static/scenarios/assets/` and writes
  `attribution.json` with licence, source and whether the photo appears in any training row of the v1 → v2.1 manifests
  (the page shows "unseen in training", "seen in training" or "composite"). Takes ~5 minutes (greps the manifests).
- Deep links keep the state: `#listing?listing.color=red`, `#qc?photo=good part`. `j` / `k` switch scenarios.
