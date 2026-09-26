# imajev playground + local API — binding spec

Goal: a local, Jev-style playground for testing the v1 model. Look and feel follow TypeSafe's Jev playground
(left: State JSON + Questions JSON; right: one answer card per question with typed result, probabilities,
confidence, latency) — plus image upload, since our state includes 1–2 images. Local only, no auth.

## Model / backend (Python, existing code)
- Base: Qwen3.5-2B from `artifacts/model.json`. Adapter: `reports/decision-v1/runs/h100x4-full/best` (PEFT/LoRA, rank 16).
- Backend A (preferred, fast): `vision_decision.backend.MLXDirect(bundle, adapter=<mlx adapter dir>)` — needs the PEFT adapter
  converted to mlx-vlm's adapter format (`adapters.safetensors` + `adapter_config.json` with `lora_parameters.keys`); see
  `mlx_vlm.trainer.utils.apply_lora_layers` / `lora_layers.LoRALinear` for the expected keys, shapes and scale. Conversion must
  pass a parity test against the PyTorch path (same image + prompt → same argmax on ≥ 20 cases and max |Δprob| ≤ 0.05).
- Backend B (fallback, slower): `scripts/torch_decision.TorchDecision(path,'mps')` + `peft.PeftModel.from_pretrained` — score one
  field at a time with `candidate_logits` (NEVER left-padded batches on MPS: they produce NaN).
- Request → fields: `vision_decision.jev_api.to_request(payload)`; answers → `vision_decision.jev_api.to_response(request, results)`.
  With MLX use `backend.score_request(images, request.fields, request.state, rotations=1)` (one image prefill for all questions).
- Limits come from `contracts.py`: 1–8 questions, choice ≤ 254 options, score 2–10 levels, string or JSON object state ≤ 128 KB (raised from 32 KB in phase 3), images JPEG/PNG/WebP ≤ 20 MB, 0–2 images.
- **Text-only (added after the first version of this spec):** a request may carry no image. The chat template is rendered with
  `num_images=0`, the vision tower is skipped and the shared state + header prefill is kept, so the same `POST /v1/systemone`
  serves both modes with the same shapes. Three or more images is still a 422.

## HTTP API (FastAPI + uvicorn, `scripts/playground/server.py`, default http://127.0.0.1:8765)
- `GET /v1/models` → `{"model": "...", "adapter": "...", "backend": "mlx"|"torch", "loaded": true, "load_seconds": n}`
- `POST /v1/systemone` — two accepted encodings:
  - `multipart/form-data`: field `request` = JSON string `{"state": ..., "questions": {...}}` (Jev shape; `model` optional and ignored),
    files `image` (1–2; first = reference, second = target);
  - `application/json`: same object plus `"images": ["data:image/jpeg;base64,...", ...]` (1–2).
  Response = `jev_api.to_response(...)` (fields per answer: `type`, and `noul` | `choice`+`probabilities`+`confidence` |
  `score`+`legend`+`probabilities`+`confidence`; always `unknown_probability`, `abstained`) plus
  `"usage": {"prefill_ms": n, "questions_ms": n, "total_ms": n, "input_tokens": n, "images": [{"sha256","width","height"}]}`.
  Errors: 422 `{"error": "...", "detail": "..."}` for validation problems (bad JSON, unsupported type, too many options, more than two images),
  413 for oversized images, 500 with message for runtime failures. Never return a fake answer on error.
- The model loads once at startup and stays resident; requests are served sequentially (a lock) — one GPU.
- `GET /` serves the playground; `GET /examples` returns a list of example requests (JSON) for the "Load example" menu.
- Log one line per request to stdout: time, n images, n questions, total_ms, abstained count.

## Playground UI (single file `scripts/playground/static/index.html` + optional `app.js`/`style.css`; no build step, no CDN
   dependencies except optionally a JSON editor from cdnjs/jsdelivr; must work offline without it)
Left column:
- Image zone: drag-and-drop / click to add 0–2 images, thumbnails with labels "reference" / "target", remove buttons, size shown.
  Images are optional: Run is enabled with none, and the request is then answered from the state alone.
- **State** editor (JSON, monospace, validates on the fly, red outline + message when invalid).
- **Questions** editor (JSON). "Insert question" menu adds a template for noul / choice / score. "Load example" menu (from `/examples`).
- **Run request** button (⌘/Ctrl+Enter), disabled while running; spinner; the last error shown inline.
Right column ("Response"): header with model name and latency shown Jev-style as `prefill_ms + questions_ms`; one card per question:
- key (monospace) + instructions text (italic, grey), type badge (Score / Noul / Choice) on the right,
- **Score**: big "3.39 of 4" + `Confidence: 57%` + "5 levels · 0–4"; expandable legend with a probability bar per level,
- **Noul**: big "99% true" + a 0–1 bar with a marker; below 0.5 shows "x% true" in muted color,
- **Choice**: top 3 options with % and bars (all options when expanded) + `Confidence: 76%` + "N options",
- every card: small `unknown xx%` chip; if `abstained` = true, an amber "Abstained — insufficient evidence / false premise / not listed" badge
  replaces the primary value (the Jev formats still show underneath, dimmed),
- "Raw JSON" toggle for the whole response; "Copy as curl" and "Copy as `vd decide` command" buttons; a history sidebar of the last 20 runs
  (click to reload state/questions/images from memory).
- Dark theme by default with a light toggle; keyboard-friendly; works in Chrome and Safari at ≥ 1100 px wide and degrades to one column below.

## Examples to ship (in `/examples`; use images under `data/images/fashion-smoke/` and `data/decision-v1/*/images/`, referenced by path so the
   server can load them for the "Load example" menu, plus the user's own uploads)
1. Product listing check: state `{"listing": {"title": ..., "color": "blue", "product_type": "shirt"}}`, noul "The garment matches `listing.color`.",
   choice product type (shirt/shoe/bag/other), score photo usability (3 levels).
2. Two-image same-product: reference + target, noul "The second photo shows the item described in `listing`.", choice "which listed field is contradicted".
3. Counting as score: "How many people are visible?" levels 0–6.
4. Abstention showcase: a question about something not in the image (expect abstained/false premise).
5. Document/scene text: a TextVQA image with a choice over OCR tokens.

## Quality gates (I will run these)
- `PYTHONPATH=src:scripts .venv/bin/python -m pytest -q` still passes (add tests for the server's request parsing and the adapter conversion).
- `curl` multipart and JSON requests return Jev-shaped answers; invalid requests return 422 with a message; total_ms < 1500 on MLX for a 3-question request.
- Parity report for the MLX adapter at `reports/adapter-mlx-parity.json`.
- The UI renders the three card types correctly against the real server (I will check in Chrome).
Do not modify `src/vision_decision/contracts.py`, `scoring.py`, `jev_api.py` or `backend.py` except for small, clearly needed fixes described in your report. Do not use git.
