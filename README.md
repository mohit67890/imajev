# imajev

**An open Jev-style decision model that also takes images.** Photo + app state + typed questions in, calibrated probabilities out,
locally, about 80 ms per request on a Mac.

imajev answers the kind of question application code needs answered, not prose: *is this listing consistent with the photo*, *which
field of this form does the image contradict*, *does the target still match the reference*, *route this ticket*, *how severe is this
report on a 1–5 scale*. Every answer is a probability distribution over the options you supplied plus an explicit `unknown`, so the
calling code can act, abstain or escalate on a number.

- **API:** the same request and response shapes as TypeSafe's Jev (`POST /v1/systemone`; question types `noul`, `choice`, `score`),
  so anything that speaks the Jev shape can point at a local imajev server by changing the base URL. imajev adds
  `unknown_probability` and `abstained` to every answer.
- **Inputs:** zero, one or two images (reference and target), a string or JSON state up to 32 KB, one to eight questions per request.
- **Output:** one forward pass per question after a shared prefill of state and images; probabilities read from a trained
  decision readout, temperature-calibrated per question type and option count.
- **Runs locally:** MLX on Apple silicon, PyTorch elsewhere. No network calls at inference.

## Two models

| | **imajev-2b** (latency tier) | **imajev-9b** (quality tier, and the teacher) |
|---|---|---|
| Base | Qwen3.5-2B (Apache-2.0) | Qwen3.5-9B (Apache-2.0) |
| Adapter | LoRA r16/α32 on the language layers + 255-code decision readout | same |
| Trained on | 504k human-labelled image and text decisions, ~416k decisions on new photo and text sources labelled by the 9B, 72k image+state and two-image decisions | the 504k human-labelled decisions |
| Latency (Mac Studio, MLX, 3-question text request) | ~80 ms | ~300 ms |
| Strengths | state-versus-photo and reference-versus-target questions; unseen photo sources; abstention | knowledge and text judgement (MMLU 73.8%), typed workflows, generalisation to unseen photo sources |
| Weights | [`mohit67890/imajev-2b`](https://huggingface.co/mohit67890/imajev-2b) | [`mohit67890/imajev-9b`](https://huggingface.co/mohit67890/imajev-9b) |

Use the 2B when the request has an image and latency matters. Use the 9B alone when the questions lean on world knowledge or text
judgement and 300 ms is fine, or as the labeller when you build your own decision data (see *How the 9B teaches the 2B*).

## Quickstart (five minutes)

```sh
git clone https://github.com/mohit67890/imajev && cd imajev
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[serve,mlx]"          # Apple silicon;  elsewhere: pip install -e ".[serve,torch]"
python scripts/download_model.py       # pinned Qwen3.5-2B into .cache/ (4.3 GB)
huggingface-cli download mohit67890/imajev-2b --local-dir adapters/imajev-2b
PYTHONPATH=src:scripts python scripts/playground/server.py \
  --adapter adapters/imajev-2b/mlx --calibration adapters/imajev-2b/calibration.json --port 8765
```

Open http://127.0.0.1:8765/ for the playground, or call the API:

```sh
curl -s http://127.0.0.1:8765/v1/systemone \
  -F 'request={"state":{"listing":{"title":"Blue ceramic mug, 350 ml","colour":"blue"}},
               "questions":{"matches":{"type":"noul","instructions":"Does the photo show the listed item?"},
                            "wrong_field":{"type":"choice","instructions":"Which listing field does the photo contradict?",
                                           "options":["title","colour","none"]}}}' \
  -F image=@photo.jpg
```

```python
import requests
r = requests.post("http://127.0.0.1:8765/v1/systemone", json={
    "state": "Ticket: 'Charged twice for one order, need the duplicate refunded.'",
    "questions": {"queue": {"type": "choice", "instructions": "Route the ticket.",
                            "options": ["billing", "shipping", "account", "other"]},
                  "urgency": {"type": "score", "instructions": "How urgent, 1 low to 5 high?", "min": 1, "max": 5}}})
print(r.json()["answers"]["queue"])   # {"type": "choice", "choice": "billing", "probabilities": {...}, "confidence": ..., "unknown_probability": ..., "abstained": false}
```

On PyTorch, pass `--backend torch --adapter adapters/imajev-2b`. For the 9B: `python scripts/download_model.py --model 9b`,
download `mohit67890/imajev-9b`, and start the server with `--model-bundle artifacts/model-qwen9b.json`.

## Results

All numbers are single-pass, one option order, on held-out data; details, per-source tables and the base-model baselines are in
`results/`. JevBench numbers are on the public splits of a **text-only** benchmark (111 hard / 72 original / 48 easy items) and are
raw (uncalibrated) unless stated.

| Panel | imajev-2b | imajev-9b | Notes |
|---|---:|---:|---|
| v1 image exam, 5 held-out photo sources (4,989) | 53.6% | 55.9% | base Qwen3.5-2B 39.4%; earlier 2B releases 50.7% |
| · real two-image comparisons (177) | 51.4% | 26.6% | |
| State-grounding probe (200, authored) | 73.0% | 72.5% | templated: measures the trained relation, not generalisation |
| Two-image pairs probe (60, authored) | 100% | 81.7% | templated, see above |
| In-distribution test, 35,528 image+text decisions | 86.3%* | 89.0% | *step-50 sibling of the released checkpoint; ECE after calibration 0.03 / 0.007 |
| MMLU-1000, text-only | 45.2% | 73.8% | the 2B abstains on ~25% of knowledge questions |
| typed-decisions test (2,000) | 59.7%* | 66.2% | |
| JevBench hard (111) / original (72) / easy (48) | 43.2 / 84.7 / 100 | 42.3 / 98.6 / 100 | 9B calibrated; kev 4B 42.3 hard, kev 8B 47.3 |

Reading: the 2B is the first open decision model that reads a state against a photo and a target against a reference; on those
questions it matches or exceeds its 9B teacher. Multi-step reasoning (JevBench hard) does not improve with this recipe at either
size, and the 2B abstains more than we would like on pure knowledge questions. See *Limitations*.

## Architecture

- **Readout, not generation.** Each option is bound to one of 255 single-token codes. The prompt renders the state, the images and
  the question with its option list, then a decision position; the logits of the option codes (plus the `unknown` code) at that
  position are the decision. One prefill per request, one forward pass per question, no decoding.
- **Adapter.** LoRA r16/α32 on all language-model projections (attention, MLP, and the DeltaNet projections in Qwen3.5); the vision
  tower is frozen. The readout head is a small float32 matrix over the code tokens, trained jointly.
- **Calibration.** One temperature per (question type, option-count bucket), fitted on a held-out calibration fold; the server
  applies it when started with `--calibration`. Optional option-order averaging (`--rotations N`) trades latency for calibration on
  hard items.
- **Abstention.** `unknown` is a first-class option in training and inference. Insufficient evidence, a false premise, a mismatched
  reference or an answer outside the listed options all train toward `unknown`.

## How the 9B teaches the 2B

1. Train the 9B on the human-labelled mixture (`results/imajev-9b/`).
2. Collect new photos and passages under commercial-use licences and render typed questions from templates
   (`scripts/v2/`, `docs/decision-v2-pseudolabel-spec.md`).
3. Label each candidate with the calibrated 9B twice, in two option orders; keep it when the argmax agrees and the top probability is
   at least 0.6 (or `unknown` at least 0.5). About 74–84% of candidates survive. The kept distribution becomes a soft target.
4. Blend the human-labelled rows with the base 2B's own distribution (0.5 / 0.5) as a regulariser against forgetting, and fine-tune the
   2B briefly from its previous adapter.

The same pipeline works with your own data: point `scripts/v2/pseudo_label.py` at records with `target: null` and the 9B adapter.

## ImajevBench

A benchmark for typed decisions over text **and photos**, with text-only, visual and joint items and an explicit Unknown reference.
The v2.0-lite preview (533 items) and its datasheet are in `bench/`; the harness is `src/imajev_bench`. Frontier API models score
about 99% on the preview, which is disclosed as a sanity check for them; the intended subjects are small local models.

## Data and licences

- Code and adapters: Apache-2.0. Base models: Qwen3.5-2B and Qwen3.5-9B, Apache-2.0.
- Training data is admitted per source under a verified licence receipt (`scripts/v1_text/common.py`). Text sources are
  Apache-2.0, MIT, CC0 or CC-BY / CC-BY-SA datasets. New photo sources: PD12M (CC0), Wikimedia Commons (CC0 / CC-BY, checked
  per file), Open Images (CC-BY-2.0). The 21 image sources inherited from v1 are admitted under their annotation licences with the
  photos remaining under their upstream terms ("policy A"); the receipts and the per-source table are in the results directory.
- No Jev outputs, no paid APIs and no benchmark test items were used in training. JevBench and typed-decisions test items were only
  ever evaluated on.
- The models output probabilities over options you supply. They are not a safety, medical, legal or hiring certificate.

## Limitations

- JevBench hard sits at ~43% for both sizes: the single-pass readout does not acquire multi-step rule application from scale or
  from this data. A reasoning pass before the readout is future work.
- imajev-2b abstains on about a quarter of pure knowledge questions (MMLU 45% vs 54% for its predecessor); the mitigation is a
  per-type offset on the unknown logit, which is being added to the calibration artefact.
- Two-image comparison generalises less than the templated probe suggests: 51% on 177 real held-out pairs.
- Counting is weak at both sizes (~32–40% on the held-out counting panel).
- Calibration temperatures are fitted in distribution; on off-distribution items the raw probabilities are better calibrated than the
  fitted ones.

## Repository map

- `src/vision_decision/` request contracts, MLX backend, Jev API translation, calibration.
- `scripts/playground/` the local server and the playground UI.
- `scripts/train_decision_lora_torch.py`, `scripts/torch_decision.py` training and the PyTorch path.
- `scripts/v2/` data collection, templates and the 9B pseudo-labelling pipeline.
- `src/imajev_bench/`, `bench/` the benchmark.
- `results/` every evaluation we report, with calibration artefacts and JevBench summaries.
- `docs/` specs and the run book.

## Citation

See `CITATION.cff`.

## Acknowledgements

Qwen3.5 (Alibaba) for the base models; TypeSafe's Jev documentation for the request contract this project mirrors; JevBench
(fstandhartinger/jevbench) for the public text splits; kev (jaredpalmer/kev) as the sibling text-only project whose recipe notes
were useful; PD12M (Spawning), Wikimedia Commons and Open Images for photos.
