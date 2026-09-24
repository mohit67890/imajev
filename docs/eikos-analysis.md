# Eikos: code-level analysis for imajev phase 2c

Written 2026-09-24. Sources:
- code: `github.com/caiovicentino/eikos` at commit `bb43e78`, cloned to the session scratchpad (`.../scratchpad/eikos/eikos`). Every `path:line` below points into that clone.
- model cards: `caiovicentino1/Eikos-4B` @`3501f815`, `caiovicentino1/Eikos-27B` @`136685e2` (cited as *4B card*, *27B card*).
- dataset card and all rows: `caiovicentino1/eikos-decisions` @`26d9a680`. This is the revision `scripts/p2/filter_eikos.py` pins. The counts in section 4 are my own counts over the released jsonl files.

Summary of the findings that change our plan:
1. **Eikos starts from the post-trained `Qwen/Qwen3.5-4B`, not from `-Base`.** The card YAML says `base_model: Qwen/Qwen3.5-4B`, `scripts/train_final.sh:19` uses the same model, and I checked the weights. Eikos-4B's untouched layer norms (`layers.0.input_layernorm`, `post_attention_layernorm`, final `norm`) are byte-identical to `Qwen/Qwen3.5-4B` and differ from `Qwen/Qwen3.5-4B-Base`. It is the same starting checkpoint as ours.
2. **GLM-5.3-Flash is not API-only.** It is `zai-org/GLM-5.3-Flash`, MIT, about 321B parameters. The dataset card says the authors ran it on their own infrastructure. **Every** generated row was blind-labelled by GLM, including the 7,869 rows that Qwen3.8-27B wrote. Our `strict` filter therefore keeps only programmatic rows, and it drops the 4,236 teacher rows in the JevBench-hard family names (judge_hard, multi_hop, long_policy, trap, …). Those rows are the part of Eikos most likely to drive JevBench hard.
3. **"1 epoch" is really about 0.8 epoch for the 4B.** `steps_total` is computed from `len(train)/BS`, but `TOK_BUDGET` splits long items into extra micro-batches, and the loop stops at `steps_total` (`training/train_dec.py:569,583,654`). I simulated the batch plan on the released rows: about 3,600 micro-batches are built, only 2,896 are consumed, so about 19–20% of the 4B training items are never seen. The 27B, with batches of 2 and dossiers of at most 12k tokens, sees 99.96%.
4. **No inference-time tricks.** There is no permutation averaging (`sym` is off everywhere reported) and T = 1 (`calib.json` is the identity). The card's JevBench numbers for Jev and Laya come from the official v1.2 leaderboard file, not from a run of their own.

---

## 1. Readout

**What is scored.** Only the next-token logits of the single-token uppercase letters `A`…`Z`, at the first generated position.
- `LETTERS = [chr(65 + i) for i in range(26)]` (`eikos/decision_core.py:17`).
- The letter ids are asserted to be single, distinct tokens (`eikos/letter_adapter.py:103-111`).
- The softmax runs over the first `len(opts)` letters only, with temperature T (`eikos/letter_adapter.py:157-160`; `decision_core.temp_for`, `eikos/decision_core.py:71-77`).
- Option text is never scored, and there is no dedicated decision token. The lm_head is tied to the embedding, so the letter logits are `embed_tokens` rows (`eikos/mlx_decide.py:44-50`).

**Position.** The position is the last prompt token after `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`, so it comes after Qwen's empty `<think>\n\n</think>\n\n` block.
- training: `training/train_dec.py:226-230, 264-265`;
- PyTorch inference: `eikos/letter_adapter.py:181-182, 211-212`;
- vLLM: `/v1/completions` with `max_tokens=1, temperature=0, logprobs=n, allowed_token_ids=letters[:n]` (`eikos/letter_adapter.py:89-90`; `evaluation/eval_vllm_suite.py:46-47`).

**Prompt template** (the released style is `semif`, set in `decision_config.json` as `"prompt_version": "letter-v1-semif"`):
- system: `"Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning."` (`eikos/decision_core.py:19-21`).
- user: the compact JSON `json.dumps({"evidence": state, "criterion": instructions, "options": [{"letter": "A", "description": "<label>: <description>"}, ...]}, ensure_ascii=False)` (`eikos/decision_core.py:53-60`). An object state is embedded as a JSON object, not stringified.

**Type mapping** (`eikos/decision_core.py:34-50`):
- `noul`/`boolean` becomes two options, `[("yes", criteria.true), ("no", criteria.false)]`, and the labels are shown as `true: …` / `false: …` (`:55-58`). At inference yes is always A. In training the pair is shuffled (section 2).
- `score` with list criteria becomes options `"0".."k"` in natural order, shown as `"0: <level text>"`. The server returns `score = argmax`, `expected = Σ k·p_k` and `confidence = p_max` (`eikos/serve.py:86-88`).
- `choice` becomes the criteria keys in order; each description is `str(crit.get(k, k))`. A `null` description renders as `"None"` and a structured criterion as a Python `repr`, which are edge cases where Eikos differs from the Jev contract.

**Unknown / abstention.** There is no `unknown` option anywhere: not in the prompt, the data or the server. Abstention is left to the caller as a confidence threshold. The card: "callers can abstain below a threshold". The reported policy is "decide on its own if max-prob ≥ 0.90" (`evaluation/release_table.py:58,70`). The `ambiguous` family asks the writer for an explicit `cannot_determine`-style label as an ordinary option (`data_pipeline/gen_pipeline.py:60-63`). 458 training rows carry such an option, and in 52 of them it is gold.

**Maximum options.** 26 letters. Beyond 26, a tournament runs blocks of 20; the top 2 of each block go to a final of at most 26, and the losers get 1e-4 of mass (`eikos/decision_core.py:80-94`; the vLLM batch version is in `evaluation/eval_vllm_suite.py:86-96`). The `large_choice` training family uses 10–26 labels (`data_pipeline/gen_pipeline.py:111-114, 212`). JevBench public hard has at most 6 labels.

**Long states.**
- *Training:* `MAXLEN=36000` tokens for the 4B and 14000 for the 27B (`scripts/train_final.sh:19-22`). A prompt over MAXLEN is cut up to 4 times, each time keeping the first 70% and the last 30% of the state characters around a `"\n[...]\n"` marker (`training/train_dec.py:221-234`).
- *Inference:* the limits differ by path.
  - PyTorch `LetterAdapter` refuses prompts over `max_tokens` (12,000 by default, `eikos/letter_adapter.py:49, 209-210`; `run_jb.py` keeps that default).
  - `serve.py` allows 16,000 (`eikos/serve.py:210`), and `serve_vllm.sh` serves with `--max-model-len 16384` (`scripts/serve_vllm.sh:13-15`).
  - The suite evaluation uses 40,960 (`evaluation/eval_vllm_suite.py:20`). The 64k probe uses `max_model_len=131072` (`evaluation/probe_long_ctx.py:30`).
- *Prefix cache:* vLLM runs with `--enable-prefix-caching --mamba-cache-mode all`, which caches the Gated DeltaNet recurrent states at block boundaries as well as the attention KV (`scripts/serve_vllm.sh:14`). There is also a PyTorch path that runs the common prefix once and replicates the KV and conv/recurrent caches over N questions (`eikos/letter_adapter.py:288-356`), and an MLX equivalent (`eikos/mlx_decide.py`).
- *64k:* the "64k" claim comes from `evaluation/probe_long_ctx.py`: 80 JevBench public original+hard items, with the state hidden in neutral filler at L ∈ {4k, 16k, 64k} and depths 0.1/0.5/0.9 (`:35-50`). It is beyond the 32k trained range. The card says so ("accuracy beyond the trained range degrades gradually").

## 2. Loss

**Main loss: soft cross-entropy over the letter logits at the answer position** (`training/train_dec.py:497-502`):

```python
lp = F.log_softmax(logits[i, LET_IDS[:n]], -1)
loss = loss - (torch.tensor(tgt, device=lp.device) * lp).sum()
...
return loss / len(encs)          # mean over the micro-batch
```

That is, `L_ce = mean_i [ -Σ_k t_ik · log softmax(z_i[letters[:n_i]])_k ]`. The softmax runs over only the item's own letters. There is no temperature in training, and no hard-label term (`ONEHOT_MIX` defaults to 0 and is not set in `train_final.sh`, `training/train_dec.py:44, 110-112`). The logits are cast to fp32 (`:265`).

**How `target_probs` are built** (`training/train_dec.py:102-113`): `tgt = gold_probs or teacher_probs`, clipped at ≥0 and renormalised. The values come from four sources:
- *Generated items (blind teacher labelling), `data_pipeline/gen_pipeline.py`:*
  1. The writer (GLM-5.3-Flash with reasoning effort "high" for standard items and default=max for hard ones, or Qwen3.8-27B) writes items with `expected`, `rationale`, `surface_answer`, and `gold_probs` for the probability family (`:136-162`).
  2. `validate` checks the schema (`:199-237`).
  3. The teacher, GLM-5.3-Flash at maximum effort, labels each item through **JevBench's own `openai_compat` adapter**, the same prompt the leaderboard uses. It returns *verbalized* JSON probabilities (the adapter's own docstring says "This is NOT logprobs"), with `max_tokens=24000`, and it never sees the gold (`:316-331`).
  4. The probabilities are renormalised. `agree = argmax == writer's expected` (`:342-347`), and only `agree` rows are trained on (`training/train_dec.py:96`).
  5. The target is the teacher distribution as returned: no sharpening, no mixing. On the released teacher rows, `target_probs == teacher_probs` for all 9,264. The mean max-prob is 0.928, the median 0.97, 25% of rows are ≥0.99 and 10% are ≤0.8 (mean entropy 0.27 nats).
- *Programmatic items:* one-hot plus light smoothing via `soft(gold, labels, p=0.95)`, which spreads the rest uniformly (`data_pipeline/prog_fin.py:25-27`, `prog_rules.py:564-566`, `prog_trade.py:30`, `prog_temporal.py:22`). Judge items (GSM8K, TAT-QA) use 0.97/0.03 (`data_pipeline/prog_judge.py:87,97`; `prog_finjudge.py:95`).
- *Probability family:* exact distributions (`data_pipeline/prog_prob.py:188-190`), or the writer's `gold_probs` (`writer_exact_probs`).
- *FinEntity:* 0.85 on the human label and 0.075 on each other option (`data_pipeline/prog_finentity.py:22`).

**Label smoothing.** Nothing is applied in the trainer. It is baked into the data (0.95 / 0.97 / 0.85).

**Option permutation** (`training/train_dec.py:139-147`, `PERM_P=1.0`):

```python
if train and t in ("choice", "noul") and rng.random() < PERM_P:
    opts = opts[:]; rng.shuffle(opts)
tgt = [r["_tgt"].get(lab, 0.0) for lab, _ in opts]
```

- It applies **per example, each time the example is drawn**, with a fresh uniform permutation from the run's `random.Random(SEED=7)`. With 1 epoch that means one permutation per item.
- Both **choice and noul** are shuffled, so yes/no swap between A and B. `score` keeps its natural order.
- The target is carried along with the labels. Logits are **not** averaged across permutations, and there is no multi-permutation consistency term.
- Dev evaluation uses the canonical order (`train=False`).
- PT↔EN view pairs share the same permutation, so that A means the same option in both views (`training/train_dec.py:431-443`, `encode_ex` with `opts`).

**Auxiliary rationale loss (`RATIONALE_W=0.3`)** (`training/train_dec.py:237-275`, `scripts/train_final.sh:31`):
- *Sequence:* `[prompt] + tok(LETTERS[g] + "\nRationale: " + rationale)[:RAT_MAX] + [EOS]`, where `g` is the argmax of the *permuted* target, so the gold letter is teacher-forced first (`:242-244`). `RAT_MAX = 192` tokens by default and is not overridden (`:57`).
- *Masking:* the letter token is excluded. Position `p0+t` predicts `rt[t+1]` for t ≥ 0 (`:268-270`), so the soft CE alone covers the answer token. No loss falls on prompt tokens.
- *Reduction:* `F.cross_entropy` averages over all rationale tokens in the micro-batch, so longer rationales weigh more (`:274`). Total: `soft_ce + 0.3 · rat_loss` (`:637`, and `jepa_step` `:449-450`).
- *Which rows:* only rows with a rationale string of at least 20 characters (`:239`). That covers 7,781 teacher rows, 5,717 prog_rules/rulebook rows (template rationales) and 204 writer-probability rows. The long-context rows have the rationale removed (`data_pipeline/make_long.py:68`). The released rationales average 409 characters (median 351, p95 811), so about 100 tokens on average and about 200 at p95. The 192-token cap rarely truncates.
- Rationales never enter inference.

**PT↔EN view consistency (4B only, `JEPA_VKL_W=0.5`, `VIEW_AUG=1`, `JEPA_V_P=1.0`)** (`scripts/train_final.sh:19-21`):
- Every training item that has a view (1,637 in the 4B run; 1,619 released) enters the same micro-batch together with its translated view, in the same option order (`training/train_dec.py:439-443`).
- The view also gets the soft CE toward the same target (augmentation).
- The loss adds `0.5 · mean_pairs ½[KL(p_i‖p_j) + KL(p_j‖p_i)]` over the letter distributions (`:370-380, 453-454`).
- The views were translated by Qwen3.8-27B with thinking off. They keep keys and labels, and ≥90% of the 3+-digit numbers must survive (`data_pipeline/make_views.py:23-64`).

**Optional JEPA losses** (recipe `4b-E` only, `scripts/train_final.sh:20-21`: `JEPA_E_W=JEPA_V_W=JEPA_R_W=0.2`, `JEPA_DROP=0.5`). They are trained through a small fp32 `JEPAHead` that is discarded for the letter readout (`training/train_dec.py:176-192`):
- **E (energy):** `softmax(cos(pred_ans(h_answer), opt(mean-pooled option-description span)) / 0.05)`, trained with soft CE to the same target (`:355-367`). It requires the semif prompt (`:198`).
- **V (view invariance):** the hidden state at the answer position of layer `int(0.75·L)` predicts the other view's hidden state, with cosine loss and a stop-gradient target (`:383-390`; the layer is chosen at `:202-210`).
- **R (latent rationale):** the same mid-layer answer state predicts the model's own no-grad last-token embedding of `"Rationale: …\nAnswer: <gold>: <text>"` (`JEPA_R_TARGET=self`, `:393-424`).
- V and R are skipped in 50% of micro-batches ("loss dropout"); E is not (`:448, 455-458`).

**What shipped:**
- **Released Eikos-4B = weight average ("soup") of the merged `4b-B` and `4b-E` models** (`scripts/train_final.sh:10`; `training/make_soup.py:20-26`). So JEPA contributes half of the released 4B.
- Eikos-27B uses no views and no JEPA (27B card).
- The card says the soup "won on our internal dev set and on the held-out family, topic and language". No per-component ablation is published.

## 3. Hyperparameters

| Item | Value (citation) |
|---|---|
| Start model | `Qwen/Qwen3.5-4B`, the post-trained hybrid-thinking checkpoint (`scripts/train_final.sh:19-20`; README step 2 downloads `Qwen/Qwen3.5-4B`; 4B card YAML). **Not** `-Base`: the norm tensors are identical to Qwen3.5-4B and differ from Base, checked by HTTP range reads of the safetensors. 27B: `Qwen/Qwen3.8-27B` (`:22`). Loaded as `AutoModelForCausalLM`, i.e. text-only `Qwen3_5ForCausalLM`; the vision and MTP weights are copied back from the base checkpoint at export (`training/fix_export.py:1-3,14-31`). |
| LoRA | r=64, α=2r=128, dropout 0.05, bias none, `target_modules="all-linear"` (`training/train_dec.py:167-168`, `RANK_LORA=64` in `train_final.sh:30`). PEFT's `all-linear` covers every `nn.Linear` except the output head. On this checkpoint that means attention `q/k/v/o_proj`, MLP `gate/up/down_proj`, **and the Gated DeltaNet projections `linear_attn.in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`**. It does not touch `linear_attn.conv1d`, `A_log`, `dt_bias`, the norms or the embeddings (module names are from the Eikos-4B `model.safetensors.index.json`). |
| Optimizer | AdamW, lr 1e-4, weight decay 0 for LoRA, betas (0.9, 0.99) (`training/train_dec.py:568`); gradient clip 1.0 (`:616, 644`) |
| Schedule | linear warmup `max(10, 0.03·steps)` (21 steps for the 4B), then cosine to 0 at `steps_total` (`:569-572`) |
| Epochs | 1 (`EPOCHS=1`, `train_final.sh:30`). The trainer default is 2 (`:37`). Effective 4B coverage is about 0.8 epoch (see summary point 3). |
| Batch | 4B: `BS=8, GRAD=4` → nominal 32 items per optimizer step, plus view copies. `TOK_BUDGET=40000` estimated tokens per micro-batch, counted as count × longest item's characters / 3.5 (`:47, 595-602`). Batches are length-bucketed inside blocks of 256 shuffled items (`:587-591`). 27B: `BS=2, GRAD=16`. |
| Steps | 4B: `ceil(23,168/8)/4 = 724` optimizer steps. 27B: 694. |
| Seq length | MAXLEN 36,000 (4B) / 14,000 (27B), with head/tail state truncation (section 1) |
| Precision | Weights loaded bf16 (`:157`). PEFT's default adapter dtype applies; the loss and letter logits are fp32; no AMP scaler. The JEPA head is fp32. |
| Memory | Gradient checkpointing with `use_reentrant=False` (`:161, 165`); `enable_input_require_grads`. `flash-linear-attention>=0.5.2` provides the DeltaNet kernels (`pyproject.toml:22`). |
| Dev / selection | Hash split, 8% of generated items and 4% of the rest (`:120-123`, `DEV_FRAC_GEN=0.08`). Dev is evaluated every 400 steps, but **no checkpoint selection**: only the final adapter is saved (`:667-670`). |
| Held out | `HOLDOUT_TOPICS="healthcare administration"`, `HOLDOUT_FAMILIES=tradeoff`, `HOLDOUT_LANGS=Spanish` (`train_final.sh:29`). Note that `tradeoff` is one of the JevBench hard families (6 of the 111 public hard items). |
| Calibration | `calib.json` = `{"mode": "T1", "t_global": 1.0}` is written unconditionally (`train_final.sh:33-34`); it is the file in the HF repo. `calib_fit.py` exists but is not used for release (section 5). |
| Hardware and wall time | **Not stated** anywhere in the code or the cards. The trainer is single-GPU (`device_map={"": DEVICE}`, `:157`). The only GPU named is an RTX PRO 6000, in the card's serving-throughput section. |

## 4. Data (`eikos-decisions`)

**Released row counts:** core train 21,150, validation 1,221, heldout 1,191; long_context train 1,517, validation 59; views 1,619. The 4B trained on 23,168 rows and the 27B on 22,216 (`release_tools/build_release_data.py:37`).

Released train (core + long = 22,667 rows):

| Breakdown | Rows |
|---|---|
| `used_in` | 21,990 both models · 669 4B-only (the dossiers of 12k–32k tokens) · 8 27B-only |
| Label source | program 10,678 · teacher 9,264 · human 1,911 · exact_probability 576 · writer_exact_probs 238 |
| Item writer | program 11,254 · Qwen3.8-27B 7,869 · FinEntity 1,911 · GLM-5.3-Flash 1,633 |
| Language | English 19,524 · Brazilian Portuguese 3,143 (all Spanish is held out) |
| Difficulty | hard 15,373 · standard 7,294 |
| Question type | choice 10,575 · noul 10,375 · score 1,717 |
| Options per question | 2: 10,422 · 3–6: about 12,000 · 7–26: 293 |

**Families:**
- *Programmatic:* judge 2,082 · rules_book 1,899 · rules_permit 1,571 · trade_exact 1,439 · finance_exact 960 · rules_route 781 · rules_tier 749 · rules_score 717 · temporal (program part) 480 · probability 814.
- *Teacher-labelled* (the generator families in `data_pipeline/gen_pipeline.py:45-115`): long_policy 893 · multi_hop 799 · judge_hard 689 · policy 656 · temporal_numeric 653 · trap 636 · adversarial 562 · extraction 527 · news_signal 457 · routing_hard 456 · tool_selection 443 · intent 442 · fact 418 · ordinal 401 · ambiguous 338 · adequacy 326 · routing 312 · large_choice 256.
- *Human:* entity_sentiment (FinEntity) 1,911.
- The generated family names and descriptions **mirror JevBench hard's families one for one**: long_policy, multi_hop, judge_hard, temporal_numeric, probability, trap, ambiguous, tradeoff, adversarial, routing_hard. Those are exactly the families in `datasets/public/hard.jsonl`, and the "hard" weights are the largest (`:47-83`).
- Topics: 19, of which 8 are finance (`:116-128`). Formats: 10 (`:129-131`). Language weights: English 75 / Portuguese 20 / Spanish 5 (`:132`). Length bands: hard items 400–2,500 tokens, standard items 80–600 (`:133-134`). The spec also imposes neutral labels, balanced yes/no, and a default option that is correct in about 1 of 4 items (`:149-158`).

**Programmatic generators (exact answers):**
- `prog_prob.py`: hypergeometric inspection, reliability, raffle, Bayes, dice and count-choice, with exact `Fraction` distributions (`:32-35, 37-190`). Its instruction text is **verbatim JevBench-hard phrasing**: "Give probabilities that reflect the evidence in the state." (`:58,89,107,125,150,172`). Five of the ten probability items in JevBench public hard use exactly that sentence, and the others use close variants.
- `prog_temporal.py`: filing deadlines in business days, SLA in business hours, renewal notice periods, prorated refunds, time zones, late fees.
- `prog_fin.py`: amortization, FX, DTI, three-way match, structuring, liquidation, pre-trade checks, P&L, settlement, option expiry.
- `prog_trade.py`: 17 trading and trade-finance templates, yes/no balanced by construction, English and Portuguese.
- `prog_rules.py`: compositional permit / route / tier / score rules and `--book` rulebooks, with the gold answer computed by executing the rule and a template rationale.
- `prog_judge.py` + `fix_judge.py`: GSM8K **train**. Candidate solutions are sampled from Qwen3.5-0.8B at T=0.8 and graded exactly; clean reference answers are added; decimal answers are dropped (a style shortcut); yes/no is balanced within each format (`prog_judge.py:47-99`; `fix_judge.py:10-26`).
- `prog_finjudge.py`: TAT-QA with realistic numeric errors. FinQA-derived rows are excluded because FinQA is used for evaluation (`data_pipeline/snapshot_final.sh:29-32`).
- `prog_finentity.py`: FinEntity entity sentiment.
- **Per-source quotas** relative to the number of agreed generated items (`snapshot_final.sh:15-17`): prob 0.06×, temporal 0.05×, judge 0.12×, fin 0.10×, finjudge 0.10×, trade 0.15×, rules 0.4×, rulebook 0.2×, long 0.2×, FinEntity 2,100. The result is about 47% programmatic.

**Long-context dossiers** (`data_pipeline/make_long.py`):
- A training item's state becomes "Document k" among the states of other training items from different topics, plus neutral filler, up to a target length L ∈ {6k, 12k, 20k, 32k} tokens (at 3.5 characters per token).
- The question is rewritten as `"Regarding Document k: …"`, the label is kept and the rationale is dropped (`:47-69`).
- Distractors and targets come only from non-dev, non-held-out rows (`:39-41`).
- Released distribution: 464 / 384 / 355 / 314 rows at 6k / 12k / 20k / 32k tokens. The 27B used only the dossiers of 12k tokens or less.

**8-gram decontamination** (`data_pipeline/dedup_check.py`):
- Word 8-grams of lower-cased `\w+` tokens over `state + instructions` (`:15-22`), against the 231 JevBench public items in easy, original and hard (`:26-29`).
- The containment for an item is `max over public items of shared/|item grams|`; above **0.15** the item is excluded (`:11-12, 43-52`).
- The release adds a two-direction check against every reported suite, flagging above 0.15 with at least 8 shared grams and ignoring suite boilerplate, plus exact-duplicate checks for the templated suites (`release_tools/check_overlap.py:1-97`). This is looser than our lint (≥2 shared 8-grams → drop). That is why `filter_eikos.py` drops all 576 `prog_prob` rows and 526 `rules_permit` rows (`data/external/eikos-decisions/report.json`: `jevbench_8gram: 1102`).

**The three probability fields:**
- `target_probs` is the training target, aligned with `options` in canonical order (noul is always `yes, no`).
- `teacher_probs` is GLM's raw blind distribution, present on generated rows only; it equals `target_probs` for `label_source=teacher`.
- `expected` is the writer's or program's gold. It equals the argmax of `target_probs` in 22,664 of 22,667 train rows (`release_tools/build_release_data.py:158-181`).

**The roughly 2% left out** (dataset card): 509 rows the models *did* train on:
- 52 rows with credential-shaped strings;
- 444 rows with personal-data-shaped values;
- 15 rows sharing the AAA arbitration clause with a LegalBench item;
- 3 rows with known label errors (TAT-QA rounding; `release_tools/known_label_errors.json`);
- 27 duplicate FinEntity copies on top of the 509.

The filter removed 21% of the long rows.

**Licences:**
- The dataset is CC BY 4.0, except the rows whose `upstream` names another licence: FinEntity ODC-BY 1.0 (1,911 rows), TAT-QA CC BY 4.0 (937), GSM8K train MIT (1,145).
- Model outputs: GLM-5.3-Flash (MIT), Qwen3.8-27B (Apache-2.0), Qwen3.5-0.8B (Apache-2.0).

**GLM-labelled rows.**
- *Which rows:* every `label_source ∈ {teacher, writer_exact_probs}` row, i.e. all generated items. The pipeline sends every generated item to the GLM teacher (`data_pipeline/gen_pipeline.py:26-33, 316-347`), and GLM's argmax had to match the writer. That is 9,502 core+long train rows and 7,985 core train rows in `filter_eikos`'s count.
- *Open weights or API:* GLM-5.3-Flash is **open-weight** (`zai-org/GLM-5.3-Flash`, `license:mit`, about 321B parameters on HF). The pipeline reaches it through "any OpenAI-compatible endpoint" (`.env.example:11-14`), and the card says "run by the authors on their own infrastructure".

## 5. Evaluation harness

**The 7 suites / 7,371 items** (`evaluation/eval_vllm_suite.py:120-121`; builders in `scripts/build_eval_suites.sh`):

| Suite | Contents |
|---|---|
| `jb` | JevBench public easy / original / hard (48 / 72 / 111 = 231) |
| `db` | DecisionBench medium + hard |
| `gen` | 9 human-labelled tasks, 2,696 items (MMLU-Pro, RewardBench, ASSIN2, BoolQ, Banking77, CLINC150, LegalBench, XNLI-es, GSM8K test) |
| `fin` | CUAD, financial sentiment, FinQA-judge |
| `fin2` | WCB stance, FinDVer |
| `trade` | 858 items |
| `rules` | 1,400 items |

7,371 − 231 = **7,140**, the pooled confidence set ("6 suites", excluding JevBench).

**How JevBench is run.** There are two paths.
- *Reference path:* `evaluation/run_jb.py`. It uses the **official jevbench `Runner`** with Eikos's `LetterAdapter` in PyTorch (`:20-23, 67, 83-85`).
  - It is serial, one item at a time, through `dist_any`; `sym` is off; the temperature comes from `--temp`/`--calib` (T = 1).
  - Its outputs `final_4b_soup_{tier}.jsonl` feed `compare_models.py` (`:66`). This produces the card's *Evaluation* table: 4B original / hard **91.7 / 72.1**, 27B **100 / 82.9**.
  - Hard ECE is `ece_top` over 10 equal bins on the 111 hard items (`:30-40, 96-111`). The same script computes a public-only proxy of the leaderboard score.
- *Release-builds path:* `eval_vllm_suite.py … ALL`. It is offline vLLM over the suite version of the public items (`evaluation/jb_public_to_suite.py`).
  - All prompts are batched with `max_num_seqs=4`, `enable_prefix_caching=True, mamba_cache_mode="all"` and `processed_logprobs` with `allowed_token_ids` (`:24-35, 44-47, 66`).
  - The card's release table: 4B **91.7 / 73.9** (82 of 111 against 80 of 111 serially); 27B 100 / 82.0.
  - The card says batched vLLM "is not bit-for-bit deterministic (1–2 items per suite can change)". An optional `WARM=1` mode serialises the requests of each shared prefix to avoid a hybrid-cache race (`:40-64`); it is off by default.
- *Neither path* averages over permutations or rotations. Option order is the item's label order (`options_of(question, labels)`), with yes first for noul. The calibration is T = 1 (`:71`).

**Jev and Laya rows.**
- **Jev, JevBench:** from the official leaderboard file `results/v1.2/jevbench-v1.2-per-task.json`, system `jev-1.13.0`, `public_tasks` correctness (`evaluation/compare_models.py:89-94`). This is where 98.6 / 73.0 comes from. Jev was not re-run.
- **Jev, the other 6 suites:** `evaluation/eval_jev_suite.py`. It sends one question per request, `model: "typesafe-ai/jev"` "via Vercel AI Gateway", noul mapped to `boolean`, `probability` taken as P(yes), 4 threads, and exponential-backoff retries on 429/5xx (`:24-54`). "Evaluation only; distilling from Jev is prohibited by its terms" (`:1-3`).
- **Laya:** the official `laya` 0.3.7 package, in router mode and in typed-decision mode. Choices too large for its head go through the package's own shortlist (k = 20 → 10 → 5), failures count as errors, and a `fits` flag records context overflow (`evaluation/eval_laya_suite.py:1-61, 84`). Its JevBench results are compared with the official leaderboard as a sanity check. Laya is also shown "recalibrated": one temperature per (type, number of options), cross-fitted on two hash halves (`compare_models.py` `laya_recal`).

**"Error at ≥90% confidence"** (`evaluation/compare_models.py:170-205`; `release_table.py:57-58, 70`):
- Pool the per-item `(conf = max option probability, ok)` over the items common to all systems in the 6 non-JevBench suites.
- `decide = #{conf ≥ 0.90}/n` and `error = 1 − acc(conf ≥ 0.90)`.
- Jev's confidence here is also the max probability, not Jev's `(n·p−1)/(n−1)` confidence.

**ECE:**
- 10 equal-width bins on top-1 confidence, `b/10 < c ≤ (b+1)/10` (with c = 0 in bin 0), `Σ (n_b/n)·|acc_b − mean conf_b|` (`release_table.py:41-47`; `compare_models.py:194-198`; the same in the trainer's dev ECE, `training/train_dec.py:526-532`).
- The headline 0.033 (4B) is pooled over the 6 suites. The JevBench-hard ECE of 0.049 uses only 111 items, so it is noisy (a few hundredths).

**Calibration choice.** `training/calib_fit.py` fits T(x) = exp(b + w·[log n_tok, log n_opts, is_noul, is_score]) and a global T on the dev logits, only on generated rows (`CALIB_SRC=gen`) and with hard items weighted 3× (`:15-25, 35`). The released default is nevertheless T = 1:
- every fitted T came out below 1 and made hard items over-confident;
- T(x) worsened out-of-distribution ECE from 0.055 to 0.110 (`:84-89`; 4B card, "Calibration").

## 6. Serving

- **vLLM ≥ 0.30.0 is enforced** (`scripts/serve_vllm.sh:6-12`). The stated reason: older builds return wrong answers when several long requests are batched on the hybrid Gated DeltaNet model, with drops of 3–6 points on shared-document items in 0.11.
- **Flags:** `--enable-prefix-caching --mamba-cache-mode all --logprobs-mode processed_logprobs --max-logprobs 32 --max-model-len 16384 --served-model-name decider` (`:13-15`). `processed_logprobs` means the log-probabilities *after* the `allowed_token_ids` mask, i.e. already renormalised over the allowed letters.
- **`/v1/systemone`** (`eikos/serve.py:183-196` → `Decider.decide_all`, `:49-78`):
  - Each question's options are built with `options_of`. With `--vllm-url`, all questions of the request go into **one** `/v1/completions` call, as a list of prompts that each repeat system + state + question (`eikos/letter_adapter.py:240-255`).
  - `allowed_token_ids = letters[:n_max]` (the largest option count in the request) and `logprobs = n_max` (`:82-101`).
  - Per question, the adapter takes the logprobs of its own `letters[:n]`, uses -1e9 for any that are missing, and applies `softmax(lp / T)` (`:254-255, 157-160`), which renormalises over that question's letters.
  - vLLM's prefix cache processes the shared state once. Local PyTorch instead uses `dist_many_cached`, the explicit prefix fork.
  - Response formats are in `:80-89`: noul returns `noul`/`probability` = P(yes), `value = P ≥ 0.5` and `confidence = max(P, 1−P)`. That confidence is *not* Jev's 0-to-1 concentration.
  - `--sym` optionally averages the reversed option order into the same batch (`:63-64, 73-75`). It is off in all reported numbers.
- **Sessions** (`eikos/serve.py:149-178`): the state is held in memory. `append` concatenates text with no separator (`:165-170`). `…/systemone` runs `decide_all` over the full current state. The cache reuse comes entirely from vLLM prefix caching: the semif prompt embeds the state as the first JSON field, so an appended state keeps a byte-identical prefix. There is no eviction beyond the stored timestamp.

**Differences from a serial `jevbench --adapter typesafe` run** (the way we run everything):
1. Their JevBench rows for Jev and Laya are official v1.2 leaderboard results, not a same-day run on the same harness. Our JevK5 73.9 and Jev numbers come from our own runs, possibly on another JevBench version.
2. The two Eikos harnesses differ by 2 hard items (72.1 serial against 73.9 batched vLLM). A serial run of `serve.py` through our typesafe adapter should land near 72.1 ±1 item.
3. Limits:
   - `serve_vllm.sh` has a 16k context limit;
   - `LetterAdapter` defaults to 12k tokens;
   - the 26-letter tournament applies above 26 options;
   - `null` or structured criteria are rendered with `str()`.

   None of these bites on JevBench public, which has at most 6 options and short states.
4. T = 1, and no rotations or `sym`. We report "raw / rot4 / calibrated" variants. The comparable row is our **raw, T = 1**.
5. Their headline ECE and ≥0.90 error are pooled over their own 6 suites; about 2,258 of the items (trade and rules) come from their own training generators. Only the 111-item JevBench-hard ECE compares with ours.

## 7. Checklist against `docs/phase-2c-plan.md`

**1. Soft targets from a 35B-A3B distribution.**
- *What Eikos does:* one blind call to a strong reasoning teacher (GLM-5.3-Flash, maximum effort) through the JevBench `openai_compat` prompt, giving verbalized JSON probabilities. It keeps a row only if `argmax == writer gold`. The target is the teacher distribution unchanged, trained with pure soft CE. Program rows are one-hot at 0.95 (0.97 for judge), probability rows use exact distributions, human rows 0.85.
- *Copy exactly:*
  - the agreement filter on argmax;
  - pure soft CE (`--soft-weight 1.0`), with no temperature or sharpening on the targets;
  - the smoothing constants for exact rows (0.95 / 0.97);
  - exact distributions for probability items.
- *Do differently:*
  - Our targets include `unknown`. Store the teacher's `unknown` mass, but for rows whose intended answer is a real option, check how much mass the 35B puts on `unknown`. If it routinely gives, say, 0.1–0.2, it will teach the model to hold back probability on JevBench, where `unknown` is never gold. Consider capping it or dropping it on non-`unknown` rows.
  - Our readout is a 255-code head at a decision position (`scripts/torch_decision.py:20-50`), not letters. The soft CE applies unchanged over candidate codes.
  - A 35B-A3B with thinking **on** (97.3 on public hard in our frozen check) is the right teacher. With thinking off it is at our 4B's level and would not be worth using.

**2. Train-time permutation.**
- *What Eikos does:* a fresh uniform shuffle every time an item is drawn, for choice **and noul**. Score keeps its order. No logit averaging.
- *Copy exactly:* per-draw shuffling of choice questions. Ours, keyed on seed+epoch+id, is equivalent.
- *Do differently:* our boolean and ordinal fields stay fixed, which is fine if the served boolean order is fixed. Eikos shuffles noul because a letter position carries a prior. If our boolean codes are position-based rather than label-based, add yes/no shuffling. It is cheap.
- With permutation in training, drop rotation averaging at inference, as the plan already says.

**3. Rationale loss 0.3.**
- *What Eikos does:* weight 0.3, as the token-mean CE over at most 192 tokens of `"<gold letter>\nRationale: <text>" + EOS`, placed after the answer position with the gold letter teacher-forced. Only rows with a rationale of at least 20 characters; none on long rows.
- *Copy exactly:* the weight 0.3, and no rationale on dev rows.
- *Do differently:*
  - **Raise our cap from 64 to about 192 tokens.** Eikos rationales are about 100 tokens on average and about 200 at p95, so 64 truncates most of them. Our 35B rationales are specified as at most 2 sentences, so about 64 may be enough for them, but not for any Eikos rationales we import.
  - Ours is not conditioned on the answer: `" Because: …"` directly after the decision position, with the first token masked. That is a defensible difference because it forces the prompt states to carry the reasoning. Keep it, but note it is not Eikos's setup.

**4. Judge and programmatic families.**
- *What Eikos does:*
  - Judge: 2,082 exact rows (GSM8K-train solutions sampled from a 0.8B model and graded exactly, with the style shortcuts removed by `fix_judge.py`; TAT-QA with numeric errors), plus 689 teacher `judge_hard` rows.
  - About 11k programmatic rows under per-source quotas, about 47% of the mixture.
  - The generated families mirror JevBench hard's family names.
- *Copy exactly:*
  - `fix_judge.py`'s de-shortcutting: balance yes/no **within** each format, and remove surface cues such as decimal answers that are almost always wrong.
  - Quota-capping of templated sources.
- *Do differently:*
  - Write our own probability instructions. Eikos copies the JevBench sentence, and our lint then drops the whole family.
  - Keep image-grounded variants, where the evidence is in an image, for imajev's own benchmark.

**5. The Eikos slice.** Under `strict` it is 10,152 rows, all programmatic: rules 5,191 (51%), judge 2,082, trade 1,439, finance 960, temporal 480, **no probability rows** because of our lint, and 62% noul.
- The rows in JevBench-hard families that Eikos trained on (4,236 teacher rows) are all excluded, because GLM labelled them.
- **GLM-5.3-Flash is an MIT open-weight model**, so the "no paid-API outputs" rule does not obviously exclude it. The `open-weights` policy already exists in `filter_eikos.py` and would give 18,120 core-train rows, plus 1,423 long rows and 1,619 views.
- *This is an owner decision.* If the owner stays with `strict`, use the slice as a judge and rules supplement, and cap it so that the rules families do not dominate.

**Eikos details we had missed that matter for JevBench hard and calibration:**
- **The start point and adapter capacity.**
  - Eikos: one fresh LoRA r64/α128 on **all linear layers including the DeltaNet projections**, lr 1e-4, cosine, about 724 steps of about 26–32 items, **from the clean post-trained Qwen3.5-4B**.
  - Ours: r16/α32 delta-tuned at lr 2e-5 for 2 epochs on top of adapters that went through v2.1 (500k easy rows, the forgetting we measured) and then phase-2 and phase-2b.
  - With comparable data volume (about 23k rows against our about 25k), Eikos reaches 72–74 hard and we reach 67.6. The start point and the update size are the biggest untested differences.
- **Calibration comes from the targets, not from a temperature.** Eikos ships T = 1 and reports ECE 0.033 pooled and 0.049 on hard. Temperatures fitted on easy dev data went below 1 and hurt the hard items. Our raw 4B hard ECE is 0.16–0.21, and our T of about 2.3 is a patch for one-hot training. After soft targets, refit T only on held-out-domain **teacher** rows, weighting hard rows, and expect T ≈ 1.
- **No checkpoint selection.** Eikos keeps the final adapter plus a 2-model soup. We select on dev loss or dev2. Keep ours, but log the last step as well.
- **Difficulty labels** are used only for evaluation and for calibration weighting (hard 3×), not for training weights.
- **Multilingual** (Portuguese views and training, Spanish held out) and **long dossiers** (6–32k tokens) matter for their product, not for JevBench public: the hard items are at most about 2.5k tokens and English. Skip both for phase 2c. Our 4,096-token limit ("refuse, never truncate") is fine for JevBench.
- **Eikos-4B saw only about 80% of its data**, because of the `steps_total` and `TOK_BUDGET` interaction. A faithful "copy" does not need to reproduce that.

## Prioritised changes to our recipe

| # | Change | Expected impact | Cost |
|---|---|---|---|
| 1 | **Add a fresh-start lane** next to the delta lane: a new LoRA r64 / α128 / dropout 0.05 on all language-layer linear modules (DeltaNet `in_proj_qkv/z/a/b`, `out_proj`, attention, MLP) of `Qwen/Qwen3.5-4B`. One epoch at lr 1e-4, cosine with 3% warmup, AdamW with wd 0 and betas (0.9, 0.99), clip 1.0, about 32 items per step. Data: the phase-2c text mixture, pure soft CE and rationale 0.3, **plus an image replay slice** (about 30–40% of steps, from v1.1 and v2.1 decision data) so ImajevBench does not regress. Gate on JevBench raw hard, hard ECE and ImajevBench against the delta lane. | This is the largest open lever. It tests whether our 5-point gap to Eikos comes from the start point and the update size rather than the data. Plausibly +3–6 hard at T = 1. The risk is image regression, which the replay slice and the gate cover. | About 1–2 H100-hours for the 4B on about 25k text rows plus replay; one extra lane on the planned pod |
| 2 | **Soft targets everywhere, as Eikos does:** pure soft CE; 35B-A3B **thinking-on** verbalized distribution; agreement on argmax; no sharpening; 0.95 / 0.97 smoothing on exact rows; exact distributions on probability rows. Audit the teacher's `unknown` mass on rows whose gold is a real option, and cap or strip it if it averages above about 0.05. | Calibration at T ≈ 1 (hard ECE toward about 0.05, from 0.16–0.21 raw). Modest accuracy gain. | Already planned: the relabelling pass (about 1.5 h on 6 GPUs) plus a small audit script |
| 3 | **Owner decision on the GLM rows.** GLM-5.3-Flash has MIT open weights and the authors ran it themselves. Switching `filter_eikos.py` to `--teacher-policy open-weights` adds about 8k teacher-soft rows, 4,236 of them in the JevBench-hard families (long_policy, multi_hop, judge_hard, trap, temporal, ambiguous, adversarial, routing_hard). | Potentially large on hard: these are the family-matched rows behind Eikos's score. | Zero compute; a policy call and a note in the data card |
| 4 | Raise `--rationale-max-tokens` from 64 to 192 (or cap at our own rationale length plus a margin). Keep the weight at 0.3. | Small to moderate. The rationale signal is not truncated. | About +50–100% tokens per example on rationale rows |
| 5 | Write our own `probability_exact` generator phrasing (not JevBench's sentence, so our lint keeps it) and put more weight on it. Ten of the 111 public hard items are probability items with `gold_probs`; Eikos's public-proxy calibration score scores them by total variation from that gold distribution (`evaluation/run_jb.py:101-111`). | Hard accuracy and the calibration axis | CPU only; already on the day-1 list |
| 6 | Fit the temperature only on held-out-domain teacher rows, with hard rows weighted 3×. If the fitted T is below 1, ship T = 1, as Eikos does. Never include programmatic rows. | Avoids the over-confidence trap Eikos documents | Minutes |
| 7 | Cap templated sources as Eikos does, relative to the teacher-row count (rules ≤ 0.4×, rulebook ≤ 0.2×, trade ≤ 0.15×, judge ≤ 0.12×, finance and finjudge ≤ 0.10×). Keep the programmatic share at 50% or less. | Prevents the strict slice (51% rules) from skewing the model | Manifest change only |
| 8 | Shuffle the boolean yes/no order in training **if** our boolean codes are positional. | Small (removes a position prior) | Trivial |
| 9 | Skip for 2c: PT↔EN views, JEPA losses, long dossiers, the soup. The soup is worth reconsidering only if we end up with two comparable lanes, e.g. delta and fresh: average the merged weights and gate on dev. | Not relevant to JevBench public, or unproven | None |

For comparisons, use Eikos-4B **serial, T = 1** at 72.1 hard (91.7 original), not the 73.9 from the batched vLLM path. Report ours in the same form: raw, T = 1, no rotations.
