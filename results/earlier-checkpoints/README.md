# decision-p2: hard typed-question delta fine-tunes (phase 2)

Pod: 4×H200 SXM `vo3jgs1nlchuyx` (terminated 2026-09-24 ~02:00 IST after pulling `pod/p2-train-results.tgz`).
Data: `data/manifests/decision-p2.jsonl` (17,898 records after contract validation; teacher half written by Qwen3.6-27B,
answered by Qwen3.6-27B (thinking) + gpt-oss-20b with agreement required; human half from licensed sources).
Training: LoRA continued from each model's previous adapter (2B = v2.1 step 1,361; 4B = v2.1 step 1,900; 9B = v1.1 best),
2 epochs, lr 3e-5, selection on `decision-v2-reasoning-dev` mean accuracy. Adapters: `runs/4b-p2-best-step250`,
`runs/9b-p2-best-step300`, `pod/train-out/2b/train/best`.

## Results (best checkpoint unless stated; raw scores, no rotations, no calibration)

| Panel | 2B | 4B | 9B |
|---|---|---|---|
| decision-p2 test (n=2,983) | 66.4 | 77.4 | 79.7 |
| reasoning dev (n=6,240) | 64.5 | 67.8 | 69.2 |
| v1.1 held-out text (n=2,000) | 65.5 | 67.8 | 68.8 |
| irrelevance test (n=2,823) | 67.8 | 79.7 | 83.4 |
| state probe (n=200) | — | 68.5 | 65.5 |
| pairs probe (n=60) | — | 98.3 | 81.7 |
| JevBench hard, best / last | 55.0 / 55.0 | 62.2 / 63.1 | 67.6 / 66.7 |
| JevBench original | 91.7 | 98.6 | 98.6 |
| JevBench easy | 100 | 100 | 100 |
| JevBench hard ECE (best) | 0.234 | 0.224 | 0.207 |
| ImajevBench v2.0-lite test (direct, torch, full rotations) | 68.5 (191/279) | 80.6 (225/279) | 81.4 (227/279) |

ImajevBench tracks — 2B: joint 77/122, text 18/37, visual 96/120. 4B: joint 98/122, text 23/37, visual 104/120.
9B: joint 99/122, text 21/37, visual 107/120. Base models on the same split (leaderboard): Qwen3.5-2B 60.2, 4B 70.6, 9B 76.7.

Before phase 2 (for reference): imajev-2b v2.1 ImajevBench 63.1/63.4, imajev-9b v1.1 81.0; JevBench hard for the v2.1
adapters was in the low 40s (2B) — phase 2 is what moved the hard split.

Notes
- The 2B state and pairs probe panels produced no predictions: the probe images under `data/decision-v2/` had been deleted
  from the H200 pod to free disk before the 2B run (4B/9B ran earlier). The 7×H100 bundle carries the 236 probe images; rerun there.
- 4B ImajevBench failed once on a missing `artifacts/model-qwen4b.json` (now added) and was rerun; the score above is the rerun.
- Unknown-prediction rates are low (0.4–2.3% on answerable panels, ~10% on irrelevance, 13–15% on the state probe); the
  phase-2b pass adds a relaxed-rule unknown slice to lift the unknown share.
- Next: phase 2b on the 7×H100 pod (`scripts/p2/pod_run_p2b_gen_h7.sh`, `cloud/pod_run_p2b_train.sh`) — target 4B ≥ 70,
  9B ≥ 74 on JevBench hard; frozen Qwen3.6-35B-A3B check decides a fourth tier.
