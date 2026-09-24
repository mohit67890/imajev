# imajevBench hidden test: policy and procedure

The public benchmark (dev, calibration and public test) lets anyone reproduce and compare results.
The **hidden test** is a separate set that only the imajevBench team holds. Its images, questions
and answers are never published, so a model cannot be tuned against it.

When someone sends us a model, we score it on the hidden set and publish aggregate numbers only.

## The hidden set

- **Current set:** `data/imajev-bench/private-1/records-audited.jsonl` (sha256 `b5b3fa93…d115cf`),
  built by `scripts/imajev_bench/build_v2.py` with its own unpublished seed and split salt, and finished by
  `scripts/imajev_bench/finish_private.sh`. 202 items, 66 evidence clusters, 15 Unknown; audited
  blind on 41 items with 0 label errors (audit files: `reports/imajev-bench-hidden/private/private-1-audit/`).
  - 36 AI-generated scenes plus 30 programmatic text items, all in one test split.
  - It uses the same families, answers-by-construction and two-model image checks as the public set.
  - All its images are Flare-generated. The generator-mix formula of the time gave small plans no
    Nano Banana 2 scenes; this is fixed for later sets.
- **Never copy it** into `imajev-release/`, any public repository, a Hugging Face dataset, or a
  shared drive. Keep it on team machines only.
- **Audit:** before its first use, the hidden set gets the same labelled audit as the public set
  (`scripts/imajev_bench/apply_model_audit.py`). The audit uses auditors that are not being scored.
  Its audited error rate is published with every result.
- **Rotation:** build a new hidden set (`private-2`, and so on) at least every six months, or
  sooner when exposure (below) uses up a set. Retired sets may be released publicly as extra
  training or dev data once no current results depend on them.

## How a submission is scored

| Tier | What the submitter sends | Who runs it | Hidden items exposed? |
| --- | --- | --- | --- |
| **A. Weights or container** (preferred) | Model weights, adapter or a container image, plus inference settings | The team, offline, with `run_local_v2.py` (MLX or `--backend torch`) or an equivalent harness run | No |
| **B. API endpoint** | An endpoint URL and key (OpenAI-compatible, Azure OpenAI, Gemini or Vertex) plus settings | The team, with `evaluate_submission.py` calling the endpoint | Yes, to the endpoint owner |

**Procedure:**
1. **The submitter declares** the model identity (name and version or hash), precision, reasoning
   or thinking setting, prompt interface (direct option scoring or structured JSON), and any
   calibration.
2. **The team runs it:**
   - Tier A: run the harness on the hidden records, then
     `evaluate_submission.py --run <run folder> ...`.
   - Tier B: `evaluate_submission.py --provider ... --model ...`, or
     `--base-url ... --key-env ...` for OpenAI-compatible endpoints.
3. **The script writes two outputs:**
   - `reports/imajev-bench-hidden/private/<submission>/`: raw predictions and receipts. Team only.
   - `reports/imajev-bench-hidden/public/<submission>.md` and `.json`: the aggregate report.
4. **The team publishes only the aggregate report:** accuracy with a cluster-bootstrap 95%
   interval, per-track accuracy, correct-Unknown and false-abstention rates, contrast-set
   consistency, error rate, tier and declared settings.

**What we never share:** item-level results, the identities of failed items, or example questions.
Sharing them would let a submitter reconstruct the set.

## Exposure rules (Tier B)

- **Why exposure happens:** an API endpoint receives every hidden image and question. Assume the
  owner can log them.
- **Allowed without rotation:** major hosted providers whose terms exclude training on and
  retention of API traffic (for example Azure OpenAI and Vertex AI enterprise terms). The report
  still says "Tier B".
- **Everything else:** after a Tier B evaluation for any other endpoint, the set is marked exposed
  to that submitter in `private/submissions.jsonl`. A new hidden set is built before that submitter
  is scored again.
- **Limit:** at most 3 scored submissions per submitter per month, so repeated probing cannot map
  the set.

## Integrity checks on every result

- **Pinned set:** each report records the hidden set's name and the SHA-256 of its records file.
  A result is valid only for that exact set.
- **Verified runs:** `verify_run` checks that the run matches the hidden set's scoring content
  (IDs, splits, model inputs and answers) before anything is scored.
- **Quarantined items:** items the audit found defective are left out of scoring, with the same
  treatment for every submission.
