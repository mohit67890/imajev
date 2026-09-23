# State-grounded and reference/target probes — baselines (23 Sept 2026)

Human-labelled probes built by the v2.1 data worker (`scripts/v2/state_grounded/`): `data/manifests/decision-v2-state-probe.jsonl`
(200 items, Open Images verified labels + ABO fields, six families) and `decision-v2-pairs-probe.jsonl` (60 composited reference/target
pairs, labels by construction). Scored through the playground API with `scripts/v2/state_grounded/probe.py`.

| Model | State probe overall | claim_supported | which_field_conflicts | count_matches | label_text | condition | instruction_changes_answer (both halves right) | Pairs overall | what_changed | target_still_matches | which_field_now_wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B v1.1 | 60.5% | 60.6% | 85.3% | 42.4% | 66.7% | 69.7% | 38.2% (4/17) | 68.3% | 80.0% | 75.0% | 50.0% |

Reading: simple claim-vs-photo checks are better than the tester feedback implied; the real gaps are instructions in the state that change
the right answer (38%), count claims (42%), and naming which state field a change invalidates (50%). Raw results: `state-probe-2b-v1.1.json`,
`pairs-probe-2b-v1.1.json`. v2.1 training data: state_grounded 60k candidates, pairs_grounded 19,975 (labels by construction; composited
edits are visibly pasted in places, see `pairs-examples/`), pairs_natural 10k for the teacher.
