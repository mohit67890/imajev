# Recovered-row adjudication (Kimi-K2.5 disagreements), phase 3

2026-09-26 (IST). Scope: the 1,616 teacher-rejected constructed rows in `data/p3/fixes/direct-recovered-rejected.jsonl`, built by
`scripts/p3/recover_rejected.py`. Kimi disagreed with our gold on 75 of the 147 sampled rows that parsed (plus 3 parse errors).
Every one of the 75 was adjudicated by reading the state and question and checking the image (GUI screenshots, defect and ABO photos)
and the generator code. 21 agreements were spot-checked.

Outputs:
* per item: `data/p3/review/recovered-adjudication.jsonl` (96 rows: 75 disagreements, 21 spot-checked agreements);
* drop list: `data/p3/fixes/drop-ids-recovered.json` (1,366 ids: `ids` / `why` / `by_rule` / `row_occurrences_by_file` /
  `eval_ids_flagged`; `build_manifest.py --drop-ids` reads `ids`);
* detector: `scripts/p3/audit_recovered.py`. It is read-only and runs in about 3 s over every file below. It rebuilds each GUI
  screen from its seed and the rebuilt meta equals the stored `screen_meta` on every row checked.

No generator code, live file or process was touched.

## 1. Verdict tally

Over the 75 disagreements: **ours 54, Kimi 15, both defensible 5, badly posed 1**. All 21 Kimi-wrong and ambiguous verdicts trace
to 13 systematic defects (section 2), and the drop filter removes every one of those sampled rows except one ambiguous GUI row that
is deliberately kept.

Kimi's own errors fall into four patterns:
* It applies a closed world (an unrecorded X-ray, a missing weight check or a "not captured" field is taken as a pass or no).
* It assumes a missing quantity is 1.
* It ignores explicit "gap of N lines" markers.
* It misreads greyed-out (disabled) buttons, and once hallucinated a defect on a clean fryum.

| family | stratum | rows in recovered file | dropped by filter | sampled | Kimi disagree | ours | Kimi | ambiguous | bad | wrong / ambiguous left in sample after filter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| image_joint_rule | unk | 769 | 108 | 40 | 22 | 17 | 4 | 1 | 0 | 0 / 0 (3 agreements were also defects, all dropped) |
| gui_action | ans | 94 | 8 | 11 | 7 | 4 | 2 | 1 | 0 | 0 / 1 (GUI_SEL_CLICK, kept) |
| gui_action | unk | 66 | 11 | 9 | 4 | 1 | 3 | 0 | 0 | 0 / 0 (1 agreement was a GUI_DARK defect, dropped) |
| judge_hard | unk | 102 | 5 | 9 | 8 | 7 | 0 | 1 | 0 | 0 / 0 |
| judge_hard | ans | 97 | 20 | 9 | 2 | 1 | 1 | 0 | 0 | 0 / 0 |
| multi_label | unk | 107 | 0 | 9 | 6 | 6 | 0 | 0 | 0 | 0 / 0 |
| multi_label | ans | 7 | 1 | 4 | 1 | 0 | 0 | 1 | 0 | 0 / 0 |
| agent_action | unk | 45 | 24 | 7 | 6 | 2 | 4 | 0 | 0 | 0 / 0 |
| agent_action | ans | 41 | 7 | 6 | 2 | 1 | 1 | 0 | 0 | 0 / 0 (1 Kimi parse error, unreviewed) |
| routing_hard | unk | 32 | 3 | 5 | 4 | 4 | 0 | 0 | 0 | 0 / 0 |
| routing_hard | ans | 25 | 0 | 5 | 0 | – | – | – | – | 0 / 0 |
| long_input | unk | 19 | 0 | 5 | 4 | 4 | 0 | 0 | 0 | 0 / 0 |
| long_input | ans | 42 | 0 | 6 | 1 | 1 | 0 | 0 | 0 | 0 / 0 |
| rule_exception | unk | 36 | 1 | 6 | 3 | 3 | 0 | 0 | 0 | 0 / 0 |
| rule_exception | ans | 26 | 2 | 5 | 2 | 1 | 0 | 1 | 0 | 0 / 0 |
| long_policy | unk | 42 | 0 | 6 | 1 | 1 | 0 | 0 | 0 | 0 / 0 |
| long_policy | ans | 66 | 6 | 8 | 2 | 1 | 0 | 0 | 1 | 0 / 0 |
| **total** | | **1,616** | **196** | 150 | 75 | 54 | 15 | 5 | 1 | |

"Ours" on a disagreement means Kimi's reason was checked and found wrong, not just outvoted. Sample sizes per stratum are 5 to 11, so
the per-stratum error estimates are coarse. The filters are backed by code-level root causes, not by the sample alone.

## 2. Defects

Codes are the rule names in `scripts/p3/audit_recovered.py`. All are DROP rules unless marked INFO.

### GUI (gen_gui.py)

**GUI_SEL_DELETE / GUI_SEL_ROW: selection status is random.** `build_toolbar` draws `sel = rng.random() < 0.5` and renders
"1 row selected" or "No rows selected" independently of the goal. `toolbar_available` then says gold = true whenever the control is
enabled.
* For "Can you permanently remove **the selected rows** from this screen right now?" with "No rows selected", the gold must be false.
  * `p3-I-gui-001353`: Delete is enabled, and the status line under the table reads "No rows selected".
  * `p3-I-gui-000891`: same.
* GUI_SEL_ROW covers the same "Can you …" question for row-level goals (this item, this task, the file) with no row selected.
* INFO GUI_SEL_CLICK: the "Is the control to … available to click right now?" wording, where the enabled button literally answers
  it. Example: `p3-I-gui-004960`, which is both defensible and kept.
* Fix: tie `sel` to selection-dependent goals, or ask only about control availability.

**GUI_DARK: page theme is independent of the "Dark theme" switch.** `build_settings` picks the theme from 7 THEMES (2 dark) at
random, then asks "do you see the app in dark colours?".
* When the switch is not shown, the gold is unknown, but the screenshot answers it.
  * `p3-I-gui-001008` and `p3-I-gui-003997`: dark pages, so the answer is true.
  * `p3-I-gui-003433` (a Kimi agreement): a white page, so the answer is false.
* When the switch is shown, the gold is wrong whenever the switch state and the page darkness differ.
* Fix: render a dark theme exactly when Dark theme is on, and never ask it as a not-on-screen unknown.

**GUI_ALT: a second control achieves the goal.**
* Goal "let a colleague open this report": when Share is absent, the gold is not_listed, but an enabled **Copy link** is on screen.
  * `p3-I-gui-001175`: Export PDF / Publish / Duplicate / Archive / Copy link.
* The rule also covers toolbar_goal rows with both Share and Copy link on screen (two right answers), "copy a web address" unknowns
  with Share on screen, and disabled-Share rows with an enabled Copy link.
* INFO:
  * GUI_ALT_WEAK: disabled Copy link with an enabled Share.
  * GUI_PRINT_PDF: disabled Export PDF with an enabled Print (`p3-I-gui-004335`, ours).
* Fix: exclude alternative achievers from the distractor sample (OTHER_LABELS).

### Image joint (gen_image_joint.py)

**IJ_WEIGHT: "weighs more than 2 kg" is treated as unseeable.** It is in `HIDDEN_ABO`, but the photo settles it for almost every
ABO type: sofa, bed, rug and shelf are heavier; phone case, ring, earrings and sandal are lighter.
* `p3-I-joint-005434`: a full sofa. The priority is 1, not unknown.
* `p3-I-joint-005287` (a storage-drawer unit) was a Kimi agreement but is the same construction.
* Fix: remove weight from HIDDEN_ABO.

**IJ_WEIGHCHECK: the hidden fact duplicates a record field.** In def_two_units, the hidden fact is "was weighed under the minimum",
but the record gives each unit `weight_check: pass/fail`. A passed weight check answers it.
* `p3-I-joint-012152`: both units pass; unit 1 has a bright-spot defect. The answer is unit_2.
* Also `012319` and `013397`.
* Fix: drop that hidden text for def_two_units.

**IJ_HALT_NODEFECT: an impossible total makes the item "unknown".** In def_severity_points, `points_so_far` is withheld and its
alternatives are {0, 999}. When the pictured unit is clean (adds 0), a unit that adds nothing cannot bring a running line to a halt.
The unknown rests on total = 999, which is a line that has already halted.
* `p3-I-joint-014860` (a clean PCB, "Will this unit bring the line to a halt?").
* `014247` and `014723` (Kimi agreements) are the same construction.
* Fix: alternative totals in 0..limit-1.

**Not a defect, but a convention the owner should confirm (section 5).** Of the 22 image_joint disagreements, 8 are
hidden-attribute unknowns, such as an X-ray result, seal-pressure test or recall status that the rule names and the record omits.
Kimi assumes "not recorded means passed". We keep them as unknown, in line with the Jev evidence convention used in every text
family ("not captured" / "not stated" means unknown), and the D-family reviewers found Kimi making the same error on text. Of the
661 image_joint unknowns kept after the filter 282 are of this kind.

### gen_policy (source A) and gen_traps (source D)

**D_BURY_IN_REPLY.** `t_bury` with `where="after_decisive"` inserts the notice block at the next blank line, which for judge_hard
falls inside a candidate reply's dashed delimiters. The reply's word count grows, but `spec.replies[].words` and the gold are stale.
In the recovered file, 15 of the 20 rows flip the length criterion.
* `p3-trap-bury-p3-pol-judg-s0-002449`: the gold is 5, the true score is 4 (Kimi was right).
* Fix: never insert inside a dashed block.

**D_NEG_PAIR_UNK.** A choice negation of an unknown parent kept as unknown. "Which proposed answer is NOT supported?" is fairly
answered "Neither answer is supported" when the material settles neither. This is the same issue as the `pair_overlap: partial`
rows already excluded, extended to full overlap.
* `p3-trap-negation-p3-pol-judg-s0-001324-u` (both defensible; Kimi's own answer was wrong).
* The D-traps count (261 overall) includes the partial rows, which are in the pool but not recovered.

**D_TOOL_LOST.** A tool_next unknown is built from a call with "no response recorded (connection reset)", and `worlds()` counts it
as possibly successful. That is wrong on two counts:
* The task says a call counts as done only when it returned status ok.
* The success-branch option carries ids or amounts that appear nowhere in the log.

So the retry is the only answer.
* `p3-trap-near_miss-p3-pol-agnt-s0-004139-u` and `p3-trap-bury-p3-pol-agnt-s0-004915-u`: Kimi was right.
* **The source A parents have the same bug** (105 rows in A-policy).
* The corrected gold is computable: run `rederive_gold` with `unk` removed and the lost step taken out of `done`.

**D_NM_NUM_UNGROUNDED.** A near_miss "number" trap on tool_next changes an amount that the state never states, for example
`take_fee(amount=375.00)` vs 375.05. The gold cannot be told apart from the added option.
* `p3-trap-near_miss-p3-pol-agnt-s0-004771`.

**D_NEG_ZERO_WORLD.** For craft/toolsel negations, `rederive_gold` treats a completion with no valid recipe or tool
(`__ambiguous__`) as invalid, so the gold stays unknown even though neither proposed answer is right in any completion. The gold
should be "Neither".
* `p3-trap-negation-p3-pol-agnt-s0-004378-u`: Kimi was right.

**D_LP_BAND.** The long_policy banded schema in the public_sector domain names the applicant tier "Band 1–4", which collides with
the fee-band numbering. "Where the band is Band 4, clause 20.2 applies" reads as the floor-area band.
* `p3-trap-negation-p3-pol-lpol-s0-002148` (badly posed; 5 of 6 recovered rows have diverging readings).
* Fix: rename the tier, as rule_exception's `TIER_OVERRIDE` already does.

**D_RX_EXCL_SCOPE.** In rule_exception `action`, an excluded category without its re-admit flag has gold "decline". The phrasing
"{cat}: excluded from this section, except where …" also reads as "this section gives no action".
* `p3-trap-near_miss-p3-pol-rexc-s0-000281` (both defensible).

**D_ML_VELOCITY.** multi_label risk/velocity rendered as raw JSON keys, so whether `txns_24h` includes declined attempts (which
decides an approved-only threshold) is lost.
* `p3-trap-negation-p3-pol-mlab-s0-000423` (both defensible).

**Checked and sound (Kimi was wrong):**
* judge_hard unknowns with a withheld amount, reference or annexed period.
* multi_label unknowns with a redacted sentence.
* long_input count_after unknowns: the "gap of N lines" is always stated and hides the last restart.
* routing_hard "not captured" unknowns.
* rule_exception / long_policy blank or "not stated" fields.
* image_joint withheld quantity, days or weight-check fields.

## 3. Affected rows per file (all flags, from `audit_recovered.py --json`)

| file | GUI_SEL_DELETE | GUI_SEL_ROW | GUI_DARK | GUI_ALT | IJ_WEIGHT | IJ_WEIGHCHECK | IJ_HALT | D_BURY | D_NEG_PAIR | D_TOOL_LOST | D_NM_NUM | D_NEG_ZERO | D_LP_BAND | D_RX_EXCL | D_ML_VEL | dropped ids |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixes/direct-recovered-rejected | 7 | 1 | 6 | 5 | 50 | 33 | 25 | 20 | 18 | 15 | 7 | 10 | 6 | 2 | 1 | 196 |
| candidates-clean/I-gui | 7 | 31 | 31 | 9 | | | | | | | | | | | | 78 |
| candidates-clean/I-joint | | | | | 102 | 44 | 67 | | | | | | | | | 213 |
| candidates-clean/D-traps | | | | | | | | 123 | 241 | 52 | 22 | 26 | 31 | 5 | 5 | ≈505 |
| candidates-clean/A-policy (source A) | | | | | | | | | | 105 | | | 172 | 21 | 9 | 307 |
| labeltrain/direct-constructed-leftovers | | | | | 9 | 9 | 13 | 21 | 130 | 83 | 5 | 16 | 74 | 9 | | 369 |
| labeltrain/…-leftovers-late | | 3 | 2 | | | | | | | | | | | | 2 | 7 |
| variants/constructed | | | | | | | | 57 | 20 | 58 | | | 142 | 15 | 3 | 295 |
| *eval* pool/heldout-fresh-gui | | 1 | 2 | | | | | | | | | | | | | *not dropped* |
| *eval* pool/heldout-fresh | | | | | 5 | | | 3 | 2 | 15 | 2 | | 4 | | | *not dropped* |

**Unique ids in the drop list: 1,366.** The recovered-file ids also appear in candidates-clean D-traps, I-gui and I-joint.

Additional INFO rows, not dropped:
* GUI_SEL_CLICK: 43 in I-gui, 8 recovered.
* GUI_ALT_WEAK: 5 / 3.
* GUI_PRINT_PDF: 6 / 1.

The GUI answerable figures cover all of I-gui, including rows the teacher kept. The teacher kept 0 of the 7 GUI_SEL_DELETE rows;
all 7 were rejected and then recovered, so the teacher's rejections were right there.

**Eval gold needs fixing:** 7 of the 7 D agent_action unknowns and 8 of the 13 A agent_action unknowns in `heldout-fresh.jsonl` are
D_TOOL_LOST. The full lists are under `eval_ids_flagged` in the drop file. Relabel or remove them before scoring held-out numbers.

## 4. Recommendations

For the recovered set:

| family | stratum | action |
|---|---|---|
| image_joint_rule | unk (769) | **KEEP-WITH-FILTER**: drop IJ_WEIGHT, IJ_WEIGHCHECK, IJ_HALT_NODEFECT (108), leaving 661. Hidden-attribute unknowns are kept subject to the convention in section 5. |
| gui_action | ans (94) | **KEEP-WITH-FILTER**: drop GUI_SEL_DELETE and GUI_SEL_ROW (8). Optionally also GUI_SEL_CLICK (8) for a stricter set. |
| gui_action | unk (66) | **KEEP-WITH-FILTER**: drop GUI_DARK and GUI_ALT (11). |
| judge_hard | ans (97) | **KEEP-WITH-FILTER**: drop D_BURY_IN_REPLY (20). |
| judge_hard | unk (102) | **KEEP-WITH-FILTER**: drop D_NEG_PAIR_UNK (5). |
| multi_label | unk (107) | **KEEP** |
| multi_label | ans (7) | **KEEP-WITH-FILTER**: drop D_ML_VELOCITY (1). |
| agent_action | unk (45) | **KEEP-WITH-FILTER**: drop D_TOOL_LOST, D_NEG_ZERO_WORLD, D_NEG_PAIR_UNK (24), leaving 21. This is the weakest stratum (4 of 7 sampled were wrong, all caught); dropping the remaining 21 costs little if the owner prefers. |
| agent_action | ans (41) | **KEEP-WITH-FILTER**: drop D_NM_NUM_UNGROUNDED (7). |
| routing_hard | unk (32) / ans (25) | **KEEP-WITH-FILTER** (unk: drop D_NEG_PAIR_UNK and D_NEG_ZERO_WORLD, 3) / **KEEP** (ans) |
| long_input | unk (19) / ans (42) | **KEEP** / **KEEP** |
| rule_exception | unk (36) / ans (26) | **KEEP-WITH-FILTER** (drop D_NEG_PAIR_UNK, 1) / **KEEP-WITH-FILTER** (drop D_RX_EXCL_SCOPE, 2) |
| long_policy | unk (42) / ans (66) | **KEEP** / **KEEP-WITH-FILTER** (drop D_LP_BAND, 6) |

Net: **1,420 of 1,616 recovered rows are kept.**

Wider:
* Pass `data/p3/fixes/drop-ids-recovered.json` to `build_manifest.py --drop-ids`, alongside `drop-ids-audit.json`. It removes the
  same defects from I-gui, I-joint, D-traps, A-policy, the direct leftovers and variants/constructed (1,366 ids).
* Generator fixes to make before any regeneration:
  * the five one-line fixes above: selection status, dark theme, alternative achievers, weight hidden attribute, two_units weigh
    text;
  * severity totals in 0..limit-1;
  * no bury insertion inside dashed blocks;
  * tool_next "lost response" means not done;
  * near-miss numbers must be grounded in the state;
  * `__ambiguous__` completions in negation re-derivation;
  * a public_sector tier name;
  * rule_exception "excluded from this section" phrasing;
  * JSON velocity keys.
* D_TOOL_LOST rows can be relabelled to the retry instead of dropped (`rederive_gold` with `unk` removed). They are dropped here to
  keep the change read-only.

## 5. For the owner's eyes (the only open calls)

1. **Closed-world convention for image_joint hidden attributes** (`p3-I-joint-010287`, `-013297`).
   * The rule names an X-ray, seal-pressure or recall result that the record omits. We say unknown; Kimi, and presumably the
     teacher, say it passed.
   * Consistent with the text families, so they are kept (282 rows).
   * If the owner prefers closed-world, drop `provenance.unknown_construction == "hidden_attribute"`.
2. **GUI_SEL_CLICK** (`p3-I-gui-004960`): whether "Is the control … available to click" with "No rows selected" should stay true.
   8 recovered rows, 43 in I-gui.
3. **Held-out eval relabel**: 15 D_TOOL_LOST held-out unknowns (plus the other `eval_ids_flagged`) currently carry wrong gold. They
   should be fixed before any held-out comparison.
