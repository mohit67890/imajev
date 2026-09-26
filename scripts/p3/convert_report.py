"""Run every phase-3 public-dataset converter and write reports/phase3/public-datasets.md (counts + short examples).

    .venv/bin/python scripts/p3/convert_report.py            # re-converts everything (ALFWorld from its expert dump)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import convert_alfworld, convert_finqa, convert_musique, convert_plumb, convert_strategyqa, convert_tatqa  # noqa: E401,E402
from candidate import read  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/phase3/public-datasets.md"
CONVERTERS = [("finqa", convert_finqa), ("tatqa", convert_tatqa), ("musique", convert_musique),
              ("strategyqa", convert_strategyqa), ("alfworld", convert_alfworld), ("plumb", convert_plumb)]


NOTES = """## Design decisions per dataset

- **FinQA** (train 6,251). Kept: programs with >= 2 steps, or gold evidence from both the table and the text. Dropped: the
  program does not execute to `exe_ans` (94), or the dataset's own `answer` string disagrees with the executed program (366,
  e.g. a sign flip or a wrong x100), because the gold is ambiguous. Options: gold + 3-5 distractors. Each distractor is the
  program re-executed with one slip: a neighbouring table cell (other year / other row), wrong operation, swapped operands,
  stopping one step early (for % questions also the raw change shown as a %), then sign / x100 / /100 slips. Every distractor
  differs from the gold by >= 2% and by >= 2 display units after rounding. Distractors are also kept within x12 of the gold,
  so absurd values such as a product of two amounts are dropped. `greater` programs become noul items. Difficulty:
  2 steps = 3, 3 steps = 4, 4+ steps = 5, +1 when table and text are both needed.
- **TAT-QA** (train 13,215 questions). Only arithmetic questions with >= 2 operations (a final `* 100` is not counted), or with
  `answer_from = table-text`. Span, multi-span and count questions are not converted in round 1. The derivation must reproduce
  the dataset answer (39 dropped). Distractors are built as for FinQA, with the dataset's scale in the option text (%, thousand,
  million). Difficulty as for FinQA.
- **MuSiQue** (MuSiQue-Full train, 19,938 answerable/unanswerable pairs). All 3- and 4-hop pairs, plus a fixed sample of 4,000
  2-hop pairs. Each pair gives two rows over the same 5 options: the answerable one (dataset gold) and the unanswerable twin
  (gold null, `insufficient_evidence`, `parent_id` = the answerable row, so both land in the same split). Distractors, in this
  order: intermediate-hop answers (stopping at the wrong hop), answers to other train questions with the same final relation
  (first those that appear in this item's passages), then same wh-word + answer-shape answers, then passage titles. A
  distractor that equals, contains or is contained in the gold or an alias is dropped. About a third of the distractors are
  the weaker "same answer class" fillers, which can be a different entity type. The hop trap and the unknown twin carry the
  difficulty. Difficulty = hops, +1 for the unanswerable twin.
- **StrategyQA** (official train 2,290; 18 one-step questions dropped). The item is noul. The state is the union of the three
  annotators' evidence paragraphs plus 4 lexically related distractor paragraphs from other questions' evidence, shuffled.
  `facts` and `decomposition` are never shown. Many steps are world knowledge (`no_evidence` / `operation`), so the question
  says general knowledge is allowed. It is not framed as "passages only", and there are no unknown golds. Difficulty:
  2-3 steps = 3, 4 = 4, 5 = 5 (+1 when there are no evidence paragraphs).
- **ALFWorld** (3,553 train games; `movable` / `Sliced` games skipped as in ALFWorld itself). Every solvable train game was
  played with ALFWorld's hand-coded expert (`convert_alfworld.py dump`, TextWorld engine). Only decisive steps are kept: take,
  move (put), heat, cool, clean, use, and a `go to X` when X is where the needed object or lamp was already seen (in the
  history), or X is the receptacle or appliance the next action uses. Exploration steps (`go to shelf 3` while searching,
  `open drawer`, `look`) are dropped because many answers are equally right there. Options are all admissible commands
  except commands that would be equally right: another instance of the same object or receptacle type, `go to` any other
  place where the needed thing was seen, or, in look-at-in-light tasks, using the lamp before taking the object. The state is
  the task, the start description and the full action/observation history. `upstream_id` is the game's trial directory;
  `provenance.step` is the step index. Of the 3,553 games, the expert won 3,002 within the 80-step cap. The other 551 (505 of them
  pick_two_obj_and_place, where the hand-coded expert loops) are dropped, so pick-two tasks are under-represented. Difficulty 3, +1 for two-object / heat / cool / clean tasks, +1 for `go to` (memory),
  +1 for step >= 15 or >= 45 options.
- **plumb-decisions** (train 5,014; test never downloaded). One candidate per question, with the full document as state.
  Choice options come from the dataset's `criteria` (key -> description). Noul items write the true/false criteria into the
  question, because some items define `true` as the negation of the question's wording. Score items use the level
  descriptions in order. `teacher_probs` (two thinking solves, or the exact distribution for probability items) is kept in
  provenance. Families keep plumb's names, except judge -> judge_hard and routing -> routing_hard. Difficulty: medium = 3,
  long = 4, very long = 5. There are no unknown golds.

## Provenance fields for decontamination

`provenance.upstream_dataset` (canonical lowercase name), `upstream_split` (always `train`) and `upstream_id` (native id:
FinQA `ADI/2009/page_49.pdf-1`, TAT-QA question uid, MuSiQue `3hop1__...` (the same id on both twins; the twin is in
`upstream_variant`), StrategyQA `qid`, ALFWorld trial path, plumb `train-...-qN`). `provenance.group_id` groups the rows that
share a source document (FinQA filing page, TAT-QA context uid, plumb document, ALFWorld game, MuSiQue pair). Keep a group
in one split when building held-out sets. `parent_id` is used only for true variants (the MuSiQue unanswerable twin).
"""


def short(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def example(r: dict) -> str:
    f = r["field"]
    st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])
    lines = [f"- `{r['id']}` (upstream `{r['provenance']['upstream_id']}`, family `{r['family']}`, difficulty {r['difficulty']})",
             f"  - state: {short(st if r['dataset'] != 'alfworld' else st[-420:], 420)}",
             f"  - {f['type']}: {short(f['question'], 300)}"]
    if f["type"] == "choice":
        opts = f["options"]
        shown = ", ".join(f"`{o['key']}`={short(o['text'], 40)}" for o in opts[:8]) + (f", ... ({len(opts)} options)" if len(opts) > 8 else "")
        lines.append(f"  - options: {shown}")
    elif f["type"] == "score":
        lines.append("  - levels: " + "; ".join(f"{lv['value']}={short(lv['description'], 40)}" for lv in f["levels"]))
    gold = r["gold"]
    if f["type"] == "choice" and gold is not None:
        gold = f"`{gold}` ({next(o['text'] for o in f['options'] if o['key'] == gold)})"
    lines.append(f"  - gold: {gold}" + (f" ({r['unknown_reason']})" if r["gold"] is None else ""))
    return "\n".join(lines)


def main() -> None:
    stats, rows_by = {}, {}
    for name, mod in CONVERTERS:
        if name == "alfworld" and not convert_alfworld.DUMP.exists():
            stats[name] = {"skipped": "no expert dump"}
            continue
        stats[name] = mod.convert()
        rows_by[name] = list(read(mod.OUT))
    total = sum(len(v) for v in rows_by.values())
    out = ["# Phase 3, Stage 0 source B: public datasets converted (round 1)", "",
           "Generated by `scripts/p3/convert_report.py`. Licences are in `reports/phase3/licences.md`. All rows are train split only,",
           "validated by `scripts/p3/candidate.py` `write()`. Decontamination against the benchmarks has not been run yet (separate step).", "",
           f"**Total: {total:,} items.** WebShop and Mind2Web are excluded on licence grounds (see licences.md).", "",
           "| Dataset | Items | Unknown gold | Choice / noul / score | Difficulty 1/2/3/4/5 | Share at difficulty >= 3 | Mean options (choice) |",
           "|---|---:|---:|---|---|---:|---:|"]
    for name, rows in rows_by.items():
        d = Counter(r["difficulty"] for r in rows)
        t = Counter(r["field"]["type"] for r in rows)
        ch = [len(r["field"]["options"]) for r in rows if r["field"]["type"] == "choice"]
        unk = sum(r["gold"] is None for r in rows)
        hard = sum(v for k, v in d.items() if k >= 3) / max(1, len(rows))
        out.append(f"| {name} | {len(rows):,} | {unk:,} | {t['choice']:,} / {t['noul']:,} / {t['score']:,} | "
                   f"{' / '.join(str(d[k]) for k in range(1, 6))} | {hard:.0%} | {sum(ch) / max(1, len(ch)):.1f} |")
    out += ["", "## Families", "", "| Dataset | Family | Items |", "|---|---|---:|"]
    for name, rows in rows_by.items():
        for fam, n in Counter(r["family"] for r in rows).most_common():
            out.append(f"| {name} | {fam} | {n:,} |")
    out += ["", "## Converter statistics (what was dropped and why)", ""]
    for name, st in stats.items():
        out.append(f"- **{name}**: `{json.dumps(st)}`")
    kinds: dict[str, Counter] = {}
    for name, rows in rows_by.items():
        c = Counter(k for r in rows for k in (r["provenance"].get("option_kinds") or {}).values() if k != "gold")
        if c:
            kinds[name] = c
    if kinds:
        out += ["", "## Distractor kinds (numeric and MuSiQue options)", ""]
        for name, c in kinds.items():
            out.append(f"- **{name}**: " + ", ".join(f"{k} {v:,}" for k, v in c.most_common()))
    out += ["", "## Examples (3 per dataset, shortened)", ""]
    for name, rows in rows_by.items():
        out.append(f"### {name}")
        step = max(1, len(rows) // 3)
        for r in rows[::step][:3]:
            out.append(example(r))
        out.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(out) + "\n"
    body += "\n" + NOTES
    REPORT.write_text(body)
    print(f"wrote {REPORT} ({total} items)")


if __name__ == "__main__":
    main()
