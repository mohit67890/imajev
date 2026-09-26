"""TAT-QA (CC BY 4.0, https://github.com/NExTplusplus/TAT-QA) train split -> phase-3 candidates (Stage 0 source B).

Keeps hard arithmetic questions only: derivations with >= 2 operations, or whose answer needs both the table and the text
(``answer_from == "table-text"``). Span / multi-span / count questions are not converted in round 1.
The derivation (e.g. ``(4,573 - 4,312) / 4,312``) is parsed and must reproduce the dataset answer; the distractors are the
derivation with one deliberate slip (wrong operation, swapped operands, stopping one step early, a neighbouring table cell
or another number of the same sentence, then sign / scale slips). Options carry the dataset's scale (%, thousand, million).

    .venv/bin/python scripts/p3/convert_tatqa.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from candidate import write  # noqa: E402
from convert_common import (MAX_STATE_TOKENS, ExprError, NumFormat, approx_tokens, build_choice_options, count_ops,  # noqa: E402
                            evaluate, numeric_key_text, numeric_options, parse_expr, parse_number, perturbations, render_table)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/p3/raw/tatqa/tatqa_dataset_train.json"
OUT = ROOT / "data/p3/candidates/B-tatqa.jsonl"
URL = "https://github.com/NExTplusplus/TAT-QA"
LICENCE = "CC-BY-4.0"
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
SUFFIX = {"percent": "%", "thousand": " thousand", "million": " million", "billion": " billion", "": ""}


def cell_number(cell: str) -> float | None:
    s = str(cell).replace(",", "").replace("$", "").strip()
    m = _NUM.search(s)
    if not m:
        return None
    v = float(m.group(0))
    if s.startswith("(") and v > 0:
        v = -v
    return v


def number_alternatives(table: list[list[str]], paragraphs: list[str]):
    """Other cells of the same row (other year), the same column one row up/down (other row), or other numbers of the same
    sentence for numbers that come from the text."""
    sentences = [s for p in paragraphs for s in re.split(r"(?<=[.;])\s+", p)]

    def alts(x: float) -> list[float]:
        out: list[float] = []
        ax = abs(x)
        for ri, r in enumerate(table):
            for ci, c in enumerate(r):
                v = cell_number(c)
                if v is not None and abs(abs(v) - ax) < 1e-9:
                    out += [cell_number(c2) for cj, c2 in enumerate(r) if cj != ci and cj > 0]
                    out += [cell_number(r2[ci]) for r2 in table[max(0, ri - 1): ri + 2] if r2 is not r and ci < len(r2)]
        if not out:
            for s in sentences:
                nums = [float(m) for m in _NUM.findall(s.replace(",", ""))]
                if any(abs(n - ax) < 1e-9 for n in nums):
                    out += [n for n in nums if abs(n - ax) > 1e-9 and not (1900 <= n <= 2100 and n == int(n))]
        seen, res = set(), []
        for v in out:
            if v is not None and abs(v - x) > 1e-9 and v not in seen:
                seen.add(v); res.append(v)
        return res
    return alts


def core_ops(tree: ast.AST) -> int:
    """Operations, not counting a final '* 100' that only turns a fraction into a percentage."""
    n = count_ops(tree)
    if isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Mult) and any(
            isinstance(s, ast.Constant) and s.value == 100 for s in (tree.left, tree.right)):
        n -= 1
    return n


def render_state(ctx: dict) -> str:
    paras = [p["text"].strip() for p in sorted(ctx["paragraphs"], key=lambda p: p.get("order", 0))]
    return "\n".join(["Excerpt from a company's financial report.", "", "Table:", render_table(ctx["table"]["table"]), "",
                      *paras]).strip()


def difficulty(ops: int, table_text: bool) -> int:
    d = {0: 1, 1: 2, 2: 3, 3: 4}.get(ops, 5)
    return max(1, min(5, d + (1 if table_text else 0)))


def convert_question(ctx: dict, q: dict, state: str, idx: int) -> tuple[dict | None, str]:
    if q.get("answer_type") != "arithmetic":
        return None, "not_arithmetic"
    try:
        tree = parse_expr(q.get("derivation") or "")
        val = evaluate(tree)
    except (ExprError, ZeroDivisionError, OverflowError):
        return None, "derivation"
    ops = core_ops(tree)
    table_text = q.get("answer_from") == "table-text"
    if not (ops >= 2 or table_text):
        return None, "easy"
    ans = q["answer"]
    num = ans if isinstance(ans, (int, float)) else parse_number(str(ans))
    if num is None:
        return None, "answer_parse"
    num = float(num)
    scale = q.get("scale") or ""
    factor = None
    for f in ((100.0, 1.0) if scale == "percent" else (1.0,)):
        if abs(val * f - num) <= 0.0051 + 0.001 * abs(num):
            factor = f
            break
    if factor is None:
        return None, "answer_mismatch"
    s = str(ans)
    dec = len(s.split(".")[1]) if "." in s else 0
    fmt = NumFormat(decimals=max(dec, 2 if abs(num) < 100 else dec), suffix=SUFFIX.get(scale, ""))
    gold = val * factor
    cands = []
    for k, v in perturbations(tree, number_alternatives(ctx["table"]["table"], [p["text"] for p in ctx["paragraphs"]])):
        cands.append((k, v * factor))
        if k == "stopped_early" and factor != 1.0:
            cands.append((k, v))
    uid = q["uid"]
    built = numeric_options(gold, cands, fmt, seed=uid)
    if not built:
        return None, "few_distractors"
    texts, gtext, kinds = built
    opts, gkey, kind_of = build_choice_options(texts, gtext, seed=uid, kinds=kinds, key_text=numeric_key_text)
    return {"id": f"p3-tatqa-{idx:06d}", "source": "B", "dataset": "tatqa", "family": "table_text_arithmetic",
            "difficulty": difficulty(ops, table_text), "state": state, "images": [],
            "field": {"type": "choice", "question": f"Based on the report excerpt: {q['question'].strip()}", "options": opts},
            "gold": gkey, "unknown_reason": None, "gold_kind": "dataset", "parent_id": None,
            "provenance": {"licence": LICENCE, "upstream_dataset": "tatqa", "upstream_split": "train", "upstream_id": uid,
                           "url": URL, "context_uid": ctx["table"]["uid"], "group_id": ctx["table"]["uid"], "derivation": q["derivation"],
                           "answer_from": q.get("answer_from"), "scale": scale, "option_kinds": kind_of}}, "ok"


def convert(raw: Path = RAW, out: Path = OUT) -> dict:
    data = json.load(open(raw))
    rows, why, nq = [], {}, 0
    for ctx in data:
        state = render_state(ctx)
        if approx_tokens(state) > MAX_STATE_TOKENS:
            why["too_long"] = why.get("too_long", 0) + len(ctx["questions"])
            continue
        for q in ctx["questions"]:
            nq += 1
            row, reason = convert_question(ctx, q, state, len(rows))
            why[reason] = why.get(reason, 0) + 1
            if row:
                rows.append(row)
    n = write(out, rows)
    return {"read": nq, "written": n, "reasons": why}


if __name__ == "__main__":
    print(json.dumps(convert(), indent=1))
