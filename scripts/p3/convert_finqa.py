"""FinQA (MIT, https://github.com/czyssrs/FinQA) train split -> phase-3 candidates (Stage 0 source B).

Keeps the hard items only: programs with >= 2 arithmetic steps, or whose gold evidence spans both the table and the text.
Numeric answers become 4-6 option choice questions; the distractors are the gold program re-executed with one deliberate
slip (wrong operation, swapped operands, stopping one step early, a neighbouring table cell = other year / other row, then
sign / scale slips). Yes/no programs (``greater``) become noul questions. Items whose own ``answer`` string disagrees with
the executed program are dropped as noisy.

    .venv/bin/python scripts/p3/convert_finqa.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from candidate import write  # noqa: E402
from convert_common import (MAX_STATE_TOKENS, ExprError, NumFormat, approx_tokens, build_choice_options, decimals_of,  # noqa: E402
                            evaluate, numeric_key_text, numeric_options, parse_number, perturbations, render_table)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/p3/raw/finqa/train.json"
OUT = ROOT / "data/p3/candidates/B-finqa.jsonl"
URL = "https://github.com/czyssrs/FinQA"
LICENCE = "MIT"

_OPS = {"add": ast.Add, "subtract": ast.Sub, "multiply": ast.Mult, "divide": ast.Div, "exp": ast.Pow}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def cell_number(cell: str) -> float | None:
    """'$ 6427' -> 6427; '$ -23158 ( 23158 )' -> -23158; '( 1.5 )' -> -1.5; '12.5% ( 12.5 % )' -> 12.5."""
    s = str(cell).replace(",", "").replace("$", "").strip()
    m = _NUM.search(s)
    if not m:
        return None
    v = float(m.group(0))
    if s.startswith("(") and v > 0:
        v = -v
    return v


def table_row_values(table: list[list[str]], name: str) -> list[float] | None:
    for r in table:
        if r and r[0].strip().lower() == name.strip().lower():
            vals = [cell_number(c) for c in r[1:]]
            vals = [v for v in vals if v is not None]
            return vals or None
    return None


def split_program(program: str) -> list[tuple[str, str, str]]:
    steps = []
    for m in re.finditer(r"(\w+)\(([^,()]*),\s*([^()]*)\)", program):
        steps.append((m.group(1), m.group(2).strip(), m.group(3).strip()))
    return steps


def arg_value(a: str) -> float:
    if a.startswith("const_"):
        c = a[6:]
        return -1.0 if c == "m1" else float(c)
    a2 = a.replace("%", "")
    return float(a2)


def program_tree(program: str, table: list[list[str]]):
    """FinQA program -> (arithmetic AST, n_steps, is_boolean). Raises ExprError on anything unsupported."""
    steps = split_program(program)
    if not steps:
        raise ExprError("empty program")
    nodes: list = []
    boolean = False
    for i, (op, a, b) in enumerate(steps):
        if op.startswith("table_"):
            vals = table_row_values(table, a)
            if not vals:
                raise ExprError(f"table row not found: {a}")
            f = op[6:]
            v = {"max": max, "min": min, "sum": sum, "average": lambda x: sum(x) / len(x)}[f](vals)
            nodes.append(ast.Constant(v))
            continue

        def node(x):
            if x.startswith("#"):
                return nodes[int(x[1:])]
            try:
                return ast.Constant(arg_value(x))
            except ValueError as e:
                raise ExprError(f"bad arg {x}") from e
        if op == "greater":
            if i != len(steps) - 1:
                raise ExprError("greater not last")
            boolean = True
            nodes.append(ast.Compare(left=node(a), ops=[ast.Gt()], comparators=[node(b)]))
            continue
        if op not in _OPS:
            raise ExprError(f"op {op}")
        nodes.append(ast.BinOp(left=node(a), op=_OPS[op](), right=node(b)))
    return nodes[-1], len(steps), boolean


def evidence_kinds(qa: dict) -> tuple[bool, bool]:
    g = qa.get("gold_inds") or {}
    return any(k.startswith("table") for k in g), any(k.startswith("text") for k in g)


def number_alternatives(item: dict) -> callable:
    """For a number in the program, other cells of the same table row (other year), then the same column (other row),
    then other numbers of the same evidence sentence."""
    table = item.get("table") or []
    gold_text = " ".join(v for k, v in (item["qa"].get("gold_inds") or {}).items() if k.startswith("text"))

    def alts(x: float) -> list[float]:
        out: list[float] = []
        for ri, r in enumerate(table):
            for ci, c in enumerate(r[1:], start=1):
                v = cell_number(c)
                if v is not None and abs(v - x) < 1e-9 * max(1, abs(x)) + 1e-9:
                    out += [cell_number(c2) for c2 in r[1:] if c2 is not c]
                    out += [cell_number(r2[ci]) for r2 in table[max(1, ri - 1): ri + 2] if r2 is not r and ci < len(r2)]
        if not out:
            nums = [float(m) for m in _NUM.findall(gold_text.replace(",", ""))]
            if any(abs(n - x) < 1e-9 for n in nums):
                out += [n for n in nums if abs(n - x) > 1e-9]
        seen, res = set(), []
        for v in out:
            if v is not None and abs(v - x) > 1e-9 and v not in seen:
                seen.add(v); res.append(v)
        return res
    return alts


def answer_format(answer: str, exe: float) -> tuple[NumFormat, float] | None:
    """Work out how the dataset states this answer and check it agrees with the executed program. None = disagreement."""
    ans = str(answer).strip()
    if not ans:
        return NumFormat(decimals=2), 1.0
    num = parse_number(ans)
    if num is None:
        return None
    if ans.endswith("%"):
        for factor in (100.0, 1.0):
            v = exe * factor
            dec = max(1, decimals_of(ans))
            if abs(v - num) <= max(0.51 * 10 ** (-decimals_of(ans)), 0.005 * abs(num)):
                return NumFormat(decimals=dec, suffix="%"), factor
        return None
    dec = min(max(decimals_of(ans), 1 if abs(exe) < 1000 else 0), 4)
    if abs(exe) < 1:
        dec = max(dec, 3)
    if abs(exe - num) <= max(0.51 * 10 ** (-decimals_of(ans)), 0.005 * abs(num)):
        return NumFormat(decimals=dec), 1.0
    return None


def render_state(item: dict) -> str:
    pre = "\n".join(s.strip() for s in item.get("pre_text", []) if s.strip() not in ("", "."))
    post = "\n".join(s.strip() for s in item.get("post_text", []) if s.strip() not in ("", "."))
    parts = ["Excerpt from a company's annual report (text is lower-cased as in the source).", "", pre, "",
             "Table:", render_table(item.get("table") or []), "", post]
    return "\n".join(parts).strip()


def difficulty(n_steps: int, table: bool, text: bool) -> int:
    d = {1: 2, 2: 3, 3: 4}.get(n_steps, 5)
    if table and text:
        d += 1
    return max(1, min(5, d))


def convert_item(item: dict, idx: int) -> tuple[dict | None, str]:
    qa = item["qa"]
    table = item.get("table") or []
    try:
        tree, n_steps, boolean = program_tree(qa["program"], table)
    except (ExprError, ValueError, KeyError, IndexError, ZeroDivisionError) as e:
        return None, f"program:{type(e).__name__}"
    has_table, has_text = evidence_kinds(qa)
    if not (n_steps >= 2 or (has_table and has_text)):
        return None, "easy"
    state = render_state(item)
    if approx_tokens(state) > MAX_STATE_TOKENS:
        return None, "too_long"
    uid = item["id"]
    base = {"id": f"p3-finqa-{idx:06d}", "source": "B", "dataset": "finqa", "difficulty": difficulty(n_steps, has_table, has_text),
            "state": state, "images": [], "unknown_reason": None, "gold_kind": "dataset", "parent_id": None,
            "provenance": {"licence": LICENCE, "upstream_dataset": "finqa", "upstream_split": "train", "upstream_id": uid, "url": URL,
                           "group_id": uid.rsplit("-", 1)[0],
                           "program": qa["program"], "n_steps": n_steps,
                           "evidence": "table+text" if has_table and has_text else ("table" if has_table else "text")}}
    q = qa["question"].strip()
    if boolean:
        cmp = tree
        try:
            gold = evaluate(cmp.left) > evaluate(cmp.comparators[0])
        except (ZeroDivisionError, ExprError):
            return None, "eval"
        if str(qa.get("exe_ans")).lower() in ("yes", "no") and gold != (str(qa["exe_ans"]).lower() == "yes"):
            return None, "disagree"
        return {**base, "family": "table_arithmetic_yesno",
                "field": {"type": "noul", "question": f"Based on the report excerpt: {q}"}, "gold": bool(gold)}, "ok"
    try:
        exe = evaluate(tree)
    except (ZeroDivisionError, ExprError, OverflowError):
        return None, "eval"
    ref = qa.get("exe_ans")
    if not isinstance(ref, (int, float)) or abs(exe - float(ref)) > max(1e-3, 1e-3 * abs(float(ref))):
        return None, "exe_mismatch"
    fmtf = answer_format(qa.get("answer", ""), exe)
    if fmtf is None:
        return None, "answer_mismatch"
    fmt, factor = fmtf
    cands = []
    for k, v in perturbations(tree, number_alternatives(item)):
        cands.append((k, v * factor))
        if k == "stopped_early" and factor != 1.0:
            cands.append((k, v))          # e.g. the raw change shown as if it were the % change
    built = numeric_options(exe * factor, cands, fmt, seed=uid)
    if not built:
        return None, "few_distractors"
    texts, gtext, kinds = built
    opts, gkey, kind_of = build_choice_options(texts, gtext, seed=uid, kinds=kinds, key_text=numeric_key_text)
    base["provenance"]["option_kinds"] = kind_of
    return {**base, "family": "table_arithmetic",
            "field": {"type": "choice", "question": f"Based on the report excerpt: {q}", "options": opts},
            "gold": gkey}, "ok"


def convert(raw: Path = RAW, out: Path = OUT) -> dict:
    data = json.load(open(raw))
    rows, why = [], {}
    for item in data:
        row, reason = convert_item(item, len(rows))
        why[reason] = why.get(reason, 0) + 1
        if row:
            rows.append(row)
    n = write(out, rows)
    return {"read": len(data), "written": n, "reasons": why}


if __name__ == "__main__":
    print(json.dumps(convert(), indent=1))
