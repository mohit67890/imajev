"""Shared helpers for the phase-3 public-dataset converters (scripts/p3/convert_<name>.py, Stage 0 source B).

- option keys and deterministic shuffling;
- a small safe arithmetic expression engine (parse, evaluate, perturb) used to build numeric distractors for FinQA and
  TAT-QA: wrong operation, swapped operands, stopping one step early, wrong row / wrong year cell, sign and scale slips;
- numeric option construction that guarantees every distractor differs from the gold after display rounding;
- text helpers (table rendering, token estimate, answer normalisation).
"""
from __future__ import annotations

import ast
import hashlib
import math
import random
import re
import string
from dataclasses import dataclass
from typing import Callable, Iterable

MAX_STATE_TOKENS = 16000


# ------------------------------------------------------------------------------------------------------------------ basics
def option_key(text: str, taken: set[str] | None = None, prefix: str = "") -> str:
    """Lowercase snake_case ASCII key, <= 40 chars, unique within a field, never 'unknown'."""
    key = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")[:40] or "option"
    if prefix and not key.startswith(prefix):
        key = (prefix + key)[:40]
    if key[0].isdigit():
        key = ("v_" + key)[:40]
    if key in ("unknown", "__unknown__", "option_unknown"):
        key = "opt_" + key
    if taken is not None:
        base, n = key, 2
        while key in taken:
            suffix = f"_{n}"
            key = base[:40 - len(suffix)] + suffix
            n += 1
        taken.add(key)
    return key


def stable_rng(seed: str) -> random.Random:
    return random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))


def stable_shuffle(items: list, seed: str) -> list:
    out = list(items)
    stable_rng(seed).shuffle(out)
    return out


def approx_tokens(text: str) -> int:
    """Rough token count (Qwen tokenizer ~3.6-4 chars/token on English prose; tables are denser, so be conservative)."""
    return int(len(text) / 3.3) + 1


def render_table(rows: list[list[str]]) -> str:
    """Pipe table; empty cells kept so column positions stay faithful."""
    out = []
    for i, r in enumerate(rows):
        cells = [str(c).replace("|", "/").replace("\n", " ").strip() for c in r]
        out.append("| " + " | ".join(cells) + " |")
        if i == 0:
            out.append("|" + "---|" * len(cells))
    return "\n".join(out)


_ARTICLES = re.compile(r"\b(a|an|the)\b")


def norm_answer(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def numeric_key_text(text: str) -> str:
    """'-9.93%' -> 'neg 9.93 pct' so a number and its negative never collide on the same key."""
    t = text.strip()
    t = re.sub(r"^-", "neg ", t).replace("%", " pct").replace(",", "")
    return t.replace(".", " ")


def build_choice_options(texts: list[str], gold_text: str, seed: str, kinds: dict[str, str] | None = None,
                         key_text: Callable[[str], str] | None = None) -> tuple[list[dict], str, dict[str, str]]:
    """Shuffle option texts deterministically, key them, return (options, gold key, {key: kind})."""
    order = stable_shuffle(texts, seed)
    taken: set[str] = set()
    opts, gold_key, kind_of = [], None, {}
    for t in order:
        k = option_key(key_text(t) if key_text else t, taken)
        opts.append({"key": k, "text": t, "description": ""})
        if t == gold_text:
            gold_key = k
        if kinds is not None:
            kind_of[k] = kinds.get(t, "")
    assert gold_key is not None
    return opts, gold_key, kind_of


# ------------------------------------------------------------------------------------------------------ expression engine
_ALLOWED_BIN = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}
_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}


class ExprError(ValueError):
    pass


def parse_expr(text: str) -> ast.expr:
    """Parse a derivation such as '(4,573 - 4,312) / 4,312', '[(9+7)/2] - 3%' or '$111/$495 * 100'."""
    s = text.replace("[", "(").replace("]", ")").replace("$", "").replace("×", "*").replace("x ", "* ").replace("÷", "/")
    s = s.replace("−", "-").replace("–", "-")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)          # thousands separators
    s = re.sub(r"(\d(?:\.\d+)?)\s*%", r"\1", s)       # '3.1%' -> 3.1 (TAT-QA writes percent points this way)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or re.search(r"[A-Za-z_]", s):
        raise ExprError(f"not arithmetic: {text!r}")
    try:
        tree = ast.parse(s, mode="eval").body
    except SyntaxError as e:
        raise ExprError(str(e)) from e
    _check(tree)
    return tree


def _check(n: ast.AST) -> None:
    if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BIN:
        _check(n.left); _check(n.right)
    elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        _check(n.operand)
    elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        pass
    else:
        raise ExprError(f"disallowed node {type(n).__name__}")


def evaluate(n: ast.AST) -> float:
    if isinstance(n, ast.Constant):
        return float(n.value)
    if isinstance(n, ast.UnaryOp):
        v = evaluate(n.operand)
        return -v if isinstance(n.op, ast.USub) else v
    if isinstance(n, ast.BinOp):
        a, b = evaluate(n.left), evaluate(n.right)
        op = type(n.op)
        if op is ast.Add:
            return a + b
        if op is ast.Sub:
            return a - b
        if op is ast.Mult:
            return a * b
        if op is ast.Div:
            if b == 0:
                raise ZeroDivisionError
            return a / b
        if op is ast.Pow:
            if abs(b) > 50:
                raise ExprError("pow too big")
            return a ** b
    raise ExprError(f"cannot evaluate {type(n).__name__}")


def count_ops(n: ast.AST) -> int:
    return sum(1 for x in ast.walk(n) if isinstance(x, ast.BinOp))


def _paths(n: ast.AST, path=()):
    yield path, n
    if isinstance(n, ast.BinOp):
        yield from _paths(n.left, path + ("left",))
        yield from _paths(n.right, path + ("right",))
    elif isinstance(n, ast.UnaryOp):
        yield from _paths(n.operand, path + ("operand",))


def _replace(root: ast.AST, path: tuple, new: ast.AST) -> ast.AST:
    import copy
    if not path:
        return new
    root = copy.deepcopy(root)
    cur = root
    for p in path[:-1]:
        cur = getattr(cur, p)
    setattr(cur, path[-1], new)
    return root


def perturbations(tree: ast.AST, alternatives: Callable[[float], list[float]] | None = None) -> list[tuple[str, float]]:
    """Programmatic wrong answers, most plausible first: (kind, value). Values that fail to evaluate are skipped."""
    import copy
    out: list[tuple[str, float]] = []

    def add(kind, t):
        try:
            v = evaluate(t)
        except (ZeroDivisionError, ExprError, OverflowError):
            return
        if isinstance(v, float) and math.isfinite(v):
            out.append((kind, v))

    nodes = list(_paths(tree))
    # stopping one step early: the value of a direct operand of the root (e.g. the change instead of the % change)
    if isinstance(tree, ast.BinOp):
        for side in ("left", "right"):
            sub = getattr(tree, side)
            if isinstance(sub, (ast.BinOp, ast.UnaryOp)):
                add("stopped_early", sub)
    # wrong operation / swapped operands at each operator
    for path, n in nodes:
        if isinstance(n, ast.BinOp) and type(n.op) in _SWAP:
            m = copy.deepcopy(n); m.op = _SWAP[type(n.op)]()
            add("wrong_operation", _replace(tree, path, m))
            if type(n.op) in (ast.Sub, ast.Div):
                m2 = copy.deepcopy(n); m2.left, m2.right = m2.right, m2.left
                add("swapped_operands", _replace(tree, path, m2))
    # wrong cell (other year / other row) for each number that appears in the evidence
    if alternatives:
        for path, n in nodes:
            if isinstance(n, ast.Constant):
                for alt in alternatives(float(n.value))[:3]:
                    add("wrong_cell", _replace(tree, path, ast.Constant(alt)))
    return out


# ----------------------------------------------------------------------------------------------------- numeric options
@dataclass
class NumFormat:
    """How the dataset states the answer: unit suffix ('%', ' million'), decimals, and whether a value is a fraction of 100."""
    decimals: int = 2
    suffix: str = ""
    prefix: str = ""

    def show(self, v: float) -> str:
        r = round(v, self.decimals)
        if r == 0:
            r = 0.0
        s = f"{r:,.{self.decimals}f}"
        if self.decimals:
            s = s.rstrip("0").rstrip(".")
        return f"{self.prefix}{s}{self.suffix}"


def decimals_of(s: str) -> int:
    m = re.search(r"\.(\d+)", s)
    return len(m.group(1)) if m else 0


def parse_number(s: str) -> float | None:
    t = str(s).replace(",", "").replace("$", "").replace("%", "").strip()
    t = re.sub(r"\s*(million|billion|thousand|percent)$", "", t, flags=re.I)
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?", t)
    if not m:
        return None
    v = float(m.group(1))
    return -abs(v) if t.startswith("(") else v


def numeric_options(gold: float, candidates: Iterable[tuple[str, float]], fmt: NumFormat, seed: str,
                    n_min: int = 4, n_max: int = 6, min_rel_gap: float = 0.02,
                    max_ratio: float = 12.0) -> tuple[list[str], str, dict[str, str]] | None:
    """Gold + 3-5 distractors that are all distinct from the gold (and each other) after display rounding and differ by at
    least `min_rel_gap` relative. Falls back to sign / scale slips, then to near misses, so a well-formed item yields 4+.
    Returns (option texts, gold text, {text: kind}) or None."""
    gtext = fmt.show(gold)
    unit = 10 ** (-fmt.decimals)
    chosen: list[tuple[str, float]] = []
    kinds = {gtext: "gold"}

    def ok(v: float, structural: bool = True) -> bool:
        t = fmt.show(v)
        if t in kinds:
            return False
        if structural and gold and not (1 / max_ratio <= abs(v / gold) <= max_ratio):
            return False                      # implausible magnitude (e.g. multiplying two amounts)
        if abs(v - gold) < max(min_rel_gap * abs(gold), 2 * unit):
            return False
        return all(abs(v - c) >= max(0.01 * abs(c), unit) for _, c in chosen)

    def take(kind, v):
        if len(chosen) < n_max - 1 and ok(v, kind not in ("sign_slip", "scale_slip")):
            chosen.append((kind, v)); kinds[fmt.show(v)] = kind

    # at most two of any structural kind first, so the set is varied
    per_kind: dict[str, int] = {}
    cands = list(candidates)
    for kind, v in cands:
        if per_kind.get(kind, 0) < 2:
            before = len(chosen)
            take(kind, v)
            if len(chosen) > before:
                per_kind[kind] = per_kind.get(kind, 0) + 1
    for kind, v in cands:
        take(kind, v)
    if len(chosen) < n_min - 1:
        take("sign_slip", -gold)
        take("scale_slip", gold * 100)
        take("scale_slip", gold / 100)
        take("scale_slip", gold * 1000)
    if len(chosen) < n_min - 1:
        rng = stable_rng(seed)
        for f in (1.1, 0.9, 1.25, 0.8, 1.5):
            take("near_miss", gold * (f + rng.uniform(-0.02, 0.02)) if gold else f)
    if len(chosen) < n_min - 1:
        return None
    texts = [gtext] + [fmt.show(v) for _, v in chosen]
    return texts, gtext, kinds
