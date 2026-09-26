"""Generic builder for numeric kinds (tables, temporal, probability): gold computed from named inputs, choice options from
typical mistakes, noul thresholds clearly away from the gold, and an unknown variant that blanks one decisive input."""
from __future__ import annotations

import random
from typing import Callable

from .common import Skip, dec, flip_witness, noul_field, numeric_choice, rq, threshold_near


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def enc(v) -> str:
    return str(v)


def numeric_item(rng: random.Random, family: str, kind: str, *, inp: dict, q: dict, compute: Callable[[dict], object],
                 wrongs: Callable[[dict], list], render: Callable[[dict, set, random.Random], str], q_choice: str,
                 q_noul: Callable[[str], str], places: int, fmt: Callable, decisive: list[str], p_noul: float = 0.3,
                 min_gap: int = 2, allow_nonpos: bool = False, witness_factors=None, force_type: str | None = None,
                 perturb: Callable[[str, dict], list] | None = None) -> dict:
    val = compute(inp)
    typ = force_type or ("noul" if rng.random() < p_noul else "choice")
    spec = {"kind": kind, "inputs": {k: enc(v) for k, v in inp.items()}, "q": q, "places": places}
    if typ == "choice":
        field, gold, ov = numeric_choice(rng, q_choice, val, wrongs(inp), places, fmt, n_opts=rng.choice([4, 5, 5, 6]), min_gap=min_gap,
                                         allow_nonpos=allow_nonpos)
        spec["option_values"] = ov

        def ans(i):
            return rq(compute(i), places)
    else:
        thr, gold = threshold_near(rng, val, places)
        field = noul_field(q_noul(fmt(thr)))
        spec["threshold"] = str(thr)

        def ans(i):
            return dec(compute(i)) > thr
    rseed = rng.random()
    state = render(inp, set(), random.Random(rseed))
    unk = None
    keys = [k for k in decisive if k in inp]
    rng.shuffle(keys)
    for k in keys:
        if perturb is not None:
            wit, base = None, ans(inp)
            for alt in perturb(k, inp):
                try:
                    if ans(alt) != base:
                        wit = [enc(inp[k]), enc(alt[k])]
                        walt = {kk: enc(alt[kk]) for kk in alt if alt[kk] != inp[kk]}
                        break
                except (Skip, ValueError, ZeroDivisionError):
                    continue
            if wit is None:
                continue
        else:
            try:
                wit = flip_witness(rng, inp, k, ans, witness_factors)
            except Skip:
                continue
            walt = {k: wit[1]}
        unk = {"state": render(inp, {k}, random.Random(rseed)), "reason": "insufficient_evidence",
               "spec": {**spec, "removed": k, "witness": wit, "witness_alt": walt}}
        break
    return {"family": family, "state": state, "field": field, "gold": gold, "unknown_reason": None, "spec": spec, "unk": unk}


def jenc(v):
    """JSON-friendly encoding of an input value (dates -> ISO, Fractions -> 'a/b', containers recursively)."""
    import datetime as _dt
    from fractions import Fraction as _F
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()
    if isinstance(v, _F):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [jenc(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jenc(x) for k, x in v.items()}
    return v


def value_item(rng: random.Random, family: str, kind: str, *, inp: dict, q: dict, compute: Callable[[dict], object],
               wrongs: Callable[[dict], list], render: Callable[[dict, set, random.Random], str], q_choice: str,
               fmt: Callable[[object], str], decisive: list[str], perturb: Callable[[str, dict], list],
               noul: Callable[[random.Random, object], tuple[str, object]] | None = None,
               judge: Callable[[object, object], bool] | None = None, p_noul: float = 0.3,
               fill: Callable[[int], object] | None = None, n_opts: int | None = None, force_type: str | None = None) -> dict:
    """Choice/noul item over arbitrary answer values (dates, times, integers). Options are distinct after formatting.
    perturb(key, inp) yields alternative input dicts in which `key` takes another value consistent with the redacted state."""
    from .common import choice_field, noul_field
    val = compute(inp)
    typ = force_type or ("noul" if (noul is not None and rng.random() < p_noul) else "choice")
    spec = {"kind": kind, "inputs": jenc(inp), "q": jenc(q)}
    if typ == "choice":
        n_opts = n_opts or rng.choice([4, 5, 5, 6])
        gtxt = fmt(val)
        texts, vals = [gtxt], {gtxt: jenc(val)}
        cands = list(wrongs(inp))
        k = 1
        while fill is not None and k < 40:
            cands.append(fill(k))
            k += 1
        for w in cands:
            if w is None or len(texts) >= n_opts:
                continue
            t = fmt(w)
            if t in texts or w == val:
                continue
            texts.append(t)
            vals[t] = jenc(w)
        if len(texts) < 3:
            raise Skip("few options")
        field, gold, ov = choice_field(rng, q_choice, gtxt, texts[1:], values=vals)
        spec["option_values"] = ov

        def ans(i):
            return fmt(compute(i))
    else:
        qtext, probe = noul(rng, val)
        gold = bool(judge(val, probe))
        field = noul_field(qtext)
        spec["probe"] = jenc(probe)

        def ans(i):
            return bool(judge(compute(i), probe))
    rseed = rng.random()
    state = render(inp, set(), random.Random(rseed))
    unk = None
    keys = [k for k in decisive if k in inp]
    rng.shuffle(keys)
    base = ans(inp)
    for k in keys:
        wit = None
        for alt in perturb(k, inp):
            try:
                if ans(alt) != base:
                    wit = [jenc(inp[k]), jenc(alt[k])]
                    walt = {kk: jenc(alt[kk]) for kk in alt if alt[kk] != inp[kk]}
                    break
            except (Skip, ValueError, ZeroDivisionError, OverflowError):
                continue
        if wit:
            unk = {"state": render(inp, {k}, random.Random(rseed)), "reason": "insufficient_evidence",
                   "spec": {**spec, "removed": k, "witness": wit, "witness_alt": walt}}
            break
    return {"family": family, "state": state, "field": field, "gold": gold, "unknown_reason": None, "spec": spec, "unk": unk}

