"""StrategyQA (MIT, https://github.com/eladsegal/strategyqa) official train set -> phase-3 noul candidates (Stage 0 source B).

Design decision (documented in reports/phase3/public-datasets.md): StrategyQA questions are implicit multi-step questions whose
steps often need world knowledge ("operation" / "no_evidence" steps), so the item cannot be framed as "from the passages
only". The state is the union of the annotators' evidence paragraphs *mixed with* lexically similar distractor paragraphs from
other questions' evidence, shuffled, and the question says general knowledge is allowed. The dataset's ``facts`` and
``decomposition`` are never shown (they would hand over the reasoning). Questions with a single decomposition step are dropped.

    .venv/bin/python scripts/p3/convert_strategyqa.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from candidate import write  # noqa: E402
from convert_common import MAX_STATE_TOKENS, approx_tokens, stable_shuffle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/p3/raw/strategyqa"
OUT = ROOT / "data/p3/candidates/B-strategyqa.jsonl"
URL = "https://github.com/eladsegal/strategyqa"
LICENCE = "MIT"
MAX_EVIDENCE = 8
N_DISTRACTORS = 4
MAX_PARA_CHARS = 1500
_STOP = set("a an the of in on at to for by with from and or is are was were be been do does did can could would should will "
            "has have had it its this that these those as than then there their they he she his her you your i we our not no "
            "yes what which who whom whose when where why how many much more most any some all ever".split())


def words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2]


def evidence_ids(item: dict) -> list[str]:
    ids: list[str] = []
    for ann in item.get("evidence", []):
        for step in ann:
            for e in step:
                if isinstance(e, list):
                    for pid in e:
                        if pid not in ids:
                            ids.append(pid)
    return ids


class ParaIndex:
    """Tiny inverted index over the evidence paragraphs for picking lexically related distractors."""

    def __init__(self, paras: dict[str, dict]):
        self.paras = paras
        self.inv: dict[str, set[str]] = defaultdict(set)
        for pid, p in paras.items():
            for w in set(words(p["title"] + " " + p["content"][:600])):
                self.inv[w].add(pid)
        self.df = {w: len(s) for w, s in self.inv.items()}

    def related(self, query: str, exclude: set[str], k: int) -> list[str]:
        score: Counter = Counter()
        for w in set(words(query)):
            if w in self.inv and self.df[w] < 400:
                for pid in self.inv[w]:
                    score[pid] += 1.0 / (1 + self.df[w]) ** 0.5
        out = []
        ex_titles = {self.paras[e]["title"] for e in exclude if e in self.paras}
        for pid, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0])):
            if pid in exclude or self.paras[pid]["title"] in ex_titles:
                continue
            out.append(pid)
            if len(out) == k:
                break
        return out


def difficulty(n_steps: int, n_evidence: int) -> int:
    d = {2: 3, 3: 3, 4: 4}.get(n_steps, 5 if n_steps >= 5 else 2)
    if n_evidence == 0:
        d += 1          # nothing to read: pure world knowledge
    return max(1, min(5, d))


def render_state(pids: list[str], paras: dict[str, dict]) -> str:
    if not pids:
        return "No passages are provided for this question."
    lines = ["Passages (some may be irrelevant):"]
    for i, pid in enumerate(pids, start=1):
        p = paras[pid]
        text = p["content"].strip()
        if len(text) > MAX_PARA_CHARS:
            text = text[:MAX_PARA_CHARS].rsplit(" ", 1)[0] + " ..."
        lines.append(f"[{i}] {p['title']}\n{text}")
    return "\n\n".join(lines)


def convert(raw_dir: Path = RAW, out: Path = OUT) -> dict:
    items = json.load(open(raw_dir / "strategyqa_train.json"))
    paras = json.load(open(raw_dir / "strategyqa_train_paragraphs.json"))
    index = ParaIndex(paras)
    rows, why = [], Counter()
    for it in items:
        n_steps = len(it.get("decomposition") or [])
        if n_steps < 2:
            why["one_step"] += 1
            continue
        ev = [p for p in evidence_ids(it) if p in paras][:MAX_EVIDENCE]
        dis = index.related(it["question"] + " " + it.get("term", ""), set(ev), N_DISTRACTORS) if ev else []
        pids = stable_shuffle(ev + dis, it["qid"])
        state = render_state(pids, paras)
        if approx_tokens(state) > MAX_STATE_TOKENS:
            why["too_long"] += 1
            continue
        rows.append({"id": f"p3-strategyqa-{len(rows):06d}", "source": "B", "dataset": "strategyqa",
                     "family": "implicit_multi_hop", "difficulty": difficulty(n_steps, len(ev)), "state": state, "images": [],
                     "field": {"type": "noul", "question": f"{it['question'].strip()} (The passages may help; general knowledge "
                                                         f"is allowed.)"},
                     "gold": bool(it["answer"]), "unknown_reason": None, "gold_kind": "dataset", "parent_id": None,
                     "provenance": {"licence": LICENCE, "upstream_dataset": "strategyqa", "upstream_split": "train",
                                    "upstream_id": it["qid"], "group_id": it["qid"], "url": URL, "n_steps": n_steps, "n_evidence": len(ev),
                                    "n_distractor_paragraphs": len(dis),
                                    "note": "paragraphs are Wikipedia text (see reports/phase3/licences.md)"}})
        why["ok"] += 1
    n = write(out, rows)
    return {"read": len(items), "written": n, "reasons": dict(why)}


if __name__ == "__main__":
    print(json.dumps(convert(), indent=1))
