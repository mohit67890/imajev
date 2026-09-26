"""MuSiQue (CC BY 4.0, https://github.com/StonyBrookNLP/musique) MuSiQue-Full *train* -> phase-3 candidates (Stage 0 source B).

MuSiQue-Full pairs every question with a contrastive twin whose paragraph set lacks a supporting paragraph. Both become
choice items over the same 5 options: the answerable one with the dataset answer as gold, the unanswerable one with
gold = null / unknown_reason "insufficient_evidence" (its ``parent_id`` is the answerable twin, so both land in one split).

Options = the gold answer + distractors, most confusable first:
  1. intermediate hop answers (stopping at the wrong hop),
  2. answers to other train questions with the same final-hop relation that also occur in this item's passages,
  3. other answers with the same final-hop relation,
  4. (years / numbers) other years or numbers found in the passages.
A distractor is dropped when it equals, contains or is contained in the gold answer or any of its aliases.

Round 1 keeps all 3- and 4-hop pairs and a fixed random sample of 2-hop pairs.

    .venv/bin/python scripts/p3/convert_musique.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from candidate import write  # noqa: E402
from convert_common import MAX_STATE_TOKENS, approx_tokens, build_choice_options, norm_answer, stable_rng  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/p3/raw/musique/musique_full_v1.0_train.jsonl"
OUT = ROOT / "data/p3/candidates/B-musique.jsonl"
URL = "https://github.com/StonyBrookNLP/musique"
LICENCE = "CC-BY-4.0"
N_OPTIONS = 5
TWO_HOP_PAIRS = 4000


def hops(qid: str) -> int:
    return int(qid[0])


def relation_key(item: dict) -> str:
    """The last decomposition step with the entity placeholders removed: 'The #1 was named for whom?' -> 'was named for whom'."""
    last = item["question_decomposition"][-1]["question"]
    if ">>" in last:
        return "rel:" + last.split(">>", 1)[1].strip().lower()
    return "q:" + re.sub(r"#\d+|[^a-z ]", " ", last.lower()).split("?")[0].strip()[-60:].strip()


_WH = re.compile(r"\b(who|whom|whose|when|where|what|which|how many|how much|how long|how old|why|how)\b")


def answer_shape(a: str) -> str:
    a = a.strip()
    if _YEAR.fullmatch(a):
        return "year"
    if _NUMBER.fullmatch(a):
        return "number"
    if re.search(r"\d", a):
        return "has_digit"
    words = a.split()
    if len(words) >= 2 and all(w[:1].isupper() for w in words if w.isalpha()):
        return "proper_multi"
    return "proper" if a[:1].isupper() else "lower"


def coarse_key(item: dict) -> str:
    """Coarser answer class for the fallback pool: wh-word of the last hop + the shape of the answer."""
    last = item["question_decomposition"][-1]["question"].lower()
    m = _WH.search(last)
    wh = m.group(1) if m else ("rel:" + last.split(">>", 1)[1].strip() if ">>" in last else "other")
    return f"{wh}|{answer_shape(item['answer'])}"


def clash(a: str, golds: list[str]) -> bool:
    na = norm_answer(a)
    if not na:
        return True
    for g in golds:
        ng = norm_answer(g)
        if not ng or na == ng or (len(na) > 2 and na in ng) or (len(ng) > 2 and ng in na):
            return True
    return False


_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def distractors(item: dict, by_relation: dict[str, list[str]], rng, by_coarse: dict[str, list[str]] | None = None
                ) -> list[tuple[str, str]]:
    gold = item["answer"]
    golds = [gold] + list(item.get("answer_aliases") or [])
    text = " ".join(p["paragraph_text"] for p in item["paragraphs"])
    out: list[tuple[str, str]] = []

    def add(kind, cand):
        cand = cand.strip()
        if cand and len(cand) <= 80 and not clash(cand, golds) and not any(norm_answer(cand) == norm_answer(c) for _, c in out):
            out.append((kind, cand))

    for step in item["question_decomposition"][:-1]:
        add("intermediate_hop", step["answer"])
    pool = [a for a in by_relation.get(relation_key(item), []) if a != gold]
    in_text = [a for a in pool if len(a) > 2 and a in text]
    for a in rng.sample(in_text, min(len(in_text), 8)):
        add("same_relation_in_passages", a)
    for a in rng.sample(pool, min(len(pool), 12)):
        add("same_relation", a)
    if _YEAR.fullmatch(gold.strip()):
        years = sorted(set(_YEAR.findall(text)))
        for y in rng.sample(years, min(len(years), 6)):
            add("year_in_passages", y)
    elif _NUMBER.fullmatch(gold.strip()):
        nums = sorted(set(_NUMBER.findall(text)))
        for y in rng.sample(nums, min(len(nums), 6)):
            add("number_in_passages", y)
    if len(out) < N_OPTIONS - 1 and by_coarse is not None:
        coarse = [a for a in by_coarse.get(coarse_key(item), []) if a != gold]
        for a in rng.sample(coarse, min(len(coarse), 12)):
            add("same_answer_class", a)
    if len(out) < N_OPTIONS - 1 and answer_shape(gold) in ("proper", "proper_multi"):
        titles = sorted({p["title"] for p in item["paragraphs"] if not p.get("is_supporting")})
        for t in rng.sample(titles, min(len(titles), 6)):
            add("passage_title", t)
    # keep the most confusable first, up to N_OPTIONS - 1
    order = {"intermediate_hop": 0, "same_relation_in_passages": 1, "year_in_passages": 1, "number_in_passages": 1,
             "same_relation": 2, "same_answer_class": 3, "passage_title": 4}
    out.sort(key=lambda kv: order[kv[0]])
    return out[: N_OPTIONS - 1]


def render_state(item: dict) -> str:
    lines = ["Passages:"]
    for i, p in enumerate(item["paragraphs"], start=1):
        lines.append(f"[{i}] {p['title']}\n{p['paragraph_text'].strip()}")
    return "\n\n".join(lines)


def difficulty(n_hops: int, answerable: bool) -> int:
    return max(1, min(5, n_hops + (0 if answerable else 1)))


def convert(raw: Path = RAW, out: Path = OUT, two_hop_pairs: int = TWO_HOP_PAIRS) -> dict:
    items = [json.loads(line) for line in open(raw)]
    by_relation: dict[str, list[str]] = defaultdict(list)
    by_coarse: dict[str, list[str]] = defaultdict(list)
    for it in items:
        if it["answerable"]:
            by_relation[relation_key(it)].append(it["answer"])
            by_coarse[coarse_key(it)].append(it["answer"])
    pairs: dict[str, dict] = defaultdict(dict)
    for it in items:
        pairs[it["id"]]["ans" if it["answerable"] else "unans"] = it
    ids = sorted(pairs)
    two = [i for i in ids if hops(i) == 2]
    keep2 = set(stable_rng("musique-2hop").sample(two, min(two_hop_pairs, len(two))))
    rows, why = [], defaultdict(int)
    for qid in ids:
        pr = pairs[qid]
        if "ans" not in pr or "unans" not in pr:
            why["unpaired"] += 1
            continue
        if hops(qid) == 2 and qid not in keep2:
            why["2hop_not_sampled"] += 1
            continue
        a = pr["ans"]
        ds = distractors(a, by_relation, stable_rng(qid), by_coarse)
        if len(ds) < 3:
            why["few_distractors"] += 1
            continue
        texts = [a["answer"]] + [c for _, c in ds]
        kinds = {a["answer"]: "gold", **{c: k for k, c in ds}}
        opts, gkey, kind_of = build_choice_options(texts, a["answer"], seed=qid, kinds=kinds)
        question = f"Based only on the passages, what is the answer to this question: {a['question'].strip()}"
        parent = None
        states = {"answerable": render_state(a), "unanswerable": render_state(pr["unans"])}
        if max(approx_tokens(s) for s in states.values()) > MAX_STATE_TOKENS:
            why["too_long"] += 1
            continue
        for variant, it in (("answerable", a), ("unanswerable", pr["unans"])):
            state = states[variant]
            rid = f"p3-musique-{len(rows):06d}"
            ans = variant == "answerable"
            rows.append({"id": rid, "source": "B", "dataset": "musique",
                         "family": "multi_hop_qa" if ans else "multi_hop_unanswerable",
                         "difficulty": difficulty(hops(qid), ans), "state": state, "images": [],
                         "field": {"type": "choice", "question": question, "options": opts},
                         "gold": gkey if ans else None, "unknown_reason": None if ans else "insufficient_evidence",
                         "gold_kind": "dataset", "parent_id": None if ans else parent,
                         "provenance": {"licence": LICENCE, "upstream_dataset": "musique", "upstream_split": "train",
                                        "upstream_id": qid, "group_id": qid, "upstream_variant": f"musique_full:{variant}", "url": URL,
                                        "hops": hops(qid), "option_kinds": kind_of,
                                        "note": "passages are Wikipedia-derived (see reports/phase3/licences.md)"}})
            if ans:
                parent = rid
            why[variant] += 1
    n = write(out, rows)
    return {"read_pairs": len(ids), "written": n, "reasons": dict(why)}


if __name__ == "__main__":
    print(json.dumps(convert(), indent=1))
