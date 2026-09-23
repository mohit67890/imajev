"""decision-v1 converter: UNK-VQA -> adversarially perturbed answerable / unanswerable choices.

UNK-VQA (TPAMI 2024, Apache-2.0) perturbs VQA v2 questions or images so the question becomes
unanswerable while staying in-distribution. Each item ships three candidate answers
(`orig`, `baseline`, `random`) and, when unanswerable, an abstention phrase plus a `reason`.

Scope restriction: three of the five perturbation types are converted - T-1 (word replacement),
T-2 (semantic negation) and I-1 (image replacement). All three reference an *unmodified* COCO
photograph by its canonical file name, so the image comes from the public COCO bucket. I-2 (image
mask) and I-3 (image copy-and-move) ship edited JPEGs ("..._mask.jpg", "..._copy.jpg") that exist
only inside the authors' Google Drive image folders, which cannot be enumerated without the Drive
API - see data/decision-v1/unkvqa/README.md.

reason_map -> abstention_cause:
  1 "It has multiple plausible answers"                  -> insufficient_evidence
  2 "It is difficult to understand"                      -> insufficient_evidence
  3 "The image lacks important concepts/information"     -> false_premise
  4 "It requires higher-level knowledge to answer"       -> insufficient_evidence

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_unkvqa.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/v1")

import _coco_shared as C  # noqa: E402
from vision_decision.contracts import Request  # noqa: E402

SOURCE = "unkvqa"
SEED = 20260922
RAW = Path(".cache/datasets/v1/unkvqa")
ROOT = Path("data/decision-v1") / SOURCE
LICENSE = ("Apache-2.0 (UNK-VQA annotations, github.com/guoyang9/UNK-VQA); "
           "images: COCO/Flickr terms, no image licence granted")

FAMILIES = {"T-1": "text_word_replacement", "T-2": "text_semantic_negation",
            "I-1": "image_replacement"}
TRAIN_TARGETS = {"T-1": 5_400, "T-2": 1_800, "I-1": 2_200}
TEST_PER_FAMILY = 200
SOURCE_ABSTAIN_RATE = 0.14      # keep the spec's 15-20% band; the source has far more available
NOT_LISTED_RATE = 0.06
MAX_PER_IMAGE = 2
DEV_FRACTION = 0.03
BIG_OPTION_RATE = 0.10
CAUSE_OF_REASON = {"1": "insufficient_evidence", "2": "insufficient_evidence",
                   "3": "false_premise", "4": "insufficient_evidence"}
NAME = re.compile(r"^COCO_(train2014|val2014)_(\d+)\.jpg$")
STOP = {"a", "an", "the", "of", "and", "or", "to", "in", "on", "is", "are", "it"}


def tokens(text: str) -> frozenset[str]:
    out = set()
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        if t not in STOP:
            out.add(t)
    return frozenset(out)


def clashes(a: str, b: str) -> bool:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return True
    return bool(ta & tb) or a in b or b in a


def prefix_of(question: str) -> str:
    return " ".join(re.findall(r"[a-z']+", question.lower())[:2])


def load(split: str) -> tuple[list[dict], dict]:
    doc = json.loads((RAW / f"annt_{split}.json").read_text())
    rows = []
    for a in doc["annotation"]:
        if a["alter_type"] not in FAMILIES:
            continue
        m = NAME.match(a["image_name"])
        if not m:          # I-2 "_mask" / I-3 "_copy" JPEGs exist only in the authors' Drive folder
            continue
        if a["alter_type"].startswith("T-") and \
                a["image_name"] != a.get("misc", {}).get("image_name_origin"):
            continue
        other = a["answerability"]["other"]
        rows.append(dict(qid=a["question_id"], question=a["question"],
                         coco_split=m.group(1), image_id=int(m.group(2)),
                         alter=a["alter_type"], answerable=bool(a["answerability"]["binary"]),
                         answer=other.get("answer"), reason=str(other.get("reason", "")),
                         options=other.get("options", {}), split=split))
    return rows, doc["answer_map"]


def main() -> None:
    rng = random.Random(SEED)
    excluded = C.exclusions()
    ledger = C.load_ledger()
    train_rows, answer_map = load("train")
    val_rows, _ = load("val")
    print(f"usable text-perturbation items: train {len(train_rows)}, val {len(val_rows)}",
          flush=True)

    by_prefix: dict[str, list[str]] = defaultdict(list)
    everything: list[str] = []
    for r in train_rows:
        for v in r["options"].values():
            if v and len(v) <= 128:
                by_prefix[prefix_of(r["question"])].append(v)
                everything.append(v)
    by_prefix = {k: [a for a, _ in Counter(v).most_common(300)] for k, v in by_prefix.items()}
    everything = [a for a, _ in Counter(everything).most_common(3000)]

    def build(r: dict, partition: str, mode: str) -> dict | None:
        """mode: 'answer' | 'source_abstain' | 'not_listed'."""
        native = [v for v in (r["options"].get("orig"), r["options"].get("baseline"),
                              r["options"].get("random")) if v and len(v) <= 128]
        if mode == "answer":
            gold = r["answer"]
            if not gold or len(gold) > 128:
                return None
            keep = [v for v in native if v != gold and not clashes(v, gold)]
            cause, target, source_answer = None, gold, gold
        elif mode == "not_listed":
            gold = r["answer"]
            if not gold or len(gold) > 128:
                return None
            keep = [v for v in native if v != gold and not clashes(v, gold)]
            cause, target, source_answer = "not_listed", None, gold
            gold = None
        else:
            keep = list(dict.fromkeys(native))
            cause = CAUSE_OF_REASON.get(r["reason"])
            if cause is None:
                return None
            target, source_answer = None, answer_map.get(str(r["answer"]), "unanswerable")
            gold = None
        rng.shuffle(keep)
        want = rng.randint(9, 21) if rng.random() < BIG_OPTION_RATE else rng.randint(0, 8)
        chosen = list(keep)
        blocked = [x for x in native if x]
        if mode in ("answer", "not_listed"):
            blocked.append(r["answer"])
        pool = list(by_prefix.get(prefix_of(r["question"]), [])) + everything
        rng.shuffle(pool)
        for cand in pool:
            if len(chosen) - len(keep) >= want:
                break
            if any(clashes(cand, b) for b in blocked if b):
                continue
            if any(clashes(cand, c) for c in chosen):
                continue
            chosen.append(cand)
        values = chosen + ([gold] if gold else [])
        values = list(dict.fromkeys(values))[:25]
        if len(values) < 2:
            return None
        rng.shuffle(values)
        field = dict(id="answer", type="choice", question=r["question"],
                     options=[dict(value=v) for v in values])
        request = dict(request_id=f"unkvqa-{r['split']}-{r['qid']}", state={}, fields=[field])
        Request.model_validate(request)
        return dict(id=f"{SOURCE}:{r['split']}-{r['qid']}", source=SOURCE, source_split=r["split"],
                    source_group=str(r["image_id"]), family=FAMILIES[r["alter"]], license=LICENSE,
                    images=[], request=request, target=target, abstention_cause=cause,
                    source_answer=source_answer, partition=partition,
                    _image_id=r["image_id"], _coco_split=r["coco_split"])

    def harvest(rows, targets, side, partition):
        quotas = {}
        for alter, n in targets.items():
            n_abst = round(n * SOURCE_ABSTAIN_RATE)
            n_hide = round(n * NOT_LISTED_RATE)
            quotas[(alter, "source_abstain")] = n_abst
            quotas[(alter, "not_listed")] = n_hide
            quotas[(alter, "answer")] = n - n_abst - n_hide
        by_image = defaultdict(list)
        for r in rows:
            if side == "fit" and r["image_id"] in excluded:
                continue
            by_image[r["image_id"]].append(r)
        order = sorted(by_image, key=lambda i: (
            0 if C.cached(f"{by_image[i][0]['coco_split']}/{i}") is not None else 1,
            hashlib.sha256(f"{SOURCE}-{side}:{i}".encode()).hexdigest()))
        out = []
        for image_id in order:
            if not any(v > 0 for v in quotas.values()):
                break
            if not C.available(ledger, image_id, side):
                continue
            cands = sorted(by_image[image_id], key=lambda r: r["qid"])
            rng.shuffle(cands)
            taken = 0
            for r in cands:
                if taken >= MAX_PER_IMAGE:
                    break
                modes = (["source_abstain"] if not r["answerable"] else ["answer", "not_listed"])
                rng.shuffle(modes)
                for mode in modes:
                    if quotas.get((r["alter"], mode), 0) <= 0:
                        continue
                    rec = build(r, partition, mode)
                    if rec is None:
                        continue
                    quotas[(r["alter"], mode)] -= 1
                    C.claim(ledger, image_id, side)
                    out.append(rec)
                    taken += 1
                    break
        return out

    fit = harvest(train_rows, TRAIN_TARGETS, "fit", "train")
    test = harvest(val_rows, {k: TEST_PER_FAMILY for k in FAMILIES}, "test", "test")
    fit_images = sorted({r["_image_id"] for r in fit},
                        key=lambda i: hashlib.sha256(f"{SOURCE}-dev:{i}".encode()).hexdigest())
    dev = set(fit_images[: max(1, round(len(fit_images) * DEV_FRACTION))])
    for r in fit:
        if r["_image_id"] in dev:
            r["partition"] = "dev"
    records = fit + test
    print(f"planned {len(records)} records over {len({r['_image_id'] for r in records})} images",
          flush=True)
    if "--plan" in sys.argv:
        print(json.dumps(dict(families=dict(Counter(r["family"] for r in records)),
                              partitions=dict(Counter(r["partition"] for r in records)),
                              abstention=dict(Counter(str(r["abstention_cause"])
                                                      for r in records)),
                              new_images=sum(C.cached(f"{r['_coco_split']}/{r['_image_id']}")
                                             is None for r in records)), indent=2))
        return
    C.save_ledger(ledger)

    keys = [f"{r['_coco_split']}/{r['_image_id']}" for r in records]
    print(f"fetching {sum(C.cached(k) is None for k in set(keys))} new COCO images ...", flush=True)
    meta = C.ensure(keys, workers=64, label=SOURCE)
    banned = C.banned_sha()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "images").mkdir(exist_ok=True)
    kept = []
    for r in records:
        key = f"{r['_coco_split']}/{r['_image_id']}"
        rec = meta.get(key)
        if rec is None:
            continue
        if rec["sha256"] in banned and r["partition"] != "test":
            continue
        r["images"] = [C.link_into(ROOT, key, rec)]
        r.pop("_image_id"), r.pop("_coco_split")
        kept.append(r)
    C.write_records(SOURCE, kept)
    print(json.dumps(dict(records=len(kept),
                          partitions=dict(Counter(r["partition"] for r in kept)),
                          families=dict(Counter(r["family"] for r in kept)),
                          abstention=dict(Counter(str(r["abstention_cause"]) for r in kept)),
                          images=len({r["images"][0]["sha256"] for r in kept})), indent=2))


if __name__ == "__main__":
    main()
