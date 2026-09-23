"""heldout_abstention: MM-UPD + TUBench -> decision-v1 records, every record partition "test".

Two independently-built abstention benchmarks, each mapped to the cause its own construction
proves, plus their answerable controls so over-abstention is measurable:

  MM-UPD AAD   the gold option was deleted from an otherwise normal MMBench question -> not_listed
  MM-UPD IASD  the whole option set was swapped for an unrelated one -> not_listed
  MM-UPD IVQD  the image was swapped, so the question presupposes content the image lacks
               -> false_premise
  MM-UPD standard  the unmodified MMBench questions, paired by index with the AAD items -> answerable
  TUBench UCR/UVQA  yes/no questions whose answer is missing from the image -> insufficient_evidence
  TUBench UCR/UVQA answerable twins, yes/no balanced -> answerable

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_heldout_abstention.py
"""
import base64
import hashlib
import io
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts/v1")
from _common_textrich import CACHE, ImageStore, make_record, native_choice, norm, write  # noqa: E402

SOURCE = "heldout_abstention"
SEED = 20260922
MAX_OPTION_CHARS = 128
QUOTA = {"aad_pairs": 275, "iasd": 250, "ivqd": 250, "tubench_unanswerable": 225, "tubench_answerable": 225}
MAX_PER_IMAGE = 3
LICENSE = "Apache-2.0 (MM-UPD annotations) / CC-BY-4.0 (TUBench annotations); underlying images per upstream"

UPD = CACHE / SOURCE / "mmupd" / "data"
TUB = CACHE / SOURCE / "tubench"
OPTION_RE = re.compile(r"\(([a-z])\)\s*")


def order(items, salt):
    """Deterministic shuffle that does not depend on file or dict ordering."""
    return sorted(items, key=lambda x: hashlib.sha256(f"{SEED}:{salt}:{x[0]}".encode()).hexdigest())


def options_of(row):
    out = []
    for letter in "ABCDE":
        v = row.get(letter)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        v = str(v).strip()
        if not v or len(v) > MAX_OPTION_CHARS:
            return None
        out.append(v)
    return out if len(out) >= 2 else None


def question_of(row):
    q = str(row["question"]).strip()
    hint = row.get("hint")
    if hint is not None and not (isinstance(hint, float) and pd.isna(hint)):
        q = f"{str(hint).strip()} {q}"
    return q[:2000]


def load_upd(name):
    df = pd.read_csv(UPD / f"{name}.tsv", sep="\t")
    rows = {}
    for r in df.to_dict("records"):
        rows.setdefault(int(r["index"]), r)  # circular-shift duplicates: keep the first ordering
    return rows


def main():
    rng = random.Random(SEED)
    store = ImageStore(SOURCE)
    records, per_image = [], defaultdict(int)

    def room(image_rec, n=1):
        return per_image[image_rec["sha256"]] + n <= MAX_PER_IMAGE

    def emit(uid, split, family, image_rec, field, target, cause, answer):
        if not room(image_rec):
            return False
        per_image[image_rec["sha256"]] += 1
        records.append(make_record(
            source=SOURCE, uid=uid, source_split=split, source_group=image_rec["sha256"],
            family=family, license=LICENSE, images=[image_rec], field=field, target=target,
            cause=cause, source_answer=answer, partition="test"))
        return True

    # ---------------------------------------------------------------- MM-UPD
    aad = load_upd("mmaad_aad_20240303_base")
    std = load_upd("mmaad_standard_20240303_base")
    iasd = load_upd("mmiasd_iasd_20240303_base")
    ivqd = load_upd("mmivqd_ivqd_20240303_base")

    def upd_image(row, key):
        return store.add(base64.b64decode(str(row["image"])), key)

    # AAD paired with its unmodified control: same question, gold present vs deleted.
    pairs = [i for i in aad if i in std]
    kept = 0
    for idx in order([(i,) for i in pairs], "aad"):
        idx = idx[0]
        if kept >= QUOTA["aad_pairs"]:
            break
        a, s = aad[idx], std[idx]
        a_opts, s_opts = options_of(a), options_of(s)
        letter = str(s.get("answer", "")).strip()
        if not a_opts or not s_opts or letter not in ("A", "B", "C", "D", "E"):
            continue
        gold_text = str(s.get(letter, "")).strip()
        masked = str(a.get("masked_answer", "")).strip()
        if gold_text not in s_opts or not masked or masked.lower() == "nan":
            continue
        if any(norm(masked) == norm(o) for o in a_opts):
            continue  # the "deleted" answer is still on the list: not a clean not_listed item
        built = native_choice(rng, question_of(a), a_opts, None)
        built_ctl = native_choice(rng, question_of(s), s_opts, gold_text)
        if not built or not built_ctl:
            continue
        img = upd_image(a, f"upd-aad-{idx}")
        if not room(img, 2):
            continue  # keep the pair atomic: an AAD item is only useful beside its control
        f1, t1 = built
        f2, t2 = built_ctl
        emit(f"mmupd-aad-{idx}", "mmupd/aad", "answerability_option_absent", img, f1, t1, "not_listed", masked)
        emit(f"mmupd-std-{idx}", "mmupd/standard", "answerable_control_mc", img, f2, t2, None, gold_text)
        kept += 1

    for name, quota, cause, family in (
        ("iasd", QUOTA["iasd"], "not_listed", "answerability_option_set_incompatible"),
        ("ivqd", QUOTA["ivqd"], "false_premise", "answerability_image_mismatch"),
    ):
        table = iasd if name == "iasd" else ivqd
        kept = 0
        for (idx,) in order([(i,) for i in table], name):
            if kept >= quota:
                break
            row = table[idx]
            opts = options_of(row)
            if not opts:
                continue
            masked = str(row.get("masked_answer", "")).strip()
            has_masked = bool(masked) and masked.lower() != "nan"
            if cause == "not_listed":
                if not has_masked or any(norm(masked) == norm(o) for o in opts):
                    continue
                answer = masked
            else:
                answer = "unanswerable: the image does not match the question (MM-UPD IVQD)"
            built = native_choice(rng, question_of(row), opts, None)
            if not built:
                continue
            img = upd_image(row, f"upd-{name}-{idx}")
            field, target = built
            if emit(f"mmupd-{name}-{idx}", f"mmupd/{name}", family, img, field, target, cause, answer):
                kept += 1

    # ---------------------------------------------------------------- TUBench
    # UCR and UVQA are yes/no/Unanswerable -> boolean; UTabMWP ships (a)(b)(c)(d) options -> choice.
    # UGeoQA is Chinese-language and is left out of this English-only mixture.
    tub = []
    for meta in sorted(TUB.glob("*/test/metadata.jsonl")):
        if meta.parent.parent.name == "UGeoQA":
            continue
        for line in meta.read_text().splitlines():
            r = json.loads(line)
            r["_path"] = meta.parent / r["file_name"]
            tub.append(r)

    def tub_key(r):
        return f'{r["dataset_name"]}/{r["file_name"]}/{r["question"]}'

    def tub_image(r):
        return store.add(r["_path"].read_bytes(), f'tub-{r["dataset_name"]}-{r["file_name"]}')

    bool_rows = [r for r in tub if not str(r.get("choices", "")).strip()
                 and str(r["answer"]).strip() in ("Yes", "No", "Unanswerable")]
    mc_rows = []
    for r in tub:
        raw = str(r.get("choices", "")).strip()
        if not raw:
            continue
        marks = list(OPTION_RE.finditer(raw))
        letters = [m.group(1) for m in marks]
        opts = [raw[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(raw))].strip()
                for i, m in enumerate(marks)]
        if len(opts) < 2 or not all(opts):
            continue
        a = str(r["answer"]).strip().strip("()")
        if a != "Unanswerable" and a not in letters:
            continue
        r["_options"] = opts
        r["_gold"] = None if a == "Unanswerable" else opts[letters.index(a)]
        mc_rows.append(r)

    kept = {"bool": 0, "mc": 0}
    half = QUOTA["tubench_unanswerable"] // 3
    for (_, r) in order([(tub_key(r), r) for r in bool_rows if r["answer"] == "Unanswerable"], "tub-u-bool"):
        if kept["bool"] >= QUOTA["tubench_unanswerable"] - half or not r["_path"].is_file():
            continue
        field = {"id": "answer", "type": "boolean", "question": str(r["question"])[:2000]}
        uid = "tubench-u-" + hashlib.sha256(tub_key(r).encode()).hexdigest()[:16]
        if emit(uid, f'tubench/{r["dataset_name"]}', "answerability_missing_information",
                tub_image(r), field, None, "insufficient_evidence", "Unanswerable"):
            kept["bool"] += 1
    for (_, r) in order([(tub_key(r), r) for r in mc_rows if r["_gold"] is None], "tub-u-mc"):
        if kept["mc"] >= half or not r["_path"].is_file():
            continue
        built = native_choice(rng, str(r["question"])[:2000], r["_options"], None)
        if not built:
            continue
        field, target = built
        uid = "tubench-u-" + hashlib.sha256(tub_key(r).encode()).hexdigest()[:16]
        if emit(uid, f'tubench/{r["dataset_name"]}', "answerability_missing_information",
                tub_image(r), field, target, "insufficient_evidence", "Unanswerable"):
            kept["mc"] += 1

    # Answerable controls: yes/no balanced for the boolean half, native gold for the choice half.
    half = QUOTA["tubench_answerable"] // 3
    want = {"Yes": (QUOTA["tubench_answerable"] - half) // 2}
    want["No"] = QUOTA["tubench_answerable"] - half - want["Yes"]
    got = {"Yes": 0, "No": 0}
    for (_, r) in order([(tub_key(r), r) for r in bool_rows if r["answer"] in ("Yes", "No")], "tub-a-bool"):
        a = str(r["answer"]).strip()
        if got[a] >= want[a] or not r["_path"].is_file():
            continue
        field = {"id": "answer", "type": "boolean", "question": str(r["question"])[:2000]}
        uid = "tubench-a-" + hashlib.sha256(tub_key(r).encode()).hexdigest()[:16]
        if emit(uid, f'tubench/{r["dataset_name"]}', "answerable_control_boolean",
                tub_image(r), field, a == "Yes", None, a):
            got[a] += 1
    kept["mc"] = 0
    for (_, r) in order([(tub_key(r), r) for r in mc_rows if r["_gold"] is not None], "tub-a-mc"):
        if kept["mc"] >= half or not r["_path"].is_file():
            continue
        built = native_choice(rng, str(r["question"])[:2000], r["_options"], r["_gold"])
        if not built:
            continue
        field, target = built
        uid = "tubench-a-" + hashlib.sha256(tub_key(r).encode()).hexdigest()[:16]
        if emit(uid, f'tubench/{r["dataset_name"]}', "answerable_control_mc",
                tub_image(r), field, target, None, r["_gold"]):
            kept["mc"] += 1

    write(SOURCE, records)


if __name__ == "__main__":
    main()
