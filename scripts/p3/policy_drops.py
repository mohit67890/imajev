"""Phase-3 review-driven drop rules (lead decisions 2026-09-26, reports/phase3/review.md + recovered-adjudication.md).

Applied to every training source; writes data/p3/fixes/drop-ids-policy.json ({ids, why, by_rule}) for build_manifest.py
--drop-ids. Rules (each from a confirmed review finding):
  D_UNK_AGENT_ROUTING  D (trap) rows with unknown gold whose parent family is agent_action or routing_hard (review: 5/5 and 4/7
                       checked D disagreements in these families were our errors: lost-tool-response and decidable negations)
  D_NEG_UNK_PARENT     D negation traps built on an unknown parent (id ...-u): the answer is often decidable (both reviewers)
  DOCIMG_BALANCE_UNK   docimg balance_check unknowns: the running-balance column restates the covered opening/closing balance
  WRITER_UNK_TABLE     writer "unknown" variants (-wu) of table / arithmetic parents: the masked number is often derivable from
                       totals or row identities (review A: tatqa, finqa, docimg)
  C_SUBJECTIVE         C (own earlier pools) subjective families: safety/moderation, sentiment/emotion, SNLI, rubric/adequacy,
                       photo quality, style (review: most C ambiguity; Kimi 50-100% disagreement)
  WRITER_LEAK          writer rows that copied "VERIFIED ANSWER" into the item
  ORDINAL_BEING_TESTED ordinal unknowns whose marker "[being tested]" reads as a real state, not missing data
"""
import argparse, glob, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C_SUBJ = {"content_safety", "toxicity_screening", "moderation_taxonomy", "sentiment", "emotion", "emotion_curiosity", "snli",
          "response_rubric", "adequacy_of_answer", "photo_quality", "style", "intent", "stance"}
TABLE_FAM = re.compile(r"table|arithmetic|numerical|numeric_comparison|finance", re.I)
TABLE_DS = re.compile(r"tatqa|finqa|hybridqa|convfinqa", re.I)
FILES = ["data/p3/teacher/kept.jsonl", "data/p3/teacher-pod/kept.jsonl", "data/p3/variants/constructed.jsonl",
         "data/p3/variants-pod/constructed.jsonl", "data/p3/labeltrain/direct-constructed-leftovers.jsonl",
         "data/p3/labeltrain/direct-constructed-leftovers-late.jsonl", "data/p3/fixes/direct-recovered-rejected.jsonl",
         *sorted(glob.glob(str(ROOT / "data/p3/candidates-clean/I-*.jsonl"))), *sorted(glob.glob(str(ROOT / "data/p3/candidates-clean/D-*.jsonl")))]


def item_of(r):
    return r.get("item") if isinstance(r.get("item"), dict) else r


def gold_of(r, it):
    lab = r.get("label") or {}
    if "target" in lab:
        return lab["target"]
    return it.get("gold")


def rules(r):
    it = item_of(r); iid = it.get("id") or r.get("id") or ""
    src, fam, ds = it.get("source"), it.get("family") or "", it.get("dataset") or ""
    g = gold_of(r, it)
    unk = g is None or g == "unknown"
    out = []
    blob = json.dumps(it, ensure_ascii=False)
    if src == "D" and unk and re.search(r"-(agnt|rout)-", iid):
        out.append("D_UNK_AGENT_ROUTING")
    if src == "D" and "trap-negation" in iid and re.search(r"-u($|-)", iid.split("trap-negation-", 1)[-1]):
        out.append("D_NEG_UNK_PARENT")
    if "docimg" in iid and "balance" in (fam + " " + str(it.get("provenance", {}).get("template", "")) + " " + blob[:0]) and unk:
        out.append("DOCIMG_BALANCE_UNK")
    if "docimg" in iid and unk and "balance_check" in blob:
        out.append("DOCIMG_BALANCE_UNK")
    if re.search(r"-wu\d*$", iid) and (TABLE_FAM.search(fam) or TABLE_DS.search(ds) or TABLE_DS.search(iid)):
        out.append("WRITER_UNK_TABLE")
    if src == "C" and fam in C_SUBJ:
        out.append("C_SUBJECTIVE")
    if "VERIFIED ANSWER" in blob.upper():
        out.append("WRITER_LEAK")
    if unk and "[being tested]" in blob:
        out.append("ORDINAL_BEING_TESTED")
    return sorted(set(out)), iid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "data/p3/fixes/drop-ids-policy.json"))
    a = ap.parse_args()
    ids, by_rule, per_file = set(), Counter(), {}
    for f in FILES:
        p = ROOT / f
        if not p.exists():
            continue
        n = hit = 0
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("keep") is False:
                continue
            n += 1
            rs, iid = rules(r)
            if rs:
                hit += 1; ids.add(iid); by_rule.update(rs)
        per_file[f] = {"rows": n, "dropped": hit}
    Path(a.out).write_text(json.dumps({"ids": sorted(ids), "why": __doc__.split("Rules")[0].strip(), "by_rule": dict(by_rule),
                                       "per_file": per_file}, indent=0) + "\n")
    print(json.dumps({"ids": len(ids), "by_rule": dict(by_rule)}, indent=1))
    for f, v in per_file.items():
        print(f"{f:70} {v['rows']:7} -{v['dropped']}")


if __name__ == "__main__":
    main()
