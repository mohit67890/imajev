"""Phase-3 source-A policy generators (scripts/p3/gen_policy.py + gen_policy_families/) and source-D traps (scripts/p3/gen_traps.py).

Checks: row validity, determinism, independent gold re-derivation (each module's `recheck`, a separate code path from the generator),
unknown variants are truly unanswerable (long_policy: re-derived with the removed fact set to different plausible values), trap gold
correctness, long-document word ranges, long-input token ranges, multi-label grouping, and Stage-2 variants.
"""
import copy
import datetime as dt
import json
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/p3"))
from candidate import validate, write  # noqa: E402
import gen_policy as gp  # noqa: E402
import gen_traps as gt  # noqa: E402
from gen_policy_families.common import approx_tokens, words  # noqa: E402

FAMS = gp.FAMILY_NAMES
N_PER_FAM = {"long_policy": 60, "rule_exception": 80, "judge_hard": 80, "agent_action": 80, "long_input": 12, "multi_label": 20}


@pytest.fixture(scope="module")
def sample():
    rows = {}
    for fam in FAMS:
        out = []
        for i in range(N_PER_FAM[fam]):
            out.extend(gp.gen_one(fam, 123, i))
        rows[fam] = out
    return rows


def module_for(row):
    for fam in FAMS:
        mod = gp.MODULES[fam]
        if row["family"] in getattr(mod, "FAMILIES", (fam,)):
            return mod
    raise KeyError(row["family"])


def recheck_matches(row) -> bool:
    mod = module_for(row)
    v = mod.recheck(row["provenance"]["spec"])
    gold, f = row["gold"], row["field"]
    if gold is None:
        return v is None
    if f["type"] in ("noul", "score"):
        return v == gold and type(v) is type(gold)
    spec = row["provenance"]["spec"]
    if spec.get("opt_values"):
        return spec["opt_values"][gold] == v
    if isinstance(v, int) and not isinstance(v, bool):
        return 0 <= v < len(f["options"]) and f["options"][v]["key"] == gold
    if isinstance(v, str):
        gt_ = next(o["text"] for o in f["options"] if o["key"] == gold)
        return gt_.strip().lower() == v.strip().lower()
    return False


# ------------------------------------------------------------------------------------------------------------ validity
def test_rows_are_valid_and_labelled(sample):
    ids = Counter()
    for fam, rows in sample.items():
        assert rows, fam
        for r in rows:
            assert validate(r) == [], (r["id"], validate(r))
            assert r["source"] == "A" and r["dataset"] == "gen_policy" and r["gold_kind"] == "constructed"
            assert r["provenance"]["licence"] == "generated"
            assert 3 <= r["difficulty"] <= 5
            assert r["images"] == []
            ids[r["id"]] += 1
            f = r["field"]
            if f["type"] == "choice":
                assert 2 <= len(f["options"]) <= 254
                assert len({o["text"].strip().lower() for o in f["options"]}) == len(f["options"]), r["id"]
            json.dumps(r)  # serialisable
    assert max(ids.values()) == 1


def test_families_and_kinds_covered(sample):
    fams = Counter(r["family"] for rows in sample.values() for r in rows)
    for f in ("long_policy", "rule_exception", "judge_hard", "agent_action", "routing_hard", "long_input", "multi_label"):
        assert fams[f] > 0, f
    lp_kinds = {r["provenance"]["kind"] for r in sample["long_policy"]}
    assert len(lp_kinds) >= 15  # 17 kinds, 60 draws cycle through them
    doms = {r["provenance"]["domain"] for rows in sample.values() for r in rows}
    assert len(doms) >= 8


def test_english_only(sample):
    for rows in sample.values():
        for r in rows[:40]:
            text = json.dumps(r["state"], ensure_ascii=False) + r["field"]["question"]
            assert not re.search(r"[Ѐ-ӿ؀-ۿ一-鿿぀-ヿ]", text), r["id"]


# ------------------------------------------------------------------------------------------------------------ determinism
def test_deterministic_by_seed_and_index():
    for fam in FAMS:
        a = gp.gen_one(fam, 7, 3)
        b = gp.gen_one(fam, 7, 3)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), fam
        c = gp.gen_one(fam, 8, 3)
        assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True), fam


def test_driver_output_independent_of_workers(tmp_path):
    rows1 = gp.generate({"rule_exception": 40, "long_policy": 10}, seed=5, workers=1, log=lambda *_: None)
    rows2 = gp.generate({"rule_exception": 40, "long_policy": 10}, seed=5, workers=3, log=lambda *_: None)
    assert [json.dumps(r, sort_keys=True) for r in rows1] == [json.dumps(r, sort_keys=True) for r in rows2]


# ------------------------------------------------------------------------------------------------------------ gold
def test_gold_rederived_independently(sample):
    bad = []
    for rows in sample.values():
        for r in rows:
            if not recheck_matches(r):
                bad.append((r["id"], r["provenance"]["kind"], r["gold"]))
    assert not bad, bad[:5]


def test_gold_not_degenerate(sample):
    """Answers are spread: no single option position / boolean dominates a family."""
    for fam, rows in sample.items():
        nouls = [r["gold"] for r in rows if r["field"]["type"] == "noul" and r["gold"] is not None]
        if len(nouls) >= 20:
            share = sum(nouls) / len(nouls)
            assert 0.15 <= share <= 0.85, (fam, share)
        pos = [next(i for i, o in enumerate(r["field"]["options"]) if o["key"] == r["gold"])
               for r in rows if r["field"]["type"] == "choice" and r["gold"] is not None and len(r["field"]["options"]) >= 3]
        if len(pos) >= 30:
            assert max(Counter(pos).values()) / len(pos) < 0.6, (fam, Counter(pos))


# ------------------------------------------------------------------------------------------------------------ unknowns
def test_unknown_variants_are_linked_and_unanswerable(sample):
    for fam, rows in sample.items():
        by_id = {r["id"]: r for r in rows}
        unk = [r for r in rows if r["gold"] is None]
        assert unk, fam
        for u in unk:
            assert u["unknown_reason"] in ("insufficient_evidence", "not_listed", "false_premise", "mismatched_reference")
            assert u["parent_id"] in by_id, u["id"]
            parent = by_id[u["parent_id"]]
            assert parent["gold"] is not None
            assert parent["provenance"]["kind"] == u["provenance"]["kind"]
            assert u["state"] != parent["state"]
            assert module_for(u).recheck(u["provenance"]["spec"]) is None


def _share_unknown(rows):
    return sum(r["gold"] is None for r in rows) / len(rows)


def test_unknown_share_near_15_percent():
    for fam in ("long_policy", "rule_exception", "judge_hard", "agent_action"):
        rows = [r for i in range(300 if fam != "long_policy" else 150) for r in gp.gen_one(fam, 11, i)]
        assert 0.09 <= _share_unknown(rows) <= 0.22, (fam, _share_unknown(rows))


def _shift(iso, k):
    return (dt.date.fromisoformat(iso) + dt.timedelta(days=k)).isoformat()


# the removed fact, set to different plausible values, must give different answers (re-derived with the independent recheck)
ALTS = {
    ("window", "event"): lambda s: [("c", "event", (dt.date.fromisoformat(s["c"]["sent"]) - dt.timedelta(days=k)).isoformat()) for k in (1, 30, 90, 200)],
    ("window", "member_since"): lambda s: [("c", "member_since", (dt.date.fromisoformat(s["c"]["event"]) - dt.timedelta(days=k)).isoformat()) for k in (20, 5000)],
    ("window", "received"): lambda s: [("c", "received", (dt.date.fromisoformat(s["c"]["sent"]) + dt.timedelta(days=k)).isoformat()) for k in (0, 14)],
    ("window", "sent"): lambda s: [("c", "sent", (dt.date.fromisoformat(s["c"]["received"]) - dt.timedelta(days=k)).isoformat()) for k in (0, 14)],
    ("window", "flag"): lambda s: [("c", "flag", True), ("c", "flag", False)],
    ("window", "exc"): lambda s: [("p", "exc", v) for v in (1, 3, 400)],
    ("banded", "tier"): lambda s: [("c", "tier", t) for t in [s["p"]["special_tier"], "__other__"]],
    ("banded", "event"): lambda s: [("c", "event", _shift(s["p"]["amend"]["date"], k)) for k in (-30, 0, 30)],
    ("banded", "base"): lambda s: [("c", "base", "10.00"), ("c", "base", "99999.00")],
    ("scope", "excl"): lambda s: [("r", "excl", True), ("r", "excl", False)],
    ("scope", "date"): lambda s: [("r", "date", "2000-01-01"), ("r", "date", "2100-01-01")],
    ("authority", "category"): lambda s: [("c", "category", s["p"]["cats"][0]), ("c", "category", "__other__")],
    ("authority", "date"): lambda s: [("c", "date", _shift(s["p"]["amend"]["date"], k)) for k in (-20, -1, 0, 20)],
    ("sla", "u0"): lambda s: [("c", "u0", 1), ("c", "u0", 10 ** 6)],
    ("notice", "address"): lambda s: [("c", "address_ok", True), ("c", "address_ok", False)],
    ("notice", "time"): lambda s: [("c", "sent", s["c"]["sent"][:11] + "08:00"), ("c", "sent", s["c"]["sent"][:11] + "23:00")],
}


def test_long_policy_unknowns_depend_on_removed_fact():
    rows = [r for i in range(400) for r in gp.gen_one("long_policy", 21, i)]
    lp = gp.MODULES["long_policy"]
    checked = Counter()
    for u in (r for r in rows if r["gold"] is None):
        spec = u["provenance"]["spec"]
        key = (spec["schema"], spec["missing"])
        if key not in ALTS:
            continue
        answers = set()
        for where, fld, val in ALTS[key](spec):
            s2 = copy.deepcopy(spec)
            s2.pop("missing")
            if where == "r":
                s2["recs"][spec["miss_idx"]][fld] = val
            else:
                s2[where][fld] = val
            answers.add(json.dumps(lp.recheck(s2)))
        assert len(answers) >= 2, (u["id"], key)
        checked[key] += 1
    assert sum(checked.values()) >= 20, checked


def test_unknown_states_show_the_gap():
    rows = [r for i in range(150) for r in gp.gen_one("long_policy", 31, i)]
    markers = ("not recorded", "not reproduced", "not stated", "not attached", "not included", "pending", "not captured", "missing",
               "Order Form", "see order form")
    for u in (r for r in rows if r["gold"] is None):
        assert any(m in u["state"] for m in markers), u["id"]


# ------------------------------------------------------------------------------------------------------------ lengths
def test_long_policy_documents_within_word_range(sample):
    for r in sample["long_policy"]:
        n = words(r["state"])
        assert 1500 <= n <= 2800, (r["id"], n)


def test_long_input_states_within_token_range(sample):
    for r in sample["long_input"]:
        text = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])
        assert 4000 <= approx_tokens(text) <= 16000, (r["id"], approx_tokens(text))


# ------------------------------------------------------------------------------------------------------------ multi-label
def test_multi_label_groups(sample):
    groups = defaultdict(list)
    for r in sample["multi_label"]:
        assert r["field"]["type"] == "noul"
        gid = r["provenance"].get("group_id")
        if r["parent_id"] is not None:  # unknown children link to their label item by parent_id
            continue
        assert gid, r["id"]
        groups[gid].append(r)
    parents = [g for g in groups.values() if any(r["parent_id"] is None for r in g)]
    assert parents
    for g in parents:
        n = sum(r["parent_id"] is None for r in g)
        assert 4 <= n <= 10, n


# ------------------------------------------------------------------------------------------------------------ traps
@pytest.fixture(scope="module")
def trap_parents(sample):
    return [r for rows in sample.values() for r in rows]


def _gold_text(row):
    return next(o["text"] for o in row["field"]["options"] if o["key"] == row["gold"])


def test_traps_valid_and_gold_exact(trap_parents):
    by_id = {r["id"]: r for r in trap_parents}
    made = Counter()
    for p in trap_parents:
        for kind in gt.KINDS:
            t = gt.make_trap(p, kind, seed=3)
            if t is None:
                continue
            made[kind] += 1
            assert validate(t) == [] and t["source"] == "D" and t["parent_id"] == p["id"]
            assert t["provenance"]["trap"] == kind
            if kind in ("near_miss", "qualifier"):
                assert t["gold"] == p["gold"]
                added = next(o for o in t["field"]["options"] if o["key"] == t["provenance"]["trap_info"]["added_option"])
                assert len(t["field"]["options"]) == len(p["field"]["options"]) + 1
                if p["gold"] is not None:
                    assert added["text"].strip().lower() != _gold_text(p).strip().lower()
                    spec = p["provenance"]["spec"]
                    if spec.get("opt_values") and p["provenance"]["trap_hints"].get("near_miss") and t["provenance"]["trap_info"].get("source") == "hint":
                        assert added["text"] in p["provenance"]["trap_hints"]["near_miss"]
            elif kind == "bury":
                assert t["gold"] == p["gold"] and t["field"] == p["field"]
                if isinstance(p["state"], str):
                    # the original lines survive, in order; only notice lines were added
                    orig = [ln for ln in p["state"].split("\n") if ln.strip()]
                    new = [ln for ln in t["state"].split("\n") if ln.strip()]
                    it = iter(new)
                    assert all(any(ln == x for x in it) for ln in orig)
                    added = [ln for ln in new if ln not in set(orig)]
                    assert added and all(ln.startswith("- ") or ln.endswith(":") for ln in added)
                else:
                    extra = set(t["state"]) - set(p["state"])
                    assert len(extra) == 1 and all(t["state"][k] == v for k, v in p["state"].items())
            elif kind == "negation":
                pf, tf = p["field"], t["field"]
                if p["gold"] is None:
                    # a negated noul question of an unknown item stays unknown; score / choice negations of an unknown item may
                    # become decidable (gold re-derived by enumeration: tests/test_p3_trap_fix.py)
                    if pf["type"] == "noul":
                        assert t["gold"] is None
                elif pf["type"] == "noul":
                    assert t["gold"] is (not p["gold"])
                elif pf["type"] == "choice":
                    gtext = _gold_text(p)
                    chosen = next(o["text"] for o in tf["options"] if o["key"] == t["gold"])
                    assert chosen.startswith("Answer ") and not chosen.endswith(gtext)
                    other = [o["text"] for o in tf["options"] if o["text"].startswith("Answer ") and o["key"] != t["gold"]]
                    assert other and other[0].endswith(gtext)
                else:
                    L = t["provenance"]["trap_info"]["level"]
                    assert t["gold"] is (L != p["gold"])
    for k in gt.KINDS:
        assert made[k] > 5, made


def test_trap_hint_negations_match_recheck(sample):
    """A generator's own negated question must have the opposite gold of the parent (recomputed by recheck)."""
    n = 0
    for rows in sample.values():
        for r in rows:
            neg = r["provenance"].get("trap_hints", {}).get("neg")
            if neg and r["gold"] is not None and r["field"]["type"] == "noul":
                assert neg["gold"] is (not module_for(r).recheck(r["provenance"]["spec"]))
                n += 1
    assert n > 30


def test_trap_build_mix_and_unknown_share(trap_parents):
    traps = gt.build(trap_parents, 300, seed=1)
    assert 250 <= len(traps) <= 300
    kinds = Counter(t["provenance"]["trap"] for t in traps)
    assert all(kinds[k] > 0 for k in gt.KINDS), kinds
    unk = sum(t["gold"] is None for t in traps) / len(traps)
    assert 0.08 <= unk <= 0.2, unk
    assert len({t["id"] for t in traps}) == len(traps)


def test_traps_work_on_foreign_candidate_rows():
    row = {"id": "p3-reason-000001", "source": "A", "dataset": "gen_reasoning", "family": "logic_chain", "difficulty": 4,
           "state": "Rule 1: all blips are blops.\n\nRule 2: no blop is a blup.\n\nFact: X is a blip.", "images": [],
           "field": {"type": "choice", "question": "How many business days does the review take?",
                     "options": [{"key": "a", "text": "10 business days"}, {"key": "b", "text": "12 business days"}, {"key": "c", "text": "15 days"}]},
           "gold": "a", "unknown_reason": None, "gold_kind": "constructed", "parent_id": None, "provenance": {"licence": "generated"}}
    for kind in gt.KINDS:
        t = gt.make_trap(row, kind, seed=0)
        assert t is not None, kind
        assert validate(t) == []
    q = gt.make_trap(row, "qualifier", 0)
    added = next(o for o in q["field"]["options"] if o["key"] == q["provenance"]["trap_info"]["added_option"])
    assert added["text"] in ("10 calendar days",) and q["gold"] == "a"
    nm = gt.make_trap(row, "near_miss", 0)
    added = next(o for o in nm["field"]["options"] if o["key"] == nm["provenance"]["trap_info"]["added_option"])
    assert added["text"] != "10 business days" and "business days" in added["text"]


def test_perturb_number_leaves_dates_and_ids_alone():
    import random
    r = random.Random(0)
    assert gt.perturb_number("12 March 2026", r) is None
    assert gt.perturb_number("2026-03-12", r) is None
    assert gt.perturb_number("INC-676528", r) is None
    out = gt.perturb_number("$1,098.40", r)
    assert out and out != "$1,098.40" and out.startswith("$")


# ------------------------------------------------------------------------------------------------------------ variants
def test_variant_of_keeps_family_kind_difficulty(tmp_path, sample):
    parents = [r for fam in ("long_policy", "rule_exception", "judge_hard") for r in sample[fam][:12]]
    src = tmp_path / "parents.jsonl"
    write(src, parents)
    var = gp.variants_of(src, per=2)
    assert len(var) >= int(len(parents) * 1.8)
    by_id = {r["id"]: r for r in parents}
    for v in var:
        assert validate(v) == []
        p = by_id[v["parent_id"]]
        assert v["family"] == p["family"] and v["difficulty"] == p["difficulty"]
        assert v["provenance"]["kind"] == p["provenance"]["kind"]
        assert (v["gold"] is None) == (p["gold"] is None)
        assert v["state"] != p["state"]
        assert recheck_matches(v)


def test_multi_label_questions_use_serving_wording(sample):
    sys.path.insert(0, str(ROOT / "src"))
    from vision_decision.jev_api import multi_label_question
    groups = defaultdict(set)
    for r in sample["multi_label"]:
        q = r["field"]["question"]
        instr, label_line, tail = q.rsplit("\n", 2)
        assert tail == "Does this label apply?" and label_line.startswith("Label: ")
        m = re.match(r"Label: (.+?) \((.+)\)$", label_line)
        assert m, label_line
        assert q == multi_label_question(instr, m.group(1), m.group(2))
        if r["parent_id"] is None:
            groups[r["provenance"]["group_id"]].add(instr)
        # the definitions live in the question, as at serving time, not in the state
        assert "Does this label apply?" not in str(r["state"])
    assert all(len(v) == 1 for v in groups.values())  # one shared instruction per multi-label item
