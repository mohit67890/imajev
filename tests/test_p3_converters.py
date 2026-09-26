"""Phase-3 public-dataset converters (scripts/p3/convert_*.py): small in-memory fixtures, no network, no raw data needed."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/p3"))
from candidate import validate  # noqa: E402
import convert_common as cc  # noqa: E402
import convert_finqa, convert_tatqa, convert_musique, convert_strategyqa, convert_plumb, convert_alfworld  # noqa: E402,E401


# ------------------------------------------------------------------------------------------------------------ common
def test_option_key_is_snake_unique_and_never_unknown():
    taken = set()
    assert cc.option_key("Unknown", taken) != "unknown"
    a, b = cc.option_key("9.93%", taken), cc.option_key("9.93%", taken)
    assert a != b and a[0].isalpha()
    assert cc.option_key(cc.numeric_key_text("-9.93%")) == "neg_9_93_pct"
    assert cc.option_key(cc.numeric_key_text("9.93%")) == "v_9_93_pct"


def test_expression_engine_parses_tatqa_style_and_perturbs():
    t = cc.parse_expr("[(4,573 - 4,312)] / 4,312 * 100")
    assert cc.evaluate(t) == pytest.approx(6.0528, abs=1e-3)
    assert cc.count_ops(t) == 3
    kinds = {k for k, _ in cc.perturbations(t, lambda x: [4000.0] if x == 4312 else [])}
    assert {"stopped_early", "wrong_operation", "swapped_operands", "wrong_cell"} <= kinds
    with pytest.raises(cc.ExprError):
        cc.parse_expr("__import__('os')")
    with pytest.raises(cc.ExprError):
        cc.parse_expr("revenue - cost")


def test_numeric_options_are_distinct_after_rounding_and_never_equal_gold():
    fmt = cc.NumFormat(decimals=1, suffix="%")
    cands = [("wrong_cell", 9.93001), ("wrong_cell", 9.96), ("wrong_operation", 20.0), ("swapped_operands", -9.9),
             ("stopped_early", 43.0), ("wrong_cell", 5.0), ("wrong_operation", 900000.0)]
    texts, gold, kinds = cc.numeric_options(9.93, cands, fmt, seed="x")
    assert gold == "9.9%" and texts[0] == gold and 4 <= len(texts) <= 6
    assert len(set(texts)) == len(texts)
    vals = [float(t.rstrip("%")) for t in texts[1:]]
    assert all(abs(v - 9.93) >= 0.2 for v in vals)          # 9.93001 and 9.96 would round to / sit on the gold
    assert "900,000%" not in texts                           # implausible magnitude dropped


def test_numeric_options_fall_back_to_slips_when_few_structural_distractors():
    texts, gold, kinds = cc.numeric_options(12.5, [], cc.NumFormat(decimals=2), seed="y")
    assert len(texts) >= 4 and set(kinds.values()) - {"gold"} <= {"sign_slip", "scale_slip", "near_miss"}


# ------------------------------------------------------------------------------------------------------------ FinQA
def finqa_item(program, exe, answer, gold_inds=None, qid="ABC/2015/page_10.pdf-2"):
    return {"id": qid, "pre_text": ["net cash provided by operating activities was $ 476 million in 2015 ."],
            "post_text": ["."], "table": [["", "2015", "2014", "2013"], ["operating cash", "$ 476", "$ 433", "$ 400"],
                                          ["capex", "$ 120", "$ 110", "$ 90"]],
            "qa": {"question": "what is the percent increase in operating cash from 2014 to 2015?", "program": program,
                   "exe_ans": exe, "answer": answer,
                   "gold_inds": gold_inds or {"table_1": "operating cash of 2015 is $ 476 ; 2014 is $ 433"}}}


def test_finqa_hard_item_converts_with_program_distractors():
    row, why = convert_finqa.convert_item(finqa_item("subtract(476, 433), divide(#0, 433)", 0.09931, "9.9%"), 0)
    assert why == "ok" and validate(row) == []
    opts = {o["key"]: o["text"] for o in row["field"]["options"]}
    assert opts[row["gold"]] == "9.9%"
    assert row["provenance"]["upstream_id"] == "ABC/2015/page_10.pdf-2" and row["provenance"]["upstream_split"] == "train"
    kinds = set(row["provenance"]["option_kinds"].values())
    assert "gold" in kinds and kinds & {"wrong_cell", "stopped_early", "swapped_operands", "wrong_operation"}
    assert row["difficulty"] >= 3 and "unknown" not in opts


def test_finqa_drops_easy_and_noisy_items():
    assert convert_finqa.convert_item(finqa_item("divide(476, 433)", 1.09931, "109.9%"), 0)[1] == "easy"
    # the dataset's answer string disagrees with its own program
    assert convert_finqa.convert_item(finqa_item("subtract(476, 433), divide(#0, 433)", 0.09931, "-9.9%"), 0)[1] == \
        "answer_mismatch"


def test_finqa_table_ops_and_yes_no():
    it = finqa_item("table_average(capex, none), greater(#0, 100)", "yes", "yes")
    row, why = convert_finqa.convert_item(it, 0)
    assert why == "ok" and row["field"]["type"] == "noul" and row["gold"] is True and validate(row) == []


# ------------------------------------------------------------------------------------------------------------ TAT-QA
def tatqa_ctx(derivation, answer, scale="percent", answer_from="table"):
    return {"table": {"uid": "ctx-1", "table": [["", "2019", "2018"], ["Revenue", "4,573", "4,312"], ["Cost", "(1,200)", "1,100"]]},
            "paragraphs": [{"uid": "p1", "order": 1, "text": "Revenue grew to 4,573 in 2019 from 4,312."}],
            "questions": [{"uid": "q-1", "question": "What is the percentage change in revenue?", "answer": answer,
                           "derivation": derivation, "answer_type": "arithmetic", "answer_from": answer_from, "scale": scale}]}


def test_tatqa_percentage_change_converts():
    ctx = tatqa_ctx("(4,573-4,312)/4,312", 6.05)
    row, why = convert_tatqa.convert_question(ctx, ctx["questions"][0], convert_tatqa.render_state(ctx), 0)
    assert why == "ok" and validate(row) == []
    opts = {o["key"]: o["text"] for o in row["field"]["options"]}
    assert opts[row["gold"]] == "6.05%" and len(opts) >= 4
    assert row["provenance"]["upstream_id"] == "q-1" and row["provenance"]["context_uid"] == "ctx-1"


def test_tatqa_filters():
    ctx = tatqa_ctx("4,573-4,312", 261, scale="")
    assert convert_tatqa.convert_question(ctx, ctx["questions"][0], "", 0)[1] == "easy"
    ctx = tatqa_ctx("4,573-4,312", 261, scale="", answer_from="table-text")
    assert convert_tatqa.convert_question(ctx, ctx["questions"][0], "s", 0)[1] == "ok"
    ctx = tatqa_ctx("(4,573-4,312)/4,312", 9.99)
    assert convert_tatqa.convert_question(ctx, ctx["questions"][0], "", 0)[1] == "answer_mismatch"


# ------------------------------------------------------------------------------------------------------------ MuSiQue
def musique_pair(qid, answer, hop_answers, aliases=()):
    paras = [{"idx": i, "title": f"Title {i}", "paragraph_text": f"Paragraph {i} mentions Paris and Lyon and 1901.",
              "is_supporting": i < 2} for i in range(6)]
    dec = [{"id": i, "question": f"step {i}", "answer": a, "paragraph_support_idx": i} for i, a in enumerate(hop_answers)]
    dec.append({"id": 9, "question": "#2 >> capital", "answer": answer, "paragraph_support_idx": 1})
    base = {"id": qid, "question": f"What is the capital of the thing in {qid}?", "question_decomposition": dec,
            "answer": answer, "answer_aliases": list(aliases)}
    return [{**base, "paragraphs": paras, "answerable": True}, {**base, "paragraphs": paras[1:], "answerable": False}]


def test_musique_pairs_become_answerable_and_unknown_twins(tmp_path):
    rows = []
    rows += musique_pair("3hop1__1_2_3", "Paris", ["France", "Europe"], aliases=["City of Paris"])
    for n, cap in enumerate(["Lyon", "Berlin", "Madrid", "Rome", "Vienna"]):
        rows += musique_pair(f"2hop__{n}_9", cap, ["Country"])
    raw = tmp_path / "m.jsonl"
    raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "o.jsonl"
    stats = convert_musique.convert(raw, out, two_hop_pairs=0)
    got = [json.loads(line) for line in open(out)]
    assert stats["written"] == 2 and all(validate(r) == [] for r in got)
    ans, unans = got
    assert ans["gold"] is not None and unans["gold"] is None and unans["unknown_reason"] == "insufficient_evidence"
    assert unans["parent_id"] == ans["id"] and ans["field"]["options"] == unans["field"]["options"]
    assert ans["provenance"]["upstream_id"] == unans["provenance"]["upstream_id"] == "3hop1__1_2_3"
    texts = [o["text"] for o in ans["field"]["options"]]
    assert "Paris" in texts and "France" in texts and "City of Paris" not in texts   # hop trap in, alias out
    assert ans["difficulty"] == 3 and unans["difficulty"] == 4


# ------------------------------------------------------------------------------------------------------------ StrategyQA
def test_strategyqa_hides_facts_and_mixes_distractors(tmp_path):
    paras = {f"P{i}-1": {"title": f"P{i}", "content": f"Shrimp scampi recipe note {i} with garlic butter."} for i in range(8)}
    items = [{"qid": "abc123", "term": "Shrimp scampi", "question": "Is shrimp scampi free of plastic?", "answer": False,
              "facts": ["SECRET FACT about microplastics."], "decomposition": ["What is in it?", "Does #1 contain plastic?"],
              "evidence": [[[["P0-1", "P1-1"]], ["no_evidence"]]]},
             {"qid": "one", "term": "x", "question": "Is it?", "answer": True, "facts": [], "decomposition": ["Is it?"],
              "evidence": []}]
    (tmp_path / "strategyqa_train.json").write_text(json.dumps(items))
    (tmp_path / "strategyqa_train_paragraphs.json").write_text(json.dumps(paras))
    stats = convert_strategyqa.convert(tmp_path, tmp_path / "o.jsonl")
    rows = [json.loads(line) for line in open(tmp_path / "o.jsonl")]
    assert stats["written"] == 1 and validate(rows[0]) == []
    r = rows[0]
    assert "SECRET FACT" not in r["state"] and "What is in it" not in r["state"]
    assert r["provenance"]["n_evidence"] == 2 and r["provenance"]["n_distractor_paragraphs"] >= 1
    assert r["field"]["type"] == "noul" and r["gold"] is False and r["provenance"]["upstream_id"] == "abc123"


# ------------------------------------------------------------------------------------------------------------ plumb
def test_plumb_rows_keep_criteria_semantics():
    noul = {"id": "train-1-000001-q0", "family": "judge", "domain": "d", "length": "very long (1,500-2,800 words)",
            "state": "doc", "question": {"type": "noul", "instructions": "Determine if SwiftHaul is liable.",
                                         "criteria": {"true": "SwiftHaul is not liable.", "false": "SwiftHaul is liable."}},
            "expected": "true", "explanation": "e", "teacher_probs": {"true": 1.0, "false": 0.0}}
    row, why = convert_plumb.convert_row(noul, 0)
    assert why == "ok" and validate(row) == [] and row["gold"] is True
    assert "Answer true if: SwiftHaul is not liable." in row["field"]["question"]
    assert row["family"] == "judge_hard" and row["difficulty"] == 5 and row["provenance"]["group_id"] == "train-1-000001"
    score = {**noul, "question": {"type": "score", "instructions": "Rate the risk of the claim.", "criteria": ["low", "mid", "high"]},
             "expected": "2", "family": "rubric", "length": "medium (250-450 words)"}
    row, _ = convert_plumb.convert_row(score, 1)
    assert row["gold"] == 2 and [lv["value"] for lv in row["field"]["levels"]] == [0, 1, 2] and validate(row) == []
    choice = {**noul, "question": {"type": "choice", "instructions": "Which total is correct?",
                                   "criteria": {"total_1": "a", "unknown": "b"}}, "expected": "total_1"}
    row, _ = convert_plumb.convert_row(choice, 2)
    assert validate(row) == [] and all(o["key"] != "unknown" for o in row["field"]["options"])


def test_plumb_refuses_test_split(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text("")
    with pytest.raises(AssertionError):
        convert_plumb.convert(p, tmp_path / "o.jsonl")


# ------------------------------------------------------------------------------------------------------------ ALFWorld
def alf_record():
    adm0 = ["go to shelf 1", "go to shelf 2", "go to cart 1", "look", "inventory"]
    return {"game": "train/pick_and_place_simple-Candle-None-Cart-1/trial_T1", "task_type": "pick_and_place_simple",
            "won": True, "intro": "-= Welcome =-\n\nYou are in the middle of a room. Looking quickly around you, you see a cart 1, "
                                  "a shelf 1, and a shelf 2.\n\nYour task is to: put a candle in cart.",
            "steps": [
                {"action": "go to shelf 1", "admissible": adm0,
                 "obs_after": "You arrive at shelf 1. On the shelf 1, you see a candle 1, a candle 2, and a pen 1."},
                {"action": "take candle 1 from shelf 1",
                 "admissible": adm0 + ["take candle 1 from shelf 1", "take candle 2 from shelf 1", "take pen 1 from shelf 1"],
                 "obs_after": "You pick up the candle 1 from the shelf 1."},
                {"action": "go to cart 1", "admissible": adm0 + ["move candle 1 to shelf 1"],
                 "obs_after": "You arrive at cart 1. On the cart 1, you see nothing."},
                {"action": "move candle 1 to cart 1", "admissible": adm0 + ["move candle 1 to cart 1", "examine cart 1"],
                 "obs_after": "You move the candle 1 to the cart 1."}]}


def test_alfworld_keeps_only_decisive_steps_and_removes_equivalent_options():
    rec = alf_record()
    got = {i: (g, o) for i, g, o in convert_alfworld.decisive_steps(rec)}
    assert 0 not in got                                  # exploration: any shelf would do
    g, opts = got[1]
    assert g == "take candle 1 from shelf 1" and "take candle 2 from shelf 1" not in opts and "take pen 1 from shelf 1" in opts
    assert got[2][0] == "go to cart 1" and got[3][0] == "move candle 1 to cart 1"
    state = convert_alfworld.render_state(rec, 2)
    assert "Task: put a candle in cart." in state and "take candle 1 from shelf 1" in state and "go to cart 1" not in state


def test_alfworld_convert_writes_valid_rows(tmp_path):
    dump = tmp_path / "d.jsonl"
    dump.write_text(json.dumps(alf_record()) + "\n")
    stats = convert_alfworld.convert(dump, tmp_path / "o.jsonl")
    rows = [json.loads(line) for line in open(tmp_path / "o.jsonl")]
    assert stats["written"] == 3 and all(validate(r) == [] for r in rows)
    assert {r["provenance"]["upstream_id"] for r in rows} == {"train/pick_and_place_simple-Candle-None-Cart-1/trial_T1"}
    assert all(r["provenance"]["upstream_split"] == "train" for r in rows)
