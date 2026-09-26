"""Audit constructed-gold rows for the generator defects found in the Kimi review of the recovered rows (2026-09-26).

Report: reports/phase3/recovered-adjudication.md. Read-only on every data file; writes only --flags / --drop-out when given.

Each rule returns a defect code when a row's constructed gold is wrong or not defensible from what the model is shown.
Rules marked DROP go to the drop list; rules marked INFO are counted only.

GUI (gen_gui.py; the screen is re-built from its seed + screen_id, which reproduces provenance.screen_meta exactly):
  GUI_SEL_DELETE    DROP  toolbar_available "Can you permanently remove the selected rows ... right now?", gold true, but
                          the screen says "No rows selected" (the status line is random and independent of the goal)
                          -> gold should be false.
  GUI_SEL_ROW       DROP  same for "Can you <goal> ... right now?" where the goal acts on one row/item/file ("this item",
                          "this task", "the file") and the screen says "No rows selected" -> not settled by the screen.
  GUI_SEL_CLICK     INFO  the same screens asked as "Is the control to <goal> available to click right now?": the enabled
                          button answers that literally, gold true is defensible.
  GUI_DARK          DROP  settings_state / settings_which about "Dark theme": the page theme is drawn independently of the
                          switch, so the screenshot itself shows whether the app is dark. Unknown rows (setting not on
                          screen) are decidable from the colours; answerable rows are wrong when switch != page darkness.
                          (settings_which is not affected; only settings_state is flagged.)
  GUI_ALT           DROP  a toolbar control other than the generator's "correct" one also achieves the goal
                          ("Copy link" lets a colleague open the report; "Share" gives a web address to the file):
                          toolbar_goal unknowns (not_listed) with the alternative on screen, toolbar_goal answerable with both
                          on screen, toolbar_available false (correct control disabled) with the alternative enabled.
  GUI_ALT_WEAK      INFO  toolbar_available false for "copy a web address" (Copy link disabled) with Share enabled.
  GUI_PRINT_PDF     INFO  goal "save the report as a PDF file" with an enabled Print control (print-to-PDF reading).
Image joint (gen_image_joint.py):
  IJ_WEIGHT         DROP  unknown whose hidden attribute is "weighs more than 2 kg": the photo settles it for almost every
                          ABO product type (sofa, bed, rug vs phone case, ring, sandal).
  IJ_WEIGHCHECK     DROP  def_two_units unknown whose hidden attribute is "was weighed under the minimum" while the record
                          gives each unit's weight_check pass/fail: a passed weight check answers it.
  IJ_HALT_NODEFECT  DROP  def_severity_points unknown built by withholding points_so_far, photo has no defect (adds 0):
                          a unit that adds 0 cannot bring a running line to a halt; the "unknown" rests on the impossible
                          alternative total 999.
gen_policy (source A) and gen_traps (source D) rows, rules in d_rules() (codes D_*; all DROP):
  D_BURY_IN_REPLY, D_NEG_PAIR_UNK, D_ML_VELOCITY, D_TOOL_LOST, D_NM_NUM_UNGROUNDED, D_NEG_ZERO_WORLD, D_LP_BAND,
  D_RX_EXCL_SCOPE (each explained at its test).

    .venv/bin/python scripts/p3/audit_recovered.py [--flags OUT.jsonl] [--drop-out data/p3/fixes/drop-ids-recovered.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "p3"))

TRAIN_FILES = [
    "data/p3/fixes/direct-recovered-rejected.jsonl",
    "data/p3/candidates-clean/I-gui.jsonl",
    "data/p3/candidates-clean/I-joint.jsonl",
    "data/p3/candidates-clean/D-traps.jsonl",
    "data/p3/candidates-clean/A-policy.jsonl",
    "data/p3/labeltrain/direct-constructed-leftovers.jsonl",
    "data/p3/labeltrain/direct-constructed-leftovers-late.jsonl",
    "data/p3/variants/constructed.jsonl",
]
EVAL_FILES = ["data/p3/pool/heldout-fresh-gui.jsonl", "data/p3/pool/heldout-fresh.jsonl"]
DROP_RULES = {"GUI_SEL_DELETE", "GUI_SEL_ROW", "GUI_DARK", "GUI_ALT", "IJ_WEIGHT", "IJ_WEIGHCHECK", "IJ_HALT_NODEFECT",
              "D_BURY_IN_REPLY", "D_NEG_PAIR_UNK", "D_ML_VELOCITY", "D_TOOL_LOST", "D_NM_NUM_UNGROUNDED", "D_NEG_ZERO_WORLD",
              "D_LP_BAND", "D_RX_EXCL_SCOPE"}

# ------------------------------------------------------------------------------------------------ GUI
_GG = None
_SCREEN_CACHE: dict = {}
DARK_BG = {"#121417", "#0f1a1f"}
ROW_GOALS = {"hide this item from the list without deleting it", "give this task to a teammate",
             "leave a note for your team on this task", "mark the task as finished", "change the file's name",
             "put the file in another folder", "copy a web address that points to this file",
             "save the original file to your computer"}
ALT = {"let a colleague open this report": "Copy link", "copy a web address that points to this file": "Share"}


def _gg():
    global _GG
    if _GG is None:
        import gen_gui
        _GG = gen_gui
    return _GG


def screen_html(pv: dict) -> str | None:
    """Re-build the screen from its seed; returns the HTML, or None when the rebuilt meta differs from the stored one."""
    gg = _gg()
    from gen_image_joint import rng_for
    sid = pv["screen_id"]
    m = re.fullmatch(r"(?:(\w+)-)?s(\d{5})", sid)
    if not m:
        return None
    tag, i = m.group(1), int(m.group(2))
    seed = gg.SEED if tag is None else f"p3-gui-heldout-{tag}"
    key = (seed, i, pv["screen_kind"])
    if key not in _SCREEN_CACHE:
        s = gg.KINDS[pv["screen_kind"]][1](rng_for(seed, "screen", i), f"s{i:05d}")
        _SCREEN_CACHE[key] = s["html"] if s["meta"] == pv["screen_meta"] else None
    return _SCREEN_CACHE[key]


def _enabled(html: str, label: str) -> bool:
    gg = _gg()
    from convert_common import option_key
    m = re.search(r'data-el="b_%s"( disabled)?' % re.escape(option_key(label)), html)
    return bool(m) and not m.group(1)


def gui_rules(r: dict) -> list[str]:
    pv = r.get("provenance") or {}
    task, meta = pv.get("task"), pv.get("screen_meta") or {}
    out = []
    if task in ("toolbar_available", "toolbar_goal", "settings_state"):
        html = screen_html(pv)
        if html is None:
            return ["GUI_REBUILD_MISMATCH"]
    if task == "toolbar_available" and r["gold"] is True and "No rows selected" in html \
            and (meta["goal"] == "permanently remove the selected rows" or meta["goal"] in ROW_GOALS):
        if r["field"]["question"].startswith("Is the control"):
            out.append("GUI_SEL_CLICK")      # "is the control available to click": the enabled button answers it
        elif meta["goal"] == "permanently remove the selected rows":
            out.append("GUI_SEL_DELETE")
        else:
            out.append("GUI_SEL_ROW")
    if task == "settings_state" and meta.get("target") == "Dark theme":
        bg = re.search(r"body\{margin:0;background:(#[0-9a-f]{6})", html).group(1)
        dark = bg in DARK_BG
        if r["gold"] is None or r["gold"] != dark:
            out.append("GUI_DARK")
    goal = meta.get("goal")
    if task in ("toolbar_goal", "toolbar_available") and goal in ALT and ALT[goal] in meta["labels"]:
        alt = ALT[goal]
        strong = goal == "let a colleague open this report"
        if task == "toolbar_goal" and (strong or r["gold"] is None):
            out.append("GUI_ALT")        # unknown: the alternative achieves the goal; answerable: two defensible answers
        elif task == "toolbar_available" and r["gold"] is False and _enabled(html, alt):
            out.append("GUI_ALT" if strong else "GUI_ALT_WEAK")
    if task == "toolbar_available" and goal == "save the report as a PDF file" and r["gold"] is False \
            and "Print" in meta["labels"] and _enabled(html, "Print"):
        out.append("GUI_PRINT_PDF")
    return out


# ------------------------------------------------------------------------------------------------ image joint
def ij_rules(r: dict) -> list[str]:
    if r["gold"] is not None:
        return []
    pv = r.get("provenance") or {}
    p = pv.get("params") or {}
    hidden = [x["hidden"] for x in p.get("preds", []) if x.get("hidden")]
    out = []
    if "weighs more than 2 kg" in hidden:
        out.append("IJ_WEIGHT")
    if pv.get("template") == "def_two_units" and "was weighed under the minimum" in hidden:
        out.append("IJ_WEIGHCHECK")
    if pv.get("template") == "def_severity_points" and p.get("_withheld") == ["total"] \
            and not pv["photo_facts"][0]["defective"]:
        out.append("IJ_HALT_NODEFECT")
    return out


# ------------------------------------------------------------------------------------------------ gen_policy / D traps
NOTICE_HEADS = ("Office notices (for information):", "Unrelated operational notes:", "General notices:", "Noticeboard:",
                "Housekeeping announcements:", "Other updates this week:")
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PARENTS: dict | None = None
PARENT_FILE = "data/p3/candidates/A-policy.jsonl"


def _parents() -> dict:
    """gen_policy parents of agent_action / routing_hard traps (for the negation pair keys)."""
    global _PARENTS
    if _PARENTS is None:
        _PARENTS = {}
        for line in open(ROOT / PARENT_FILE):
            if '"agent_action"' in line or '"routing_hard"' in line:
                r = json.loads(line)
                _PARENTS[r["id"]] = r
    return _PARENTS


def _dashed_blocks(state: str) -> list[list[str]]:
    out, cur = [], None
    for ln in state.split("\n"):
        if re.fullmatch(r"-{6,}", ln.strip()):
            if cur is None:
                cur = []
            else:
                out.append(cur)
                cur = None
        elif cur is not None:
            cur.append(ln)
    return out


def _craft_ok(spec, inv):
    recipes, goal, tools = spec["recipes"], spec["goal"], set(spec["tools"])
    outs = {r["out"] for r in recipes}
    need, fr = set(), [goal]
    while fr:
        nx = []
        for x in fr:
            if x in need or inv.get(x, 0) >= 1:
                continue
            need.add(x)
            nx += [i for r in recipes if r["out"] == x for i, _ in r["in"] if i in outs]
        fr = nx
    return [r["id"] for r in recipes if r["out"] in need and (not r["tool"] or r["tool"] in tools)
            and min([inv.get(i, 0) - q for i, q in r["in"]]) >= 0]


def _zero_world(spec) -> bool:
    """Some completion of the withheld fact leaves no recipe / tool at all (the solver's '__ambiguous__')."""
    from gen_policy_families import agent_action as AA
    k, unk = spec.get("k"), spec.get("unk")
    if k == "craft" and unk:
        return any(len(_craft_ok(spec, dict(spec["inv"], **{unk["raw"]: v}))) == 0 for v in unk["vals"])
    if k == "toolsel" and unk:
        vals = {"age": [1, 45, 95, 200, 400], "amount": [5, 50, 250, 1000, 3000, 6000], "region": AA.REGIONS,
                "verified": [True, False], "channel": ["own website", "marketplace", "wholesale account", "in store"]}[unk]
        for v in vals:
            req, n = dict(spec["req"], **{unk: v}), 0
            for tl in spec["tools"]:
                if tl["cap"] != spec["cap"]:
                    continue
                good = True
                for d, vv in tl["c"]:
                    x = req[d]
                    if isinstance(vv, bool):
                        good &= (x or not vv)
                    elif isinstance(vv, list) and vv and isinstance(vv[0], int) and d in ("age", "amount"):
                        good &= vv[0] <= x <= vv[1]
                    else:
                        good &= x in vv
                n += good
            if n == 0:
                return True
    return False


def d_rules(r: dict) -> list[str]:
    """Rules over gen_policy rows (source A) and their gen_traps children (source D); both carry provenance.spec."""
    pv = r.get("provenance") or {}
    sp = pv.get("spec") or {}
    fam, gold, trap, ti = r.get("family"), r.get("gold"), pv.get("trap"), pv.get("trap_info") or {}
    st = r["state"] if isinstance(r.get("state"), str) else json.dumps(r.get("state"), ensure_ascii=False)
    out = []
    # D_BURY_IN_REPLY: the bury notice block landed between a candidate reply's dashed delimiters, so reply word counts
    # (the rubric's length criterion) are stale; 15/20 recovered judge_hard rows flip the length verdict.
    if trap == "bury" and any(any(h in b for h in NOTICE_HEADS) for b in _dashed_blocks(st)):
        out.append("D_BURY_IN_REPLY")
    # D_NEG_PAIR_UNK: choice negation of an unknown parent kept as unknown; "which proposed answer is NOT supported?"
    # is fairly answered "Neither" when the material settles neither (superset of the excluded pair_overlap=partial rows).
    if trap == "negation" and ti.get("source") == "pair" and gold is None:
        out.append("D_NEG_PAIR_UNK")
    # D_ML_VELOCITY: multi_label risk/velocity rendered as raw JSON keys; whether txns_24h includes declines decides it.
    if fam == "multi_label" and sp.get("k") == "risk" and sp.get("ask") == "velocity" and "Record: {" in st:
        p, rec = sp["params"]["velocity"], sp["rec"]
        if p.get("approved_only") and rec["declined_24h"] > 0 and not sp.get("missing") \
                and rec["txns_24h"] - rec["declined_24h"] < p["v"] <= rec["txns_24h"]:
            out.append("D_ML_VELOCITY")
    # D_TOOL_LOST: tool_next unknown built from a "no response recorded" call; the task says a call counts only when it
    # returned ok and the success-branch option carries ids/amounts that appear nowhere -> the retry is the answer.
    if sp.get("k") == "tool" and sp.get("unk") and gold is None:
        out.append("D_TOOL_LOST")
    # D_NM_NUM_UNGROUNDED: near-miss changed a number in the gold tool call that the state never states -> the gold and
    # the added option cannot be told apart.
    if trap == "near_miss" and ti.get("source") == "number" and gold is not None and pv.get("kind") == "tool_next":
        opts = {o["key"]: o["text"] for o in r["field"]["options"]}
        diff = set(NUM.findall(opts[gold])) - set(NUM.findall(opts.get(ti.get("added_option"), "")))
        if any(x not in st for x in diff):
            out.append("D_NM_NUM_UNGROUNDED")
    # D_NEG_ZERO_WORLD: craft/toolsel choice negation left unknown because one completion has no valid recipe/tool,
    # although neither proposed answer is right in any completion -> gold should be "Neither".
    if trap == "negation" and r["field"]["type"] == "choice" and gold is None and sp.get("k") in ("craft", "toolsel") \
            and _zero_world(sp):
        from gen_policy_families import agent_action as AA
        import gen_traps as GT
        par = _parents().get(r.get("parent_id"))
        real = {w for w in AA.worlds(sp) if not str(w).startswith("__")}
        if par is not None and not (real & set(GT._pair_keys(par, r))):
            out.append("D_NEG_ZERO_WORLD")
    # D_LP_BAND: long_policy banded schema in public_sector names the applicant tier "Band 1-4", colliding with the fee
    # band numbering ("where the band is Band 4, clause 20.2 applies").
    if fam == "long_policy" and sp.get("schema") == "banded" and (sp.get("p") or {}).get("dom") == "public_sector":
        out.append("D_LP_BAND")
    # D_RX_EXCL_SCOPE: rule_exception action, excluded category without the re-admit flag -> gold "deny", but the clause
    # "{cat}: excluded from this section, except where ..." also reads as "this section gives no action".
    if fam == "rule_exception" and sp.get("kind") == "action":
        pol, rec = sp.get("policy") or {}, sp.get("record") or {}
        fe = pol.get("f_exc")
        if rec.get("cat") == pol.get("excl") and fe is not None and rec.get("flags") and rec["flags"][fe] is False \
                and "excluded from this section" in st:
            out.append("D_RX_EXCL_SCOPE")
    return out


def rules_for(r: dict) -> list[str]:
    fam = r.get("family")
    if fam == "gui_action":
        return gui_rules(r)
    if fam == "image_joint_rule":
        return ij_rules(r)
    if (r.get("provenance") or {}).get("spec") and r.get("source") in ("A", "D"):
        return d_rules(r)
    return []


def scan(path: str, flags_fh=None):
    counts, denom, ids = collections.Counter(), collections.Counter(), collections.defaultdict(set)
    for line in open(ROOT / path):
        r = json.loads(line)
        fam = r.get("family")
        if not (fam in ("gui_action", "image_joint_rule") or r.get("source") in ("A", "D")):
            continue
        if r.get("source") == "A" and not (r.get("provenance") or {}).get("spec"):
            continue
        stratum = f'{r.get("source")}/{fam}/{"unk" if r.get("gold") is None else "ans"}'
        denom[stratum] += 1
        for code in rules_for(r):
            counts[(stratum, code)] += 1
            ids[code].add(r["id"])
            if flags_fh:
                flags_fh.write(json.dumps({"file": path, "id": r["id"], "family": fam, "gold": r.get("gold"), "defect": code}) + "\n")
    return counts, denom, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=TRAIN_FILES + EVAL_FILES)
    ap.add_argument("--flags", default=None, help="write one JSONL line per flagged row")
    ap.add_argument("--drop-out", default=None, help="write {ids, why} for DROP rules over the training files")
    ap.add_argument("--json", default=None, help="write the per-file counts as JSON")
    a = ap.parse_args()
    fh = open(a.flags, "w") if a.flags else None
    report, drop = {}, set()
    for f in a.files:
        if not (ROOT / f).is_file():
            print(f"missing {f}")
            continue
        counts, denom, ids = scan(f, fh)
        report[f] = {"rows": dict(denom), "flags": {f"{s} {c}": n for (s, c), n in sorted(counts.items())}}
        print(f"\n{f}")
        for (s, c), n in sorted(counts.items()):
            print(f"  {s:38s} {c:18s} {n:6d} / {denom[s]}")
        if f in TRAIN_FILES:
            for c, s in ids.items():
                if c in DROP_RULES:
                    drop |= s
    if fh:
        fh.close()
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    if a.drop_out:
        Path(a.drop_out).write_text(json.dumps({"ids": sorted(drop), "rules": sorted(DROP_RULES)}, indent=1))
        print(f"\n{len(drop)} drop ids -> {a.drop_out}")


if __name__ == "__main__":
    main()
