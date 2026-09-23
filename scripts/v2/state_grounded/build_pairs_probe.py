"""Build `data/manifests/decision-v2-pairs-probe.jsonl`: 60 composited reference/target pairs.

The answers are known because we made the edits (`build_pairs.py`, run with its held-out probe
pass: references from the `test` partition that are in no record).  Known-by-construction is not
the same as fair, though -- an edit can land somewhere nobody would notice -- so every candidate
here was LOOKED AT (contact sheets from `pair_sheets.py --grid`), and the ones where the edit is
invisible, ambiguous, or reads as a different kind of change are listed in `EYEBALL_REJECT` with
the reason.  Every kept row carries `eyeballed: true`.

    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/state_grounded/build_pairs_probe.py --sheets DIR
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/state_grounded/build_pairs_probe.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from v2.common import OUT, ROOT, read_jsonl, verified_license
from v2.state_grounded import build_pairs as BP
from v2.state_grounded import edits as E

CANDIDATES = OUT / BP.COMPOSITE / "probe_candidates.jsonl"
OUT_PATH = ROOT / "data" / "manifests" / "decision-v2-pairs-probe.jsonl"
PER_KIND = {"add_object": 13, "remove_object": 13, "hide_label": 13, "recolour_region": 13}
PER_KIND_DEFAULT = 7
# how many of each edit kind every family gets: 35 relevant / 25 irrelevant overall, and
# `what_changed` sees every answer at least twice
PLAN = {
    "what_changed": {"add_object": 4, "remove_object": 4, "hide_label": 4, "recolour_region": 2,
                     "brightness": 2, "contrast": 1, "small_crop": 2, "mirror": 1},
    "target_still_matches_reference": {"add_object": 3, "remove_object": 3, "hide_label": 3,
                                       "recolour_region": 1, "brightness": 2, "contrast": 2,
                                       "small_crop": 2, "mirror": 2, "corner_object": 2},
    "which_state_field_now_wrong": {"add_object": 3, "remove_object": 4, "hide_label": 3,
                                    "recolour_region": 1, "brightness": 2, "contrast": 2,
                                    "small_crop": 1, "mirror": 2, "corner_object": 2},
}
FAMILIES = ("what_changed", "target_still_matches_reference", "which_state_field_now_wrong")

# key (reference pd12m id) -> why it was dropped after looking at it.  Filled in by eye.
# All 88 shortlisted candidates were looked at on the contact sheets (23 Sept 2026).
EYEBALL_REJECT: dict[str, str] = {
    "9af45a627f8423b4fec4af7dd5bc145b": "remove_object on a bottle packshot: the pasted object "
                                        "in image 1 cannot be made out at normal viewing size",
}


def shortlist():
    rows = read_jsonl(CANDIDATES)
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["meta"]["kind"], []).append(r)
    out = []
    for kind in sorted(by_kind):
        out += by_kind[kind][:PER_KIND.get(kind, PER_KIND_DEFAULT)]
    return out


def main(sheets=None):
    rows = shortlist()
    if sheets:
        from v2.state_grounded.pair_sheets import grid
        sheets = Path(sheets)
        sheets.mkdir(parents=True, exist_ok=True)
        for k in range(0, len(rows), 8):
            grid(rows[k:k + 8], sheets / f"probe-{k // 8:02d}.png")
        print(f"{len(rows)} candidates -> {sheets}")
        return
    licence = verified_license(Path("data/decision-v2/licenses/pd12m/CC0-1.0.deed.html"), "CC0-1.0")
    kept = [r for r in rows if r["key"] not in EYEBALL_REJECT]
    rng = random.Random("pairs-probe")
    pool = {}
    for r in kept:
        pool.setdefault(r["meta"]["kind"], []).append(r)
    out = []
    for family in FAMILIES:
        for kind, n in PLAN[family].items():
            for _ in range(n):
                if not pool.get(kind):
                    raise SystemExit(f"not enough eyeballed {kind} candidates for {family}")
                r = pool[kind].pop(0)
                rec = BP.composite_record(r, family, 0, licence, rng, source="pairs_grounded_probe")
                rec.update({"partition": "test", "heldout_family": True, "probe_family": family,
                            "gold": rec["target"], "eyeballed": True,
                            "pseudo_label": "construction-eyeballed",
                            "rationale": rationale(r, rec), "photo_source": "pd12m"})
                out.append(rec)
    OUT_PATH.write_text("".join(json.dumps(x, ensure_ascii=True) + "\n" for x in out))
    print(json.dumps({"items": len(out), "families": dict(Counter(x["family"] for x in out)),
                      "edit_kinds": dict(Counter(x["edit_kind"] for x in out)),
                      "rejected_by_eye": len(EYEBALL_REJECT),
                      "gold": dict(Counter(str(x["gold"]) for x in out))}, indent=2))


def rationale(r, rec):
    kind = r["meta"]["kind"]
    what = {
        "add_object": "a cut-out of another CC0 photograph's subject was pasted into image 2",
        "remove_object": "image 1 is the photo with a pasted cut-out and image 2 the untouched "
                         "original, so something in image 1 is missing from image 2",
        "recolour_region": f"one object was recoloured from {r['meta'].get('colour_before')} to "
                           f"{r['meta'].get('colour_after')} in image 2",
        "hide_label": f"part of image 2 was {r['meta'].get('how', 'covered')}",
        "brightness": "only the brightness of image 2 changed",
        "contrast": "only the contrast of image 2 changed",
        "small_crop": "image 2 is a slightly tighter crop of the same photograph",
        "mirror": "image 2 is the same photograph mirrored left-right",
        "corner_object": "a tiny unrelated object was placed at the very edge of image 2, which "
                         "the rules in the state say to ignore",
    }[kind]
    return (f"Known by construction: {what}. Relevant change: "
            f"{'yes' if kind in E.RELEVANT_KINDS else 'no'}. Checked by eye on the contact sheet.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default=None)
    main(ap.parse_args().sheets)
