"""Helpers for scripts/p3/build_manifest.py: verified candidate -> trainer record, label contract, stratified selection.

Trainer record format (scripts/decision_data.py load_records/render; e.g. data/manifests/decision-p2b.jsonl):
    {"id", "source", "source_split", "source_group", "partition", "family", "domain", "license": {"spdx", ...},
     "images": [{"image": "data/...", "sha256", "width", "height"}],
     "request": {"schema_version": "1.0", "request_id", "state", "fields": [<one field>]},
     "target": <value | None>, "abstention_cause": <reason | None>,
     "target_probs": {label: p} (optional; labels "true"/"false", option values, level values as strings, "unknown"),
     "rationale": "<= 2 sentences" (optional), "mix": hard_text | hard_image | replay_text | replay_image, "p3": {...}}

Candidate -> record mapping (inverse of scripts/p3/convert_own_text.py for C rows):
    noul   -> boolean field (yes/no descriptions kept);           gold True/False -> target
    choice -> choice field, option value = candidate key; description = the option text when it differs from the key,
              then the option description ("text — description"), so the model reads what the generator showed;
    score  -> ordinal field; level values = provenance.level_values when present (C-text p2 levels start at 1), else 0..n-1;
              target = the level VALUE (lv[gold]).
    gold null -> target None + abstention_cause = unknown_reason.

Label contract for teacher-verified rows (the streamed teacher writes one JSON row per KEPT item):
    either a full candidate row (variants, unknown variants, anything not in the pool) or {"id": <pool id>} joined with the pool,
    or a teacher result row {"id", "keep", "item": <candidate row>, "label": ..., ...} (scripts/p3/azure_teacher.py results.jsonl /
    kept.jsonl / heldout-results.jsonl; see `unwrap_verified`),
    plus "label": {"target": <candidate-space answer: option key | true/false | level index (0-based) | null for unknown>,
                   "probs": {<candidate-space label>: p} | null,   # "true"/"false", option keys, "0".."n-1", "unknown"
                   "rationale": str | null,
                   "target_kind": "teacher" (train toward probs) | "gold" (hard constructed gold, teacher disagreed; capped at
                                  25% of A) | "constructed" (a generator variant, gold by construction, no teacher; not capped),
                   "review": {"verdict": "correct" | ..., "verified_by": "owner" | "kimi"} | null}
    Top-level "probs"/"rationale"/"target" keys (gen_answer.py distribution rows) are accepted as a flat label.
Producers: azure_teacher.py (kept.jsonl, heldout-results.jsonl), variants.py (variants/constructed.jsonl), review_report.py
(review/dev-slice.jsonl). Consumers: build_manifest.py, kimi_review.py.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "scripts/p2", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import assembly_common as ac  # noqa: E402

UNKNOWN_LABEL = "unknown"
MAX_VALUE_CHARS = 128        # contracts.Option.value
MAX_TEXT_CHARS = 2000        # contracts.Text (question, option/level descriptions)
_IMG_META: dict[str, tuple[int, int]] = {}


# ------------------------------------------------------------------------------------------------ labels
def candidate_labels(cand: dict) -> list[str]:
    f = cand["field"]
    if f["type"] == "noul":
        labs = ["true", "false"]
    elif f["type"] == "choice":
        labs = [str(o["key"]) for o in f["options"]]
    else:
        labs = [str(l["value"]) for l in f["levels"]]
    return labs + [UNKNOWN_LABEL]


def to_label(cand: dict, value) -> str:
    """candidate-space value (key / bool / level index / None) -> label string."""
    if value is None:
        return UNKNOWN_LABEL
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def unwrap_verified(row: dict) -> dict | None:
    """A teacher result row ({"item": <candidate>, "label": ..., "keep": ...}) -> the candidate row with its label; None when the
    teacher dropped it. Any other row is returned unchanged."""
    if "item" not in row or not isinstance(row.get("item"), dict):
        return row
    if row.get("keep") is False:
        return None
    out = dict(row["item"])
    out["label"] = row.get("label")
    return out


def parse_label(row: dict) -> dict:
    """The row's label object, normalised: {target, probs, rationale, target_kind, review}."""
    lab = row.get("label")
    if lab is None:
        lab = {k: row[k] for k in ("target", "probs", "rationale", "target_kind", "review") if k in row}
        if "target" not in lab and "value" in row:
            lab["target"] = row["value"]
    if not isinstance(lab, dict):
        raise ValueError("label is not an object")
    return {"target": lab.get("target", "__missing__"), "probs": lab.get("probs"), "rationale": lab.get("rationale"),
            "target_kind": lab.get("target_kind") or ("teacher" if lab.get("probs") else "gold"), "review": lab.get("review")}


def _norm_target(cand: dict, t):
    """Accept the candidate-space target in label or value form; returns key / bool / level index / None."""
    f = cand["field"]
    if t is None or t == UNKNOWN_LABEL:
        return None
    if f["type"] == "noul":
        if isinstance(t, bool):
            return t
        if str(t).lower() in ("true", "false"):
            return str(t).lower() == "true"
        raise ValueError(f"bad noul target {t!r}")
    if f["type"] == "score":
        v = int(t)
        if not 0 <= v < len(f["levels"]):
            raise ValueError(f"level index {t!r} out of range")
        return v
    keys = [o["key"] for o in f["options"]]
    if t not in keys:
        raise ValueError(f"target {t!r} is not an option key")
    return t


def normalise_probs(cand: dict, probs: dict) -> dict:
    """Candidate-space probabilities over every label (+ unknown), renormalised; unknown labels are an error."""
    labs = candidate_labels(cand)
    out = {l: 0.0 for l in labs}
    for k, v in probs.items():
        k2 = to_label(cand, k) if isinstance(k, bool) or k is None else str(k)
        if k2 in ("__unknown__", "null"):
            k2 = UNKNOWN_LABEL
        if cand["field"]["type"] == "noul":
            k2 = k2.lower()
        if k2 not in out:
            raise ValueError(f"probability for unknown label {k!r}")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"bad probability {v!r}")
        out[k2] += float(v)
    tot = sum(out.values())
    if tot <= 0:
        raise ValueError("probabilities sum to 0")
    return {k: v / tot for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ record conversion
def level_values(cand: dict) -> list[int]:
    lv = (cand.get("provenance") or {}).get("level_values")
    n = len(cand["field"]["levels"])
    if isinstance(lv, list) and len(lv) == n and all(isinstance(x, int) and not isinstance(x, bool) for x in lv):
        return lv
    return list(range(n))


def option_description(o: dict) -> str | None:
    text, desc = (o.get("text") or "").strip(), (o.get("description") or "").strip()
    show_text = text and text != str(o["key"])
    if show_text and desc:
        return f"{text} — {desc}"
    return text if show_text else (desc or None)


def to_field(cand: dict) -> dict:
    f = cand["field"]
    q = f["question"]
    if len(q) > MAX_TEXT_CHARS:
        raise ValueError("question over 2000 chars")
    if f["type"] == "noul":
        out = {"id": "decision", "type": "boolean", "question": q}
        for k in ("yes_description", "no_description"):
            if f.get(k):
                out[k] = f[k][:MAX_TEXT_CHARS]
        return out
    if f["type"] == "choice":
        opts = []
        for o in f["options"]:
            v = str(o["key"])
            if len(v) > MAX_VALUE_CHARS:
                raise ValueError("option key over 128 chars")
            d = option_description(o)
            if d and len(d) > MAX_TEXT_CHARS:
                raise ValueError("option description over 2000 chars")
            opts.append({"value": v, **({"description": d} if d else {})})
        return {"id": "decision", "type": "choice", "question": q, "options": opts}
    lv = level_values(cand)
    levels = []
    for l, v in zip(f["levels"], lv):
        d = (l.get("description") or "").strip() or str(v)
        if len(d) > MAX_TEXT_CHARS:
            raise ValueError("level description over 2000 chars")
        levels.append({"value": v, "description": d})
    return {"id": "decision", "type": "ordinal", "question": q, "levels": levels}


def to_value(cand: dict, v):
    """candidate-space value -> record target value."""
    if v is None or cand["field"]["type"] != "score":
        return v
    return level_values(cand)[v]


def to_record_probs(cand: dict, probs: dict) -> dict:
    """candidate-space label probabilities -> record label probabilities (level indices -> level values as strings)."""
    if cand["field"]["type"] != "score":
        return dict(probs)
    lv = level_values(cand)
    return {(k if k == UNKNOWN_LABEL else str(lv[int(k)])): p for k, p in probs.items()}


def image_entry(rel: str) -> dict:
    path = rel if rel.startswith("data/") else f"data/{rel}"
    full = ROOT / path
    if path not in _IMG_META:
        try:
            from PIL import Image
            with Image.open(full) as im:
                _IMG_META[path] = im.size
        except Exception:
            _IMG_META[path] = (640, 480)
    w, h = _IMG_META[path]
    return {"image": path, "sha256": Path(path).stem if len(Path(path).stem) == 64 else ac.sha256_file(full), "width": w, "height": h}


def request_id(cid: str) -> str:
    return cid if len(cid) <= 128 else cid[:100] + "-" + hashlib.sha256(cid.encode()).hexdigest()[:16]


def candidate_to_record(cand: dict, label: dict | None, *, group: str, partition: str, mix: str, image_meta: bool = True) -> dict:
    """A trainer record for one candidate row. label None = the candidate's own gold (held-out/dev sets, large_choice)."""
    pv = cand.get("provenance") or {}
    field = to_field(cand)
    if label is None:
        tgt, probs, rationale, kind, review = cand["gold"], None, None, "gold", None
    else:
        tgt = cand["gold"] if label["target"] == "__missing__" else _norm_target(cand, label["target"])
        probs = normalise_probs(cand, label["probs"]) if label.get("probs") else None
        rationale, kind, review = label.get("rationale"), label.get("target_kind"), label.get("review")
    rec = {
        "id": cand["id"], "source": f"p3_{cand['source']}", "source_split": "p3", "source_group": group, "partition": partition,
        "family": cand["family"], "domain": pv.get("domain"),
        "license": {"spdx": pv.get("licence"), **({"attribution": pv["attribution"]} if pv.get("attribution") else {})},
        "images": [image_entry(p) if image_meta else {"image": p if p.startswith("data/") else f"data/{p}"}
                   for p in cand.get("images") or []],
        "request": {"schema_version": "1.0", "request_id": request_id(cand["id"]), "state": cand["state"], "fields": [field]},
        "target": to_value(cand, tgt),
        "abstention_cause": (cand.get("unknown_reason") or "insufficient_evidence") if tgt is None else None,
        "mix": mix,
        "p3": {"source": cand["source"], "dataset": cand["dataset"], "difficulty": cand["difficulty"],
               "gold_kind": cand["gold_kind"], "target_kind": kind, "parent_id": cand.get("parent_id"),
               "n_options": len(cand["field"].get("options") or []), "unknown_gold": cand["gold"] is None,
               **({"review": review} if review else {}), **({"heldout": pv["heldout"]} if pv.get("heldout") else {})},
    }
    if probs:
        rec["target_probs"] = to_record_probs(cand, probs)
    if isinstance(rationale, str) and rationale.strip():
        rec["rationale"] = rationale.strip()
    return rec


def check_record(rec: dict, soft: bool = True) -> str | None:
    """None if the record loads through the trainer path (expand_fields + render, soft targets included), else the error."""
    from decision_data import expand_fields, render
    try:
        for item in expand_fields(rec):
            render(item, soft_targets=soft)
            render(item, soft_targets=False)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e)[:160]}"
    return None


def record_as_candidate(rec: dict) -> dict:
    """A trainer record in candidate shape, for decontam's leakage scan (state, question, options/levels, upstream ids)."""
    f = rec["request"]["fields"][0]
    field = {"type": f["type"], "question": f.get("question", "")}
    if f.get("options"):
        field["options"] = [{"key": o["value"], "text": o["value"], **({"description": o["description"]} if o.get("description") else {})}
                            for o in f["options"]]
    if f.get("levels"):
        field["levels"] = [{"value": l["value"], "description": l.get("description", "")} for l in f["levels"]]
    pv = {}
    for k in ("upstream_dataset", "upstream_id", "upstream_split"):
        if rec.get(k) is not None:
            pv[k] = rec[k]
    return {"id": rec["id"], "state": rec["request"].get("state"), "field": field, "parent_id": None, "provenance": pv,
            "images": [i["image"] for i in rec.get("images") or []]}


def image_question_key(images: list[str], question: str) -> tuple:
    import re
    return (tuple(sorted(Path(p).stem for p in images)), " ".join(re.findall(r"[a-z0-9]+", question.lower())))


# ------------------------------------------------------------------------------------------------ selection
def stratified_take(rows: list, n: int, stratum, must_keep, salt: str) -> list:
    """n rows by proportional stratified sampling: every must-keep row first (if they alone exceed n, they are sampled
    proportionally themselves), then the rest allocated to strata in proportion to their size (largest-remainder rounding),
    stable-hash order inside a stratum. Proportional keeps the supply's shape, which the teacher quotas already weighted."""
    must = [r for r in rows if must_keep(r)]
    rest = [r for r in rows if not must_keep(r)]
    if len(must) >= n:
        return _proportional(must, n, stratum, salt + ":must")
    return must + _proportional(rest, n - len(must), stratum, salt)


def tiered_take(tiers: list[list], n: int, stratum, salt: str) -> list:
    """Fill n from the first tier, then the next (each tier proportionally stratified when it holds more than is needed)."""
    out = []
    for k, tier in enumerate(tiers):
        left = n - len(out)
        if left <= 0:
            break
        out += _proportional(tier, left, stratum, f"{salt}:tier{k}")
    return out


def _rid(r) -> str:
    return r["id"] if isinstance(r, dict) else str(r)


def _proportional(rows: list, n: int, stratum, salt: str) -> list:
    if n >= len(rows):
        return list(rows)
    if n <= 0:
        return []
    by = collections.defaultdict(list)
    for r in rows:
        by[stratum(r)].append(r)
    total = len(rows)
    raw = {k: n * len(v) / total for k, v in by.items()}
    alloc = {k: int(math.floor(x)) for k, x in raw.items()}
    left = n - sum(alloc.values())
    for k in sorted(by, key=lambda k: (-(raw[k] - alloc[k]), ac.stable_hash(salt, "rem", str(k))))[:left]:
        alloc[k] += 1
    out = []
    for k, v in by.items():
        v.sort(key=lambda r: ac.stable_hash(salt, str(k), _rid(r)))
        out += v[:alloc[k]]
    return out


def deepcopy(x):
    return copy.deepcopy(x)


def jdump(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def trainer_problem(cand: dict) -> str | None:
    """None if the candidate, mapped to a record with its own gold, loads through the trainer path (the serving contract:
    state <= 32 KB, option values <= 128 chars, no option aliasing `unknown`/`null`, texts <= 2,000 chars ...)."""
    try:
        rec = candidate_to_record(cand, None, group="g", partition="train", mix="check", image_meta=False)
    except (ValueError, KeyError, TypeError) as e:
        return f"convert: {e}"
    for o in rec["request"]["fields"][0].get("options") or []:
        if str(o["value"]).lower() in ("null", "unknown", "__unknown__"):
            return "option value aliases unknown"
    err = check_record(rec, soft=False)
    if err and "State exceeds" in err:
        return "state over the serving contract's state limit"
    return err
