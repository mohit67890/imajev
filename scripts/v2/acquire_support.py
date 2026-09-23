"""Acquire customer-support / task-oriented service conversations for decision-v2 (TEXT source).

Three upstreams, each admitted only on a primary licence file in its own repository:

  * schema_guided (SGD)  google-research-datasets/dstc8-schema-guided-dialogue -- LICENSE.txt is
                         the CC BY-SA 4.0 legal code.
  * multiwoz             budzianowski/multiwoz -- LICENSE is the MIT licence.
  * taskmaster2          google-research-datasets/Taskmaster -- TM-2-2020/README.md carries the
                         authors' explicit CC BY 4.0 grant.  There is no standalone LICENSE file
                         in that repository; see the source README for how that was judged.

Product-review corpora were looked for and NOT admitted; the reasons are recorded in the source
README (Bitext customer support is CDLA-Sharing-1.0, outside the allowlist; the Amazon/Yelp
review corpora are research-use-only).

Output
  data/decision-v2-raw/support_reviews/passages.jsonl
  data/decision-v2-raw/support_reviews/manifest.json
  data/decision-v2-raw/support_reviews/files/<sub>/<name>.json   (the downloaded originals)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2.common import get_with_backoff, session  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "decision-v2-raw" / "support_reviews"

SGD_REPO = "google-research-datasets/dstc8-schema-guided-dialogue"
SGD_REV = "master"
MWOZ_REPO = "budzianowski/multiwoz"
MWOZ_REV = "master"
TM_REPO = "google-research-datasets/Taskmaster"
TM_REV = "master"
TM_FILES = ["flights", "food-ordering", "hotels", "movies", "music", "restaurant-search", "sports"]

CAPS = {"schema_guided": 8000, "multiwoz": 7000, "taskmaster2": 15000}


def raw_url(repo: str, rev: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{rev}/{path}"


def fetch(sess, url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        dest.write_bytes(get_with_backoff(sess, url).content)
    return {"url": url, "path": str(dest.relative_to(ROOT)), "bytes": dest.stat().st_size,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}


def transcript(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{who}: {text}" for who, text in turns)


def from_sgd(sess, files: list[dict], cap: int) -> list[dict]:
    out: list[dict] = []
    listing = get_with_backoff(sess, f"https://api.github.com/repos/{SGD_REPO}/contents/train").json()
    names = sorted(x["name"] for x in listing if x["name"].endswith(".json"))
    for name in names:
        if len(out) >= cap:
            break
        rec = fetch(sess, raw_url(SGD_REPO, SGD_REV, f"train/{name}"), RAW / "files" / "schema_guided" / name)
        files.append(rec)
        for dialogue in json.loads((ROOT / rec["path"]).read_text()):
            turns = [(t["speaker"].title(), t["utterance"].strip()) for t in dialogue["turns"] if t.get("utterance")]
            if len(turns) < 4:
                continue
            out.append({
                "id": f"sgd-{dialogue['dialogue_id']}",
                "subsource": "schema_guided",
                "domains": sorted({s.split("_")[0] for s in dialogue.get("services", [])}),
                "services": dialogue.get("services", []),
                "turns": len(turns),
                "transcript": transcript(turns),
                "first_user_turn": next((t for w, t in turns if w.lower() == "user"), ""),
            })
    return out[:cap]


def from_multiwoz(sess, files: list[dict], cap: int) -> list[dict]:
    out: list[dict] = []
    listing = get_with_backoff(sess, f"https://api.github.com/repos/{MWOZ_REPO}/contents/data/MultiWOZ_2.2/train").json()
    names = sorted(x["name"] for x in listing if x["name"].endswith(".json"))
    for name in names:
        if len(out) >= cap:
            break
        rec = fetch(sess, raw_url(MWOZ_REPO, MWOZ_REV, f"data/MultiWOZ_2.2/train/{name}"),
                    RAW / "files" / "multiwoz" / name)
        files.append(rec)
        for dialogue in json.loads((ROOT / rec["path"]).read_text()):
            turns = [(t["speaker"].title(), t["utterance"].strip()) for t in dialogue["turns"] if t.get("utterance")]
            if len(turns) < 4:
                continue
            out.append({
                "id": f"mwoz-{dialogue['dialogue_id'].replace('.json', '')}",
                "subsource": "multiwoz",
                "domains": sorted(dialogue.get("services", [])),
                "services": dialogue.get("services", []),
                "turns": len(turns),
                "transcript": transcript(turns),
                "first_user_turn": next((t for w, t in turns if w.lower() == "user"), ""),
            })
    return out[:cap]


def from_taskmaster(sess, files: list[dict], cap: int) -> list[dict]:
    out: list[dict] = []
    for stem in TM_FILES:
        if len(out) >= cap:
            break
        rec = fetch(sess, raw_url(TM_REPO, TM_REV, f"TM-2-2020/data/{stem}.json"),
                    RAW / "files" / "taskmaster2" / f"{stem}.json")
        files.append(rec)
        for dialogue in json.loads((ROOT / rec["path"]).read_text()):
            turns = [(u["speaker"].title(), u["text"].strip())
                     for u in dialogue.get("utterances", []) if u.get("text")]
            if len(turns) < 4:
                continue
            out.append({
                "id": f"tm2-{dialogue['conversation_id']}",
                "subsource": "taskmaster2",
                "domains": [stem],
                "services": [dialogue.get("instruction_id", stem)],
                "turns": len(turns),
                "transcript": transcript(turns),
                "first_user_turn": next((t for w, t in turns if w.lower() == "user"), ""),
            })
    return out[:cap]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(CAPS), action="append")
    args = ap.parse_args()
    wanted = args.only or sorted(CAPS)
    RAW.mkdir(parents=True, exist_ok=True)
    sess = session()
    files: list[dict] = []
    rows: list[dict] = []
    for sub, fn in (("schema_guided", from_sgd), ("multiwoz", from_multiwoz), ("taskmaster2", from_taskmaster)):
        if sub not in wanted:
            continue
        got = fn(sess, files, CAPS[sub])
        print(f"  {sub}: {len(got)} conversations", file=sys.stderr)
        rows.extend(got)

    # Trim overlong transcripts rather than truncating them silently, and drop duplicate
    # conversation ids (Taskmaster-2 repeats a handful of dialogues across its domain files).
    rows = [r for r in rows if 200 <= len(r["transcript"]) <= 6000]
    seen: set[str] = set()
    unique = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        unique.append(r)
    duplicates = len(rows) - len(unique)
    rows = unique
    rows.sort(key=lambda r: r["id"])
    out = RAW / "passages.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["subsource"]] = counts.get(r["subsource"], 0) + 1
    manifest = {
        "source": "support_reviews",
        "subsources": {
            "schema_guided": {"repo": f"https://github.com/{SGD_REPO}", "ref": SGD_REV, "spdx": "CC-BY-SA-4.0"},
            "multiwoz": {"repo": f"https://github.com/{MWOZ_REPO}", "ref": MWOZ_REV, "spdx": "MIT"},
            "taskmaster2": {"repo": f"https://github.com/{TM_REPO}", "ref": TM_REV, "spdx": "CC-BY-4.0"},
        },
        "counts": counts,
        "files": files,
        "passages": len(rows),
        "duplicate_ids_dropped": duplicates,
        "output": {
            "path": str(out.relative_to(ROOT)),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "bytes": out.stat().st_size,
        },
        "excluded_for_licence": {
            "Bitext/Bitext-customer-support-llm-chatbot-training-dataset": "CDLA-Sharing-1.0, not in the allowlist",
            "Amazon product reviews (McAuley-Lab/Amazon-Reviews-2023, amazon_us_reviews)": "no licence file; upstream terms are research-use-only",
            "Yelp Open Dataset": "Yelp Dataset Licence, research use only",
        },
    }
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"passages": len(rows), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
