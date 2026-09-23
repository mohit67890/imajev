"""Independent licence spot-checks for the two decision-v2 image sources.

Neither check is decorative:

  flickr    Open Images states its photographs are CC BY 2.0 but explicitly disclaims any warranty
            about each individual photo ("we make no representations or warranties regarding the
            license status of each image and you should verify the license for each image
            yourself").  This samples selected photos, fetches the Flickr photo page named in the
            dataset's own manifest, and records which Creative Commons licence that page links to.
            It is a SAMPLE, not a per-photo verification, and the README says so.

  commons   the Commons licence for every admitted file was read from the API at collection time;
            this stores a sample of those API responses verbatim, so the licence claim in the
            records can be checked against the upstream wording rather than against our summary.

    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/spotcheck_licenses.py flickr --sample 40
    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/spotcheck_licenses.py commons --sample 20
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from v2.common import OUT, RAW, get_with_backoff, read_jsonl, session

CC = re.compile(r"creativecommons\.org/licenses/([a-z-]+)/([0-9.]+)")


def flickr(sample: int, seed: int, pause: float) -> dict:
    rows = read_jsonl(RAW / "openimages_v2" / "selection.jsonl")
    rng = random.Random(seed)
    picked = rng.sample(rows, min(sample, len(rows)))
    sess = session(pool=2)
    results, counts = [], Counter()
    for row in picked:
        url = row["flickr_landing_url"]
        try:
            r = get_with_backoff(sess, url, timeout=(20, 60), tries=3)
            found = sorted({f"CC {a.upper()} {b}" for a, b in CC.findall(r.text)})
            status = r.status_code
        except Exception as exc:
            found, status = [], f"error: {exc}"[:120]
        verdict = "CC BY 2.0" if found == ["CC BY 2.0"] else ("mixed/other" if found else "not readable")
        counts[verdict] += 1
        results.append({"image_id": row["image_id"], "flickr_landing_url": url,
                        "http": status, "licenses_linked_on_page": found, "verdict": verdict})
        time.sleep(pause)
    out = {
        "checked": len(results), "verdicts": dict(counts), "sampled_with_seed": seed,
        "what_this_shows": "A sample of Open Images test photos whose Flickr page still links a "
                           "Creative Commons licence. It does not verify every photo; Google's own "
                           "note disclaims per-image licence warranties.",
        "results": results,
    }
    path = RAW / "openimages_v2" / "flickr_license_spotcheck.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    (OUT / "licenses" / "openimages_v2" / "flickr-license-spotcheck.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("checked", "verdicts")}, indent=2))
    return out


def commons(sample: int, seed: int, pause: float) -> dict:
    rows = read_jsonl(RAW / "commons_photos" / "metadata.jsonl")
    rng = random.Random(seed)
    by_spdx: dict[str, list] = {}
    for row in rows:
        by_spdx.setdefault(row["spdx"], []).append(row)
    picked = []
    for spdx, group in sorted(by_spdx.items()):
        picked += rng.sample(group, min(max(sample // max(1, len(by_spdx)), 1), len(group)))
    sess = session(pool=2)
    samples = []
    for row in picked:
        data = get_with_backoff(sess, "https://commons.wikimedia.org/w/api.php", params={
            "action": "query", "format": "json", "formatversion": "2", "titles": row["title"],
            "prop": "imageinfo", "iiprop": "extmetadata|url|size|mime",
            "iiextmetadatafilter": "License|LicenseShortName|UsageTerms|LicenseUrl|Artist|Credit|Restrictions",
        }).json()
        page = (data.get("query", {}).get("pages") or [{}])[0]
        info = (page.get("imageinfo") or [{}])[0]
        samples.append({
            "title": row["title"], "recorded_spdx": row["spdx"],
            "api_extmetadata": info.get("extmetadata", {}),
            "file_page": info.get("descriptionurl", ""),
        })
        time.sleep(pause)
    out = {
        "checked": len(samples), "sampled_with_seed": seed,
        "what_this_shows": "Verbatim Commons API licence blocks for a sample of admitted files, one "
                           "batch per SPDX id. Every admitted file was screened on exactly these "
                           "fields (LicenseShortName plus the machine-readable License tag).",
        "samples": samples,
    }
    path = OUT / "licenses" / "commons_photos" / "commons-api-license-samples.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    mismatches = [s["title"] for s in samples
                  if s["api_extmetadata"].get("LicenseShortName", {}).get("value") not in
                  {"CC0", "CC0 1.0", "CC BY 3.0", "CC BY 4.0"}]
    print(json.dumps({"checked": len(samples), "unexpected_license_short_names": mismatches}, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["flickr", "commons"])
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--pause", type=float, default=1.2)
    args = ap.parse_args()
    (flickr if args.which == "flickr" else commons)(args.sample, args.seed, args.pause)
