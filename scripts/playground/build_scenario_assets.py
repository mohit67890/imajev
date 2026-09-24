"""Copy the scenario photos into static/scenarios/assets/ and write their provenance.

The scenarios page (static/scenarios/) only shows images listed here. Each entry records the
dataset, licence and whether the photo appears in a *training* row of any manifest the released
2B was trained on (v1 → v1.1 → v2 → v2.1, each run continued from the previous adapter), so the
page can say truthfully which photos the model never saw in training.

    .venv/bin/python scripts/playground/build_scenario_assets.py              # scenarios page
    .venv/bin/python scripts/playground/build_scenario_assets.py --wardrobe   # wardrobe page credits

The wardrobe mode writes 768-pixel demo copies of the catalogue pieces the wardrobe scenarios use
(static/wardrobe/assets/demo/, smaller images keep a flip under a second and inside the model's image
limit) and their credits from catalog.json provenance. The training-row check runs on the unchanged
originals in static/wardrobe/assets/, and ABO pieces are also matched by product ID.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "static/scenarios/assets"
MAX_EDGE = 1024
TRAIN_MANIFESTS = ["decision-v1.jsonl", "decision-v1.1.jsonl", "decision-v2.jsonl", "decision-v2.1.jsonl"]

CC_BY = "https://creativecommons.org/licenses/by/4.0/"
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"
ABO = {"dataset": "Amazon Berkeley Objects (ABO)", "license": "CC-BY-4.0", "license_url": CC_BY,
       "credit": "Amazon Berkeley Objects, CC BY 4.0"}
VISA = {"dataset": "VisA (Visual Anomaly), pipe_fryum", "license": "CC-BY-4.0", "license_url": CC_BY,
        "credit": "VisA dataset (Amazon), CC BY 4.0"}
VIZWIZ = {"dataset": "VizWiz (photos taken by blind users)", "license": "CC-BY-4.0", "license_url": CC_BY,
          "credit": "VizWiz, CC BY 4.0"}
PD12M = {"dataset": "PD12M (Spawning), CC0 rows via Wikimedia Commons", "license": "CC0-1.0", "license_url": CC0,
         "credit": "PD12M / Wikimedia Commons, CC0"}

ASSETS = [
    ("loafers.jpg", "data/decision-v1/abo/images/0e7a7926f538766beeb5e5ee3c3ea1474be10fa616d778b514a89a30965b1df0.jpg", ABO, None),
    ("slipon-listing.jpg", "data/decision-v1/abo/images/8e3ee0bec7cfd287e0a4b87500cec3beb9834b675faed7d441060f7d429a510e.jpg", ABO, None),
    ("slipon-return.jpg", "data/decision-v1/abo/images/c70e1d14dc2fc47f9979fabcb1f130b9d86f06d12b2a3980b35a828630691334.jpg", ABO, None),
    ("fryum-reference.jpg", "data/decision-v1/defects/images/97ec5a92f9c12a7ff0df90514aed7b3333a5ae80f30f8bf5527cdf76ea22ee33.jpg", VISA, None),
    ("fryum-chipped.jpg", "data/decision-v1/defects/images/5876e231713139e095b5f91e93e353c48ff3afead885cb2f13b6330373d00385.jpg", VISA, None),
    ("fryum-good.jpg", "data/decision-v1/defects/images/5da8ee962e24009187a3ea33145f2dadb0a2ec481b12c1929623fffe2ec940be.jpg", VISA, None),
    ("can-lid.jpg", "data/decision-v1/vizwiz_quality/images/58598a15a76f1ec0152758d07f592ff7e65c64ec8ce18a6a8a0db130f3073e4f.jpg", VIZWIZ, None),
    ("tray-before.jpg", "data/decision-v2/pairs_grounded/images_edited/2319d707d4d100817d8bf1e03b9e0ed893841cd87e598e08d86741d2d2c4469b.jpg", PD12M,
     "composite: small CC0 cut-outs pasted into the original photo (imajev pairs_grounded); the 'after' photo is the unedited original"),
    ("tray-after.jpg", "data/decision-v2/pd12m/images/68b2f477b5e73c33e14fb2feb50612899049635a9016ebb8d4a7d4718305e0a9.jpg", PD12M, None),
    ("cat-on-chair.jpg", "../imajev-release/space/examples/cat-on-chair.jpg", PD12M, None),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trained_on(digest):
    """Manifests (of TRAIN_MANIFESTS) holding a train-partition row with this image hash."""
    hits = []
    for name in TRAIN_MANIFESTS:
        manifest = ROOT / "data/manifests" / name
        if not manifest.is_file():
            hits.append(f"{name}: missing, not checked")
            continue
        found = subprocess.run(["grep", "-F", digest, str(manifest)], capture_output=True, text=True).stdout
        if any(json.loads(line).get("partition") == "train" for line in found.splitlines() if line):
            hits.append(name)
    return hits


def trained_on_many(digests):
    """{digest: [manifests with a train-partition row holding it]} with one grep pass per manifest."""
    hits = {d: [] for d in digests}
    patterns = OUT.parent / ".digests.txt"
    patterns.write_text("\n".join(digests) + "\n")
    try:
        for name in TRAIN_MANIFESTS:
            manifest = ROOT / "data/manifests" / name
            found = subprocess.run(["grep", "-F", "-f", str(patterns), str(manifest)], capture_output=True, text=True).stdout
            for line in found.splitlines():
                row = json.loads(line)
                if row.get("partition") != "train":
                    continue
                for digest in digests:
                    if digest in line and name not in hits[digest]:
                        hits[digest].append(name)
    finally:
        patterns.unlink()
    return hits


# Second-hand photos show the whole inspection table (tape measures, cables): crop to the garment.
WARDROBE_CROPS = {"r13.jpg": (300, 170, 880, 600), "r04.jpg": (380, 0, 900, 660)}


def wardrobe():
    folder = Path(__file__).resolve().parent / "static/wardrobe"
    demo = folder / "assets/demo"
    demo.mkdir(exist_ok=True)
    stems = sorted(set(re.findall(r"'assets/demo/([\w-]+)\.jpg'", (folder / "scenarios.js").read_text()))
                   | set(re.findall(r"\{ id: '(\w\d\d)'", (folder / "stylist.js").read_text())))
    catalog = {Path(item["image"]).stem: item for item in json.loads((folder / "catalog.json").read_text())["items"]}
    used = [Path(catalog[stem]["image"]).name for stem in stems]
    catalog = {Path(item["image"]).name: item for item in catalog.values()}
    digests = {name: sha256(folder / "assets" / name) for name in used}
    for name in used:
        image = Image.open(folder / "assets" / name).convert("RGB")
        if name in WARDROBE_CROPS:
            image = image.crop(WARDROBE_CROPS[name])
        image.thumbnail((768, 768))
        image.save(demo / f"{Path(name).stem}.jpg", quality=90)
    # Training used resized copies, so a hash can miss the same photo; ABO pieces are also matched by product ID.
    products = {name: catalog[name]["provenance"]["source_id"] for name in used
                if catalog[name]["provenance"].get("source_name") == "abo"}
    hits = trained_on_many(list(digests.values()) + list(products.values()))
    rows = []
    for name in used:
        item, prov = catalog[name], catalog[name]["provenance"]
        synthetic = bool(prov.get("synthetic"))
        if synthetic:
            meta = {"dataset": "generated for imajev (image_gen)", "license": "generated", "license_url": None,
                    "credit": "AI-generated catalogue image, made for this demo", "prompt": prov.get("prompt")}
        else:
            meta = {"dataset": prov.get("source_name"), "license": prov.get("license"), "source_page": prov.get("source_page"),
                    "credit": f"{prov.get('attribution', prov.get('source_name'))}, {prov.get('license')}"}
        found = sorted(set(hits[digests[name]] + hits.get(products.get(name), [])))
        rows.append({"file": f"{Path(name).stem}.jpg", "derived_from": f"assets/{name} ({'cropped to the garment and ' if name in WARDROBE_CROPS else ''}resized to 768 px)", "title": item["title"], **meta, "synthetic": synthetic, "edit": None,
                     "source_sha256": digests[name], "in_training_rows": found, "held_out": not found})
        print(f"{name:8s} {str(meta['license']):22s} synthetic={synthetic} held_out={not found}")
    (demo / "attribution.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    if "--wardrobe" in sys.argv:
        return wardrobe()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, source, meta, edit in ASSETS:
        src = (ROOT / source).resolve()
        digest = sha256(src)
        image = Image.open(src).convert("RGB")
        image.thumbnail((MAX_EDGE, MAX_EDGE))
        image.save(OUT / name, quality=90)
        hits = trained_on(digest)
        rows.append({"file": name, **meta, "source_path": source, "source_sha256": digest,
                     "width": image.width, "height": image.height, "edit": edit,
                     "in_training_rows": hits, "held_out": not hits})
        print(f"{name:22s} {meta['license']:10s} held_out={not hits} {hits}")
    (OUT / "attribution.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
