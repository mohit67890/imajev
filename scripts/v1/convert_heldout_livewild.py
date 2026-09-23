"""heldout_livewild: LIVE In the Wild (CLIVE) -> ordinal image-quality records, all partition "test".

CLIVE is 1,162 authentically distorted photographs rated by >8,100 crowd workers (~175 ratings per
image). Its terms are the most permissive in the whole IQA survey: UT Austin grants use, copying,
modification and distribution "for any purpose" provided the copyright notice travels with it, so
the notice is reproduced verbatim in this source's README.

The release ships a mean opinion score and a standard deviation per image and *not* the per-image
rating histogram, so no `target_distribution` is written: a Gaussian reconstructed from MOS and
sigma would be a modelling assumption, not source data. The MOS is on the 0-100 continuous bar
whose printed anchors are the five ACR labels, so it is binned into those five ordinal levels.

About 17% of records offer a contiguous sub-range of the scale that excludes the true level: an
honest `not_listed`, since the right rating is genuinely not among the offered levels.

Run: PYTHONPATH=src .venv/bin/python scripts/v1/convert_heldout_livewild.py
"""
import hashlib
import random
import sys
from pathlib import Path

from scipy.io import loadmat

sys.path.insert(0, "scripts/v1")
from _common_textrich import CACHE, ImageStore, make_record, ordinal_field, write  # noqa: E402

SOURCE = "heldout_livewild"
SEED = 20260922
NOT_LISTED_RATE = 0.17
LICENSE = "Permissive custom grant (UT Austin LIVE: use/copy/modify/distribute for any purpose, notice required)"
ROOT = CACHE / SOURCE / "x" / "LIVEC"

LEVELS = {1: "bad", 2: "poor", 3: "fair", 4: "good", 5: "excellent"}
QUESTIONS = [
    "How would a viewer rate the overall picture quality of this photograph?",
    "Judging the photo as a whole, what is its visual quality?",
    "Rate the technical quality of this image on the scale shown.",
    "How good does this picture look?",
    "What quality rating best matches this photograph?",
    "On the quality scale offered, where does this photo sit?",
]


def band(mos: float) -> int:
    return max(1, min(5, int(mos // 20) + 1))


def main():
    rng = random.Random(SEED)
    store = ImageStore(SOURCE)
    data = ROOT / "Data"
    names = [str(x[0][0]) for x in loadmat(data / "AllImages_release.mat")["AllImages_release"]]
    mos = [float(v) for v in loadmat(data / "AllMOS_release.mat")["AllMOS_release"].ravel()]
    std = [float(v) for v in loadmat(data / "AllStdDev_release.mat")["AllStdDev_release"].ravel()]

    items = [(n, m, s) for n, m, s in zip(names, mos, std) if not Path(n).stem.startswith("t")]
    items.sort(key=lambda it: hashlib.sha256(f"{SEED}:{it[0]}".encode()).hexdigest())

    records = []
    for name, score, sigma in items:
        path = ROOT / "Images" / name
        if not path.is_file():
            continue
        gold = band(score)
        abstain = rng.random() < NOT_LISTED_RATE
        if abstain:
            below = [v for v in LEVELS if v < gold]
            above = [v for v in LEVELS if v > gold]
            side = [s for s in (below, above) if len(s) >= 2]
            if not side:
                abstain = False
            else:
                window = rng.choice(side)
        if not abstain:
            width = rng.choice([5, 5, 5, 4, 3])
            start = rng.randint(max(1, gold - width + 1), min(gold, 5 - width + 1))
            window = list(range(start, start + width))
        levels = [(v, LEVELS[v]) for v in sorted(window)]
        field, target, cause = ordinal_field(rng.choice(QUESTIONS), levels, gold, abstain=abstain)
        img = store.add(path.read_bytes(), name)
        records.append(make_record(
            source=SOURCE, uid=Path(name).stem, source_split="test", source_group=img["sha256"],
            family="image_quality", license=LICENSE, images=[img], field=field, target=target,
            cause=cause, source_answer=f"MOS {score:.2f} (sd {sigma:.2f}) on 0-100 -> level {gold}",
            partition="test"))
    write(SOURCE, records)


if __name__ == "__main__":
    main()
