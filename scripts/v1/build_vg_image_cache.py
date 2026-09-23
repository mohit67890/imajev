"""Normalise the Visual Genome images that ship inside the GQA parquet mirror once.

The GQA mirror `lmms-lab-encoder/GQA` embeds the VG photographs GQA uses, keyed by the
Visual Genome image id.  cs.stanford.edu serves the same files at ~3 images/s, so the
parquet is the only affordable route for tens of thousands of them.  This writes
`.cache/datasets/v1/vg_common/proc/<vg_image_id>.jpg`, already RGB / <=1 MP / JPEG q90,
which convert_gqa.py, convert_vg_attributes.py and convert_tallyqa.py all read.
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_vg_group import CACHE_ROOT, VG_PROC, normalise

SHARD_DIRS = [CACHE_ROOT / "gqa/hf/train_balanced_images", CACHE_ROOT / "gqa/hf/val_balanced_images"]


def work(item):
    vg_id, raw = item
    got = normalise(raw)
    if got is None:
        return vg_id, False
    data, _, _ = got
    tmp = VG_PROC / f".{vg_id}.tmp"
    tmp.write_bytes(data)
    tmp.replace(VG_PROC / f"{vg_id}.jpg")
    return vg_id, True


def main():
    VG_PROC.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in VG_PROC.glob("*.jpg")}
    print(f"already cached: {len(have)}")
    written = failed = 0
    with ProcessPoolExecutor() as ex:
        for shard_dir in SHARD_DIRS:
            for shard in sorted(shard_dir.glob("*.parquet")):
                pf = pq.ParquetFile(shard)
                batch_items = []
                for batch in pf.iter_batches(batch_size=256, columns=["id", "image"]):
                    d = batch.to_pydict()
                    for vg_id, img in zip(d["id"], d["image"]):
                        if vg_id in have or img is None or not img.get("bytes"):
                            continue
                        batch_items.append((vg_id, img["bytes"]))
                    if len(batch_items) >= 512:
                        for _, ok in ex.map(work, batch_items, chunksize=16):
                            written += ok
                            failed += not ok
                        batch_items = []
                for _, ok in ex.map(work, batch_items, chunksize=16):
                    written += ok
                    failed += not ok
                print(f"  {shard.name}: written={written} failed={failed}", flush=True)
    print(f"done: written={written} failed={failed} total={len(list(VG_PROC.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
