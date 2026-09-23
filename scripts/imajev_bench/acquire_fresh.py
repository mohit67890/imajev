#!/usr/bin/env python3
"""Bounded acquisition of a manually verified Commons photo registry."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

MAX_ASSET = 20 * 1024 * 1024
MAX_TOTAL = 160 * 1024 * 1024
HOSTS = {"commons.wikimedia.org", "upload.wikimedia.org", "thumb.wikimedia.org"}


def check_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.username or parsed.password:
        raise ValueError("Only public HTTPS Wikimedia URLs are allowed")


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, limit):
    check_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "imajev-bench-research/0.1 (small attributed academic evaluation collection)"})
    opener = urllib.request.build_opener(SafeRedirect())
    with opener.open(request, timeout=30) as response:
        check_url(response.url)
        if int(response.headers.get("Content-Length", "0")) > limit:
            raise ValueError("Download exceeds byte cap")
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Download exceeds byte cap")
        return data, {"resolved_url": response.url, "content_type": response.headers.get("Content-Type"),
                      "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"),
                      "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def dhash(image):
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    bits = [pixels[y*9+x] > pixels[y*9+x+1] for y in range(8) for x in range(8)]
    return f"{sum(int(v) << i for i, v in enumerate(bits)):016x}"


def acquire(registry_path, output):
    registry = json.loads(registry_path.read_text())
    entries = registry if isinstance(registry, list) else registry["assets"]
    if len(entries) > 24:
        raise ValueError("This acquisition is capped at 24 source images")
    for entry in entries:
        check_url(entry['file_page'])
        check_url(entry['download_url'])
    output.mkdir(parents=True, exist_ok=False)
    (output / "originals").mkdir()
    (output / "assets").mkdir()
    (output / "receipts").mkdir()
    (output / "registry.json").write_text(json.dumps(registry, indent=2))
    assets, failures, total = [], [], 0
    for index, entry in enumerate(entries):
        asset_id = f"photo-{index+1:03d}"
        try:
            page, page_receipt = fetch(entry["file_page"], min(4 * 1024 * 1024, MAX_TOTAL-total))
            total += len(page)
            license_url = entry["license_url"]
            # The human-researched rights reference must occur in the source receipt.
            if license_url.removeprefix("https:").removeprefix("http:").rstrip("/").encode() not in page:
                raise ValueError("Registered license link is absent from fetched source page")
            (output / "receipts" / f"{asset_id}.html").write_bytes(page)
            blob, receipt = fetch(entry["download_url"], min(MAX_ASSET, MAX_TOTAL-total))
            total += len(blob)
            with Image.open(io.BytesIO(blob)) as original:
                if original.format not in ("JPEG", "PNG", "WEBP") or getattr(original, "is_animated", False):
                    raise ValueError("Unsupported or animated image")
                if original.width * original.height > 80_000_000:
                    raise ValueError("Image exceeds pixel cap")
                image = ImageOps.exif_transpose(original).convert("RGB")
                original_size = list(image.size)
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                # No source filename, GPS, author or EXIF is sent as model evidence.
                path = output / "assets" / f"{asset_id}.jpg"
                image.save(path, "JPEG", quality=95)
                width, height = image.size
                perceptual = dhash(image)
            (output / "originals" / f"{asset_id}.bin").write_bytes(blob)
            info = {"asset_id": asset_id, "source": entry,
                    "image": {"path": f"assets/{asset_id}.jpg", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                    "original_receipt": receipt, "page_receipt": page_receipt,
                    "width": width, "height": height, "original_size": original_size, "dhash64": perceptual,
                    "changes": "EXIF orientation applied, RGB conversion, longest side capped at 1600px, JPEG quality95; metadata stripped",
                    "acquired_at": datetime.now(timezone.utc).isoformat()}
            assets.append(info)
            (output / "receipts" / f"{asset_id}.json").write_text(json.dumps(info, indent=2))
            print(json.dumps({"asset_id": asset_id, "status": "acquired", "bytes": receipt["bytes"]}), flush=True)
        except Exception as exc:
            failures.append({"asset_id": asset_id, "source_id": entry.get("source_id"), "error": str(exc), "error_type": type(exc).__name__})
            print(json.dumps({"asset_id": asset_id, "status": "failed", "error": str(exc)}), flush=True)
        time.sleep(.15)
    near = []
    for i, first in enumerate(assets):
        for second in assets[i+1:]:
            distance = (int(first["dhash64"],16)^int(second["dhash64"],16)).bit_count()
            if distance <= 6:
                near.append({"a": first["asset_id"], "b": second["asset_id"], "hamming_distance": distance})
    manifest = {"status": "acquired_candidates_not_annotated", "source_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
                "acquisition_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "total_download_bytes": total, "assets": assets, "failures": failures, "near_duplicate_candidates": near,
                "freshness_claim": "Acquisition time is new; photographs are existing public works. Prior-project overlap audit required. No pretraining-clean or hidden-test claim."}
    (output / "acquisition.json").write_text(json.dumps(manifest, indent=2))
    lines = ["# Photo attribution", "", "Source works retain the licenses listed below. Model input copies have the changes recorded in acquisition.json. This collection does not relicense the photographs.", ""]
    for asset in assets:
        s=asset["source"]
        lines += [f"- **{asset['asset_id']}**: [{s['source_id']}]({s['file_page']}) — {s['creator']}; [{s['license']}]({s['license_url']}). {asset['changes']}."]
    (output / "ATTRIBUTION.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); acquire(a.registry,a.output)
