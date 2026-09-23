import importlib.util
import io
import json
from pathlib import Path

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "collect_public_v2", Path(__file__).resolve().parents[1] / "scripts/imajev_bench/collect_public_v2.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def page(pageid, license="CC BY 4.0", uploaded="2026-08-01T10:00:00Z", taken="2026-07-30 09:00:00",
         artist="Alice Example", title=None, width=3000, height=2000):
    meta = {"LicenseShortName": {"value": license}, "Artist": {"value": f"<a href='x'>{artist}</a>"},
            "DateTimeOriginal": {"value": taken}, "LicenseUrl": {"value": "https://creativecommons.org/licenses/by/4.0"},
            "ImageDescription": {"value": "Cafe menu board"}}
    return {"pageid": pageid, "title": title or f"File:Menu {pageid}.jpg",
            "imageinfo": [{"extmetadata": meta, "mime": "image/jpeg", "width": width, "height": height,
                           "timestamp": uploaded, "user": "uploader", "url": f"https://upload.wikimedia.org/{pageid}.jpg",
                           "thumburl": f"https://upload.wikimedia.org/thumb/{pageid}.jpg"}]}


EXCLUDE = {"pageids": {2}, "titles": set(), "artists": {"bob-trainer"}}


@pytest.mark.parametrize("kwargs,reason", [
    ({"license": "CC BY-SA 4.0"}, "license"),
    ({"uploaded": "2026-01-01T00:00:00Z"}, "too_old"),
    ({"taken": "2019-05-01 12:00:00"}, "too_old"),
    ({"width": 640, "height": 480}, "too_small"),
    ({"title": "File:Company logo 2026.jpg"}, "subject"),
    ({"artist": "Bob Trainer"}, "photographer_in_training"),
])
def test_classify_rejections(kwargs, reason):
    assert collect.classify(page(1, **kwargs), "2026-06-01", 1000, EXCLUDE) == (None, reason)


def test_classify_keeps_recent_licensed_photo_and_excludes_training_pageid():
    row, reason = collect.classify(page(1), "2026-06-01", 1000, EXCLUDE)
    assert reason is None and row["artist"] == "Alice Example" and row["spdx"] == "CC-BY-4.0" and row["taken"] == "2026-07-30"
    assert collect.classify(page(2), "2026-06-01", 1000, EXCLUDE) == (None, "in_training_or_v1")


def test_exclusion_inventory_reads_training_and_v1(tmp_path):
    meta = tmp_path / "metadata.jsonl"
    meta.write_text(json.dumps({"pageid": 7, "title": "File:A.jpg", "artist": "Bob Trainer"}) + "\n")
    registry = tmp_path / "fresh_sources.json"
    registry.write_text('{"page": "https://commons.wikimedia.org/wiki/File:Menu Board.JPG"}')
    inv = collect.exclusion_inventory(meta, [registry])
    assert inv["pageids"] == {7} and "bob-trainer" in inv["artists"] and "File:Menu Board.JPG" in inv["titles"]


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def jpeg(color, size=(1200, 900), gradient=False):
    """Stripes, or a left-to-right gradient; dHash compares structure, not colour."""
    out = io.BytesIO()
    image = Image.new("RGB", size, color)
    if gradient:
        image = Image.linear_gradient("L").rotate(90).resize(size).convert("RGB")
    else:
        for x in range(0, size[0], 50):
            for y in range(size[1]):
                image.putpixel((x, y), (255, 255, 255))
    image.save(out, "JPEG")
    return out.getvalue()


def test_search_sheet_and_download(tmp_path):
    api_pages = [page(1), page(2), page(3, artist="Carol New")]

    def opener(request, timeout):
        url = request.full_url
        assert request.headers["User-agent"].startswith("imajev-bench-v2-collector")
        if url.startswith(collect.API):
            return Response(json.dumps({"query": {"pages": api_pages}}).encode())
        return Response(jpeg((200, 30, 30), gradient=True) if url.endswith("/1.jpg") else jpeg((10, 10, 10), (1500, 1000)))

    http = {"opener": opener, "sleep": lambda s: None}
    candidates = tmp_path / "candidates.jsonl"
    stats = collect.run_search(["menu board"], "2026-06-01", 10, 1000, candidates, EXCLUDE, **http)
    assert stats == {"kept": 2, "in_training_or_v1": 1}
    assert collect.write_sheet(candidates, tmp_path / "sheet.html") == 2
    (tmp_path / "accepted.json").write_text(json.dumps({"accepted": [1, 3]}))
    training = tmp_path / "training"
    training.mkdir()
    Image.open(io.BytesIO(jpeg((10, 10, 10), (1500, 1000)))).save(training / "t.jpg")  # same as page 3's photo
    cache = tmp_path / "cache.json"
    collect.training_hashes(cache, training)
    count, rejected = collect.run_download(candidates, tmp_path / "accepted.json", tmp_path / "out", cache, **http)
    assert count == 1 and rejected == [{"pageid": 3, "reason": "near_duplicate_of_training"}]
    sources = json.loads((tmp_path / "out" / "sources.json").read_text())
    assert sources[0]["source_cluster"] == "commons-alice-example-2026-07-30"
    assert sources[0]["provenance"]["license"] == "CC BY 4.0"
    with Image.open(tmp_path / "out" / sources[0]["path"]) as image:
        assert not image.getexif() and max(image.size) <= 2048


def test_refuses_non_commons_urls():
    with pytest.raises(ValueError):
        collect.http_get("https://example.org/x.jpg")
