import base64
from io import BytesIO
import struct
import zlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageFile

from playground.server import create_app
from vision_decision.images import load_image_bytes


def oversized_png(side):
    """A 69-byte PNG with an oversized IHDR; no large pixel buffer is created."""
    output = BytesIO()
    Image.new("RGB", (1, 1)).save(output, format="PNG")
    data = bytearray(output.getvalue())
    data[16:24] = struct.pack(">II", side, side)
    data[29:33] = struct.pack(">I", zlib.crc32(data[12:29]))
    return bytes(data)


@pytest.mark.parametrize("side", [10000, 20000])
def test_pillow_pixel_guards_become_image_limit_errors(side, monkeypatch):
    data = oversized_png(side)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 89_478_485)

    def forbid_pixel_loading(*args, **kwargs):
        raise AssertionError("Oversized images must be rejected before loading pixels")

    monkeypatch.setattr(ImageFile.ImageFile, "load", forbid_pixel_loading)
    with pytest.raises(ValueError, match="20 million decoded pixels"):
        load_image_bytes(data)


@pytest.mark.parametrize("side", [10000, 20000])
def test_pillow_pixel_guards_return_http_413(side, monkeypatch):
    data = oversized_png(side)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 89_478_485)

    class Backend:
        model = "stub"

        def score(self, *args, **kwargs):
            raise AssertionError("Oversized images must not reach scoring")

    response = TestClient(create_app(Backend(), examples=[]), raise_server_exceptions=False).post(
        "/v1/systemone", json={"images": ["data:image/png;base64," + base64.b64encode(data).decode()],
                               "questions": {"q": {"type": "noul", "instructions": "Does it apply?"}}})
    assert response.status_code == 413
    assert response.json()["error"] == "bad_image"
    assert "20 million decoded pixels" in response.json()["detail"]
