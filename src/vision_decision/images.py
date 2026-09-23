import hashlib
import warnings
from pathlib import Path
from PIL import Image, ImageOps

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 20_000_000

def load_image_bytes(data):
    """Decode and validate one image already in memory (uploads, data URLs)."""
    if len(data) > MAX_BYTES:
        raise ValueError("Image exceeds 20 MiB")
    from io import BytesIO
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Only JPEG, PNG, and WebP are supported")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Multi-frame images are unsupported")
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Image exceeds 20 million decoded pixels")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    return image, {"sha256": hashlib.sha256(data).hexdigest(), "width": image.width, "height": image.height, "bytes": len(data)}

def load_image(path):
    path = Path(path)
    with path.open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    return load_image_bytes(data)
