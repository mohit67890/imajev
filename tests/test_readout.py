import json
from pathlib import Path

import pytest

from vision_decision.scoring import MAX_READOUT_CODES, readout_codes


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        prefix, _, tail = text.partition("#")
        ids = [ord(x) for x in prefix] + [1]
        if tail:
            # Make BQ invalid and all other one/two-uppercase-letter codes atomic.
            value = sum((ord(c) - 64) * (26 ** i) for i, c in enumerate(reversed(tail)))
            ids += [ord(tail[0]), ord(tail[1])] if tail == "BQ" else [1000 + value]
        return ids


def test_codebook_skips_non_single_token_pairs_and_is_bounded():
    codes = readout_codes(TinyTokenizer(), "prefix#", MAX_READOUT_CODES)
    assert [x[0] for x in codes[:26]] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert "BQ" not in {x[0] for x in codes}
    assert len(codes) == len({x[1] for x in codes}) == 255
    with pytest.raises(ValueError, match="1..255"):
        readout_codes(TinyTokenizer(), "prefix#", 256)


def test_pinned_qwen_codebook_is_reproducible():
    transformers = pytest.importorskip("transformers")
    bundle = json.loads(Path("artifacts/model.json").read_text())
    path = Path(bundle["path"])
    if not path.is_dir():
        pytest.skip("pinned local Qwen snapshot is absent")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    codes = readout_codes(tokenizer, "</think>\n\n", 255)
    assert [x[0] for x in codes[:26]] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert codes[-1][0] == "JT"
    assert len({token for _, token in codes}) == 255
