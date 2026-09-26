"""Phase 3 item 5: the 256-code readout switch (default off = the shipped 255-code readout).

No model is loaded: the torch engine and the MLX backend run on a tiny fake tokenizer and a random LM head.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from vision_decision.calibration import TemperatureCalibrator, calibration_key, option_count_bucket  # noqa: E402
from vision_decision.contracts import UNKNOWN, Result  # noqa: E402
from vision_decision.jev_api import to_request  # noqa: E402
from vision_decision.scoring import (EXTENDED_READOUT_CODES, MAX_READOUT_CODES, candidates, labels_for_count,  # noqa: E402
                                     max_options, readout_codes, result_from_logits)

HIDDEN, VOCAB = 8, 2000


class TinyTokenizer:
    """Every one- and two-letter code is one token except BQ (as in tests/test_readout.py)."""

    def encode(self, text, add_special_tokens=False):
        prefix, _, tail = text.partition("#")
        ids = [ord(x) for x in prefix] + [1]
        if tail:
            value = sum((ord(c) - 64) * (26 ** i) for i, c in enumerate(reversed(tail)))
            ids += [ord(tail[0]), ord(tail[1])] if tail == "BQ" else [1000 + value]
        return ids


# ------------------------------------------------------------------------------------------- scoring

def test_codebook_is_prefix_stable_and_limit_is_a_switch():
    tok = TinyTokenizer()
    shipped = readout_codes(tok, "p#", 255)
    extended = readout_codes(tok, "p#", 256, limit=EXTENDED_READOUT_CODES)
    assert extended[:255] == shipped and len(extended) == 256 and extended[255] not in shipped
    with pytest.raises(ValueError, match="1..255"):
        readout_codes(tok, "p#", 256)                       # default: the shipped limit
    with pytest.raises(ValueError, match="1..256"):
        readout_codes(tok, "p#", 257, limit=256)
    with pytest.raises(ValueError, match="one of"):
        readout_codes(tok, "p#", 10, limit=300)
    assert len(labels_for_count(256, limit=256)) == 256
    with pytest.raises(ValueError):
        labels_for_count(256)
    assert (max_options(), max_options(256), MAX_READOUT_CODES) == (254, 255, 255)


def test_real_qwen_codebook_256th_code_is_ju():
    transformers = pytest.importorskip("transformers")
    bundle = ROOT / "artifacts/model-qwen4b-local.json"
    if not bundle.exists():
        pytest.skip("local Qwen bundle is absent (public checkout)")
    path = Path(json.loads(bundle.read_text())["path"])
    if not path.is_dir():
        pytest.skip("local Qwen snapshot is absent")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    codes = readout_codes(tokenizer, "</think>\n\n", 256, limit=256)
    assert codes[:255] == readout_codes(tokenizer, "</think>\n\n", 255)
    assert codes[254][0] == "JT" and codes[255][0] == "JU"


def test_option_limit_follows_the_readout():
    def payload(n):
        return {"questions": {"q": {"type": "choice", "instructions": "Pick.", "criteria": {f"o{i}": None for i in range(n)}}}}
    assert len(candidates(to_request(payload(254)).fields[0])) == 255
    with pytest.raises(ValueError, match="at most 254"):
        to_request(payload(255))
    request = to_request(payload(255), max_options=255)
    assert len(candidates(request.fields[0])) == 256 and candidates(request.fields[0])[-1][0] == UNKNOWN
    with pytest.raises(ValueError, match="at most 255"):
        to_request(payload(256), max_options=255)
    with pytest.raises(ValueError):
        to_request(payload(3), max_options=300)


# --------------------------------------------------------------------------------------- calibration

def test_top_calibration_bucket_covers_255_and_keeps_its_shipped_key():
    assert option_count_bucket(254) == "26-254"
    with pytest.raises(ValueError, match="255-code"):
        option_count_bucket(255)                            # shipped limit by default
    assert option_count_bucket(255, max_options=255) == "26-254"
    with pytest.raises(ValueError, match="256-code"):
        option_count_bucket(256, max_options=255)
    assert calibration_key("choice", 255, 255) == "choice:26-254"
    calibrator = TemperatureCalibrator.load(ROOT.parent / "imajev-release/hf/imajev-4b/calibration.json") \
        if (ROOT.parent / "imajev-release/hf/imajev-4b/calibration.json").exists() else \
        TemperatureCalibrator("t", {"choice:26-254": 2.0}, {"choice:26-254": 1})
    assert calibrator.temperature("choice", 255) == calibrator.temperature("choice", 254) is not None
    choices = [(f"o{i}", None) for i in range(255)] + [(UNKNOWN, "")]
    result = result_from_logits(choices, [float(i % 7) for i in range(256)])
    calibrated = calibrator.calibrate_result(result, "choice", 255)
    assert isinstance(calibrated, Result) and calibrated.calibration_version is not None


# -------------------------------------------------------------------------------------- torch engine

def torch_engine(codes=None):
    from torch_decision import TorchDecision
    engine = TorchDecision.__new__(TorchDecision)
    engine.processor = SimpleNamespace(tokenizer=TinyTokenizer())
    engine.device = "cpu"
    engine.readout, engine._codebook = None, None
    engine.codes, engine.prompt_layout = 255, "standard"
    torch.manual_seed(0)
    engine.model = SimpleNamespace(lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False))
    engine.render = lambda prompt, n_images: "p#"
    return engine


def write_readout(directory, rows, layout=None):
    from safetensors.torch import save_file
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(rows)
    weight = torch.randn(rows, HIDDEN)
    save_file({"weight": weight}, str(directory / "decision_readout.safetensors"))
    codes = readout_codes(TinyTokenizer(), "p#", rows, limit=rows)
    manifest = {"version": 1, "codes": [{"code": c, "token_id": i} for c, i in codes]}
    if layout:
        manifest["prompt_layout"] = layout
    (directory / "decision_readout.json").write_text(json.dumps(manifest))
    return weight


def test_torch_default_is_the_shipped_255_readout(tmp_path):
    trained = write_readout(tmp_path / "a", 255)
    engine = torch_engine()
    assert engine.enable_readout(tmp_path / "a", trainable=False) is True
    assert tuple(engine.readout.weight.shape) == (255, HIDDEN) and engine.max_options == 254
    assert torch.equal(engine.readout.weight.data, trained)
    assert engine.prompt_layout == "standard"
    with pytest.raises(ValueError, match="256 candidates exceed the 255-code readout"):
        engine.labels(256)


def test_torch_extends_a_255_readout_with_the_lm_head_row(tmp_path):
    trained = write_readout(tmp_path / "a", 255)
    engine = torch_engine()
    engine.enable_readout(tmp_path / "a", trainable=False, codes=256)
    weight = engine.readout.weight.data
    token_256 = readout_codes(TinyTokenizer(), "p#", 256, limit=256)[255][1]
    assert tuple(weight.shape) == (256, HIDDEN) and engine.max_options == 255
    assert torch.equal(weight[:255], trained)
    assert torch.equal(weight[255], engine.model.lm_head.weight[token_256].float())
    assert len(engine.labels(256)) == 256


def test_torch_fresh_256_readout_is_all_lm_head_rows():
    engine = torch_engine()
    engine.enable_readout(None, codes=256)
    ids = [token for _, token in engine._codebook]
    assert torch.equal(engine.readout.weight.data, engine.model.lm_head.weight[torch.tensor(ids)].float())
    assert engine.readout.weight.requires_grad


def test_torch_readout_logits_are_bit_identical_for_254_or_fewer_options(tmp_path):
    write_readout(tmp_path / "a", 255)
    shipped, extended = torch_engine(), torch_engine()
    shipped.enable_readout(tmp_path / "a", trainable=False)
    extended.enable_readout(tmp_path / "a", trainable=False, codes=256)
    torch.manual_seed(1)
    for count in (2, 3, 5, 27, 100, 255):
        labels = shipped.labels(count)
        assert labels == extended.labels(count)
        ids = [token for _, token in shipped._codebook[:count]]
        for _ in range(8):
            hidden = torch.randn(HIDDEN) * 10
            a = shipped.readout(hidden)[shipped._readout_indices(ids)]
            b = extended.readout(hidden)[extended._readout_indices(ids)]
            assert torch.equal(a, b)


def test_torch_rejects_mismatched_readouts(tmp_path):
    write_readout(tmp_path / "b", 256)
    engine = torch_engine()
    with pytest.raises(ValueError, match="256-code readout"):
        engine.enable_readout(tmp_path / "b", trainable=False, codes=255)
    engine = torch_engine()
    engine.enable_readout(tmp_path / "b", trainable=False)        # None follows the adapter
    assert engine.codes == 256
    write_readout(tmp_path / "c", 255)
    manifest = json.loads((tmp_path / "c" / "decision_readout.json").read_text())
    manifest["codes"][3]["token_id"] += 1
    (tmp_path / "c" / "decision_readout.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="binding"):
        torch_engine().enable_readout(tmp_path / "c", codes=256)


def test_torch_save_readout_records_codes_and_layout(tmp_path):
    engine = torch_engine()
    engine.enable_readout(None, codes=256)
    engine.save_readout(tmp_path)
    manifest = json.loads((tmp_path / "decision_readout.json").read_text())
    assert len(manifest["codes"]) == 256 and "prompt_layout" not in manifest
    engine.prompt_layout = "compact"
    engine.save_readout(tmp_path)
    assert json.loads((tmp_path / "decision_readout.json").read_text())["prompt_layout"] == "compact"
    reloaded = torch_engine()
    reloaded.enable_readout(tmp_path, trainable=False)
    assert reloaded.codes == 256 and reloaded.prompt_layout == "compact"


def test_trainer_flags_default_to_the_shipped_recipe():
    source = (ROOT / "scripts/train_decision_lora_torch.py").read_text()
    assert "'--readout-codes',type=int,choices=(255,256),default=255" in source
    assert "'--prompt-layout',choices=('standard','compact'),default='standard'" in source
    # recorded in config.json only when changed, so existing outputs keep resuming
    assert "readout_codes=255,prompt_layout='standard'," in source      # (+ expand_lora_rank=0, dev_slices=False: phase-3 rank64 lane)
    assert "codes=args.readout_codes" in source and "layout=args.prompt_layout" in source


# ---------------------------------------------------------------------------------------- MLX backend

def mlx_backend(tmp_path, monkeypatch, rows=None, **kwargs):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    import mlx_vlm
    import mlx_vlm.prompt_utils
    import mlx_vlm.trainer.utils
    from vision_decision.backend import MLXDirect

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
            self.args = SimpleNamespace(tie_word_embeddings=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Language()
            self.config = SimpleNamespace(image_token_index=-1, video_token_index=-2)

    mx.random.seed(0)
    model = Model()
    processor = SimpleNamespace(tokenizer=TinyTokenizer())
    monkeypatch.setattr(mlx_vlm, "load", lambda path, trust_remote_code=False: (model, processor))
    monkeypatch.setattr(mlx_vlm.trainer.utils, "apply_lora_layers", lambda m, adapter: m)
    monkeypatch.setattr(mlx_vlm.prompt_utils, "apply_chat_template", lambda *a, **k: "p#")
    (tmp_path / "snap").mkdir(exist_ok=True)
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({"path": str(tmp_path / "snap")}))
    adapter = None
    if rows:
        adapter = tmp_path / f"adapter{rows}"
        weight = write_readout(adapter, rows, kwargs.pop("layout", None))
        mx.save_safetensors(str(adapter / "decision_readout.safetensors"), {"weight": mx.array(weight.numpy())})
    return MLXDirect(str(bundle), adapter=adapter, **kwargs), mx


def test_mlx_default_keeps_255_and_extension_appends_the_lm_head_row(tmp_path, monkeypatch):
    shipped, mx = mlx_backend(tmp_path, monkeypatch, rows=255)
    assert shipped.codes == 255 and shipped.max_options == 254 and shipped.readout.shape == (255, HIDDEN)
    assert shipped.prompt_layout == "standard"
    extended, _ = mlx_backend(tmp_path, monkeypatch, rows=255, readout_codes=256)
    assert extended.codes == 256 and extended.max_options == 255 and extended.readout.shape == (256, HIDDEN)
    assert mx.array_equal(extended.readout[:255], shipped.readout.astype(mx.float32)).item()
    token = extended._codebook[255][1]
    row = extended.model.language_model.lm_head.weight[token].astype(mx.float32)
    assert mx.array_equal(extended.readout[255], row).item()
    # the served candidate path: identical logits for every option count the shipped model supports
    shipped._labels("", 3, 0)
    for count in (2, 5, 27, 200, 255):
        hidden = mx.random.normal((HIDDEN,)) * 10
        a = shipped._candidate_logits(hidden, None, list(range(count)))
        b = extended._candidate_logits(hidden, None, list(range(count)))
        assert mx.array_equal(a, b).item()
    assert len(extended._labels("", 256, 0)) == 256
    with pytest.raises(ValueError, match="exceed the 255-code readout"):
        shipped._labels("", 256, 0)


def test_mlx_adapter_layout_and_row_count_are_honoured(tmp_path, monkeypatch):
    compact, _ = mlx_backend(tmp_path, monkeypatch, rows=256, layout="compact")
    assert compact.codes == 256 and compact.prompt_layout == compact.trained_prompt_layout == "compact"
    forced, _ = mlx_backend(tmp_path, monkeypatch, rows=256, layout="compact", prompt_layout="standard")
    assert forced.prompt_layout == "standard" and forced.trained_prompt_layout == "compact"
    with pytest.raises(ValueError, match="256-code readout"):
        mlx_backend(tmp_path, monkeypatch, rows=256, readout_codes=255)


# ------------------------------------------------------------------------------------- MLX conversion

def test_converter_accepts_256_rows_and_copies_the_layout(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from test_adapter_conversion import write_peft_adapter  # noqa: E402
    from convert_peft_adapter_to_mlx import convert
    source = tmp_path / "peft"
    write_peft_adapter(source)
    head = mx.arange(256 * 8).reshape(256, 8).astype(mx.float32)
    mx.save_safetensors(str(source / "decision_readout.safetensors"), {"weight": head})
    codes = [{"code": f"c{i}", "token_id": i} for i in range(256)]
    (source / "decision_readout.json").write_text(json.dumps({"version": 1, "codes": codes, "prompt_layout": "compact"}))
    summary = convert(source, tmp_path / "mlx")
    assert summary["readout"] is True
    assert mx.load(str(tmp_path / "mlx" / "decision_readout.safetensors"))["weight"].shape == (256, 8)
    assert json.loads((tmp_path / "mlx" / "decision_readout.json").read_text())["prompt_layout"] == "compact"
    (source / "decision_readout.json").write_text(json.dumps({"version": 1, "codes": codes[:255]}))
    with pytest.raises(ValueError, match="Invalid decision_readout.json"):
        convert(source, tmp_path / "bad")
    (source / "decision_readout.json").write_text(json.dumps({"version": 1, "codes": codes, "prompt_layout": "tiny"}))
    with pytest.raises(ValueError, match="Invalid decision_readout.json"):
        convert(source, tmp_path / "bad2")
    mx.save_safetensors(str(source / "decision_readout.safetensors"), {"weight": mx.zeros((257, 8))})
    with pytest.raises(ValueError, match="255 or 256"):
        convert(source, tmp_path / "bad3")
