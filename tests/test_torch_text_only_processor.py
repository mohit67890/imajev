"""Offline text-only inference uses real tiny Qwen weights and a real tokenizer."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]


def test_text_extra_omits_torchvision_and_preserves_image_installation():
    import tomllib
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert any(requirement.startswith("torchvision") for requirement in extras["torch"])
    assert extras["torch-text"] == [requirement for requirement in extras["torch"]
                                   if not requirement.startswith("torchvision")]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    if not all(hasattr(transformers, name) for name in ("Qwen3_5Config", "Qwen3_5ForConditionalGeneration")):
        pytest.skip("installed transformers does not include Qwen3.5")
    from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration

    directory = tmp_path_factory.mktemp("tiny-qwen-text")
    vocab = {f"token{i}": i for i in range(32)}
    for token, index in {"[UNK]": 0, "<think>": 1, "</think>": 2, "A": 3, "B": 4, "hello": 5,
                         "[EOS]": 6, "<|vision_start|>": 27, "<|vision_end|>": 28,
                         "<|image_pad|>": 29, "<|video_pad|>": 30}.items():
        del vocab[f"token{index}"]
        vocab[token] = index
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    template = ("{% for message in messages %}{% for item in message['content'] %}"
                "{% if item['type']=='image' %}<|vision_start|><|image_pad|><|vision_end|>"
                "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endfor %}"
                "{% if enable_thinking %}<think>\n{% else %}<think>\n\n</think>\n\n{% endif %}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[UNK]",
                                       eos_token="[EOS]", chat_template=template,
                                       additional_special_tokens=["<think>", "</think>", "<|vision_start|>",
                                                                  "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"])
    config = Qwen3_5Config(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                         layer_types=["full_attention"], max_position_embeddings=128, use_cache=False,
                         rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 1.0,
                                          "rope_type": "default", "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=16, patch_size=2, spatial_merge_size=1,
                           temporal_patch_size=1, num_position_embeddings=16),
        image_token_id=29, video_token_id=30, vision_start_token_id=27, vision_end_token_id=28)
    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    # Config files suffice to reproduce eager video construction, even on a machine
    # without torchvision. The optional reference processor is loaded by the parity test.
    image_config = dict(image_processor_type="Qwen2VLImageProcessor", patch_size=2,
                        temporal_patch_size=1, merge_size=1, size={"shortest_edge": 4, "longest_edge": 256})
    video_config = dict(video_processor_type="Qwen3VLVideoProcessor", patch_size=2,
                        temporal_patch_size=1, merge_size=1)
    (directory / "processor_config.json").write_text(json.dumps(dict(
        processor_class="Qwen3VLProcessor", image_processor=image_config, video_processor=video_config)))
    (directory / "preprocessor_config.json").write_text(json.dumps(image_config))
    (directory / "video_preprocessor_config.json").write_text(json.dumps(video_config))
    return directory


def test_text_only_cpu_inference_without_torchvision(checkpoint):
    script = '''
import importlib.util, pickle, sys
original = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args, **kwargs: None if name.split('.')[0] == 'torchvision' else original(name, *args, **kwargs)
sys.path[:0] = sys.argv[2:]
import torch
from torch_decision import TorchDecision
engine = TorchDecision(sys.argv[1], 'cpu', dtype=torch.float32)
engine.processor = pickle.loads(pickle.dumps(engine.processor))
assert engine.processor._multimodal is None
rendered, inputs, ids = engine.prepare([], 'hello', ['A', 'B'])
fast_rendered, fast_inputs, fast_ids = engine.prepare_fast([], 'hello', ['A', 'B'])
assert rendered == fast_rendered and ids == fast_ids
assert torch.equal(inputs['input_ids'], fast_inputs['input_ids'])
assert torch.isfinite(engine.candidate_logits(inputs, ids)).all()
examples = [engine.render_example([], 'hello', ['A', 'B']) + (0,)]
batch, batch_ids, _ = engine.collate(examples)
assert torch.isfinite(engine.candidate_logits_batch(batch, batch_ids)[0]).all()
assert engine._thinking_inputs([], 'hello')['input_ids'].shape[0] == 1
print('TEXT_ONLY_OK')
'''
    response = subprocess.run([sys.executable, "-c", script, str(checkpoint), str(ROOT / "src"), str(ROOT / "scripts")],
                              capture_output=True, text=True, timeout=60,
                              env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                   "PYTHONDONTWRITEBYTECODE": "1"})
    assert response.returncode == 0, response.stdout + response.stderr
    assert "TEXT_ONLY_OK" in response.stdout


def test_lazy_path_matches_full_processor_and_keeps_image_support(checkpoint):
    pytest.importorskip("torchvision")
    from PIL import Image
    from transformers import AutoProcessor
    from torch_decision import TorchDecision
    full = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    engine = TorchDecision(checkpoint, "cpu", dtype=torch.float32)
    engine.processor.tokenizer.padding_side = 'left'
    for images in ([], [Image.new("RGB", (4, 4), (20, 40, 80))]):
        rendered, inputs, ids = engine.prepare(images, "hello", ["A", "B"])
        messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": "hello"}]}]
        assert rendered == full.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        reference = full(text=[rendered], images=images or None, return_tensors="pt")
        for key, value in inputs.items():
            assert torch.equal(value, reference[key]), key
        if not images:
            assert torch.all(reference.get("mm_token_type_ids", torch.tensor(0)) == 0)
            with torch.inference_mode():
                assert torch.equal(engine.candidate_logits(inputs, ids), engine.candidate_logits(reference, ids))
        else:
            assert engine.processor._multimodal.tokenizer is engine.processor.tokenizer
            assert engine.processor._multimodal.tokenizer.padding_side == 'left'
