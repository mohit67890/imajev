import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest

from v1_text.build_heldout_controls import paired_controls


def panels():
    text = {'id': 'text-q', 'partition': 'test', 'images': [], 'target': True,
            'request': {'state': 'Text evidence.', 'fields': [{'id': 'q'}]}}
    image = {'id': 'image-q', 'partition': 'test', 'images': [{'image': 'a.png', 'sha256': 'abc'}],
             'target': False, 'request': {'state': {}, 'fields': [{'id': 'q'}]}}
    return [text], [image]


def test_pairs_preserve_evidence_targets_and_relevant_donors():
    text, image = panels()
    original = copy.deepcopy((text, image))
    rows = paired_controls(text, image)
    assert len(rows) == 3
    assert rows[0]['request'] == rows[1]['request'] == text[0]['request']
    assert rows[0]['target'] is rows[1]['target'] is True
    assert rows[0]['images'] == [] and rows[1]['images'] == rows[2]['images']
    assert rows[1]['image_role'] == 'irrelevant' and rows[2]['image_role'] == 'relevant'
    assert rows[0]['pair_id'] == rows[1]['pair_id']
    assert rows == paired_controls(text, image)
    assert (text, image) == original


def test_controls_reject_train_donors():
    text, image = panels()
    image[0]['partition'] = 'train'
    with pytest.raises(ValueError, match='test records only'):
        paired_controls(text, image)
