"""Build paired text-only/irrelevant-image test requests without training-image leakage."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .common import read_rows, write_jsonl


def paired_controls(text_rows, image_rows, seed='v1.1-irrelevance'):
    if not text_rows or not image_rows:
        raise ValueError('Both held-out text and image panels are required')
    if any(row.get('partition') != 'test' for row in [*text_rows, *image_rows]):
        raise ValueError('Irrelevance evaluation accepts test records only')
    if any(row.get('images') for row in text_rows):
        raise ValueError('The text panel must contain zero-image requests')
    donors = sorted((row for row in image_rows if row.get('images')), key=lambda row: row['id'])
    if not donors:
        raise ValueError('No held-out donor images found')
    output = []
    used_donors = {}
    for original in text_rows:
        donor_index = int(hashlib.sha256(f'{seed}:{original["id"]}'.encode()).hexdigest(), 16) % len(donors)
        donor = donors[donor_index]
        pair_id = original['id']
        for variant in ('text_only', 'irrelevant_image'):
            row = copy.deepcopy(original)
            row['id'] = f'{pair_id}:{variant}'
            row['pair_id'] = pair_id
            row['control_variant'] = variant
            if variant == 'irrelevant_image':
                row['images'] = copy.deepcopy(donor['images'])
                row['image_role'] = 'irrelevant'
                row['irrelevant_image_source_id'] = donor['id']
                used_donors[donor['id']] = donor
            else:
                row['images'] = []
            output.append(row)
    # Retain relevant uses of every donor in the same evaluation partition.
    for donor in sorted(used_donors.values(), key=lambda row: row['id']):
        row = copy.deepcopy(donor)
        row['image_role'] = 'relevant'
        row['control_variant'] = 'relevant_donor'
        output.append(row)
    if len({row['id'] for row in output}) != len(output):
        raise ValueError('Control IDs collide with donor or text IDs')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text_test', type=Path)
    parser.add_argument('image_test', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', default='v1.1-irrelevance')
    args = parser.parse_args()
    rows = paired_controls(read_rows(args.text_test), read_rows(args.image_test), args.seed)
    write_jsonl(args.output, rows)
    print(json.dumps({'records': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
