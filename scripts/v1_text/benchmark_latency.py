"""Measure resident MLX request latency; model loading and warmup are excluded.

The text gate is evaluated conservatively against p95 over a three-question request.
An image baseline must be supplied to report the image-regression gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter


def percentile(values, quantile):
    if not values or not 0 <= quantile <= 1:
        raise ValueError('Expected samples and quantile in [0, 1]')
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def benchmark(engine, request, images, *, repetitions=20, warmup=2, rotations=1):
    if repetitions < 1 or warmup < 0:
        raise ValueError('repetitions must be positive; warmup must be nonnegative')
    elapsed = []
    for iteration in range(warmup + repetitions):
        start = perf_counter()
        results, metadata = engine.score_request(images, request.fields, request.state, rotations)
        if len(results) != len(request.fields):
            raise ValueError('Backend returned an incomplete request')
        duration = (perf_counter() - start) * 1000
        if iteration >= warmup:
            elapsed.append(duration)
    return {'request_id': request.request_id, 'questions': len(request.fields),
            'images': len(images), 'repetitions': repetitions, 'warmup': warmup,
            'rotations': rotations, 'p50_ms': percentile(elapsed, .5),
            'p95_ms': percentile(elapsed, .95), 'samples_ms': elapsed,
            'model_load_seconds': engine.load_seconds,
            'text_latency_gate': percentile(elapsed, .95) <= 120 if not images and len(request.fields) == 3 else None}


def main():
    from vision_decision.backend import MLXDirect
    from vision_decision.images import load_image
    from vision_decision.jev_api import to_request
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True, help='Jev-format state/questions JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-bundle', default='artifacts/model.json')
    parser.add_argument('--adapter')
    parser.add_argument('--image', action='append', default=[])
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--rotations', type=int, default=1)
    args = parser.parse_args()
    if len(args.image) > 2:
        parser.error('At most two images are supported')
    request = to_request(json.loads(args.request.read_text()))
    images = [load_image(path)[0] for path in args.image]
    engine = MLXDirect(args.model_bundle, adapter=args.adapter)
    report = benchmark(engine, request, images, repetitions=args.repetitions,
                       warmup=args.warmup, rotations=args.rotations)
    report.update(adapter=args.adapter, model_bundle=engine.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('samples_ms', 'model_bundle')}))


if __name__ == '__main__':
    main()
