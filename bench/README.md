# ImajevBench

A benchmark for typed decisions over **text and photos**: text-only items, visual items and joint items where the answer needs the
state and the image together, each with an explicit Unknown option. Design: `docs/imajev-bench-v2-plan.md` and
`docs/imajev-bench-v2-families.md`; harness and scoring: `src/imajev_bench/` (`python -m imajev_bench --help`); quickstart:
`docs/imajev-bench-quickstart.md`.

Current public version: **v2.0-lite preview** (533 items; all images AI-generated and checked by two model families; labels known by
construction; human audit in progress). Results and the datasheet with its unmet release gates are in
`RESULTS-v2-lite-preview.md` and `DATASHEET-v2-lite-preview.md`. Records and images are distributed as a Hugging Face dataset
(see the project README), not in this repository.
