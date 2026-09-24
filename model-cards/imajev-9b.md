# imajev-9b

The full model card lives on Hugging Face: https://huggingface.co/mohit67890/imajev-9b

Summary: Qwen3.5-9B + LoRA (r16/α32, language layers) + 255-code decision readout; the quality tier (p50 96 ms raw, 316 ms as shipped with four option orders, per decision on one
H100 under load, ~19 GB resident) and (in an earlier version) the labeller of the 2B's and 4B's new photo-source data. ImajevBench v2.0-lite test 82.1%, JevBench public hard 69.4%
(our runs, as shipped). Trained on about half a million human-labelled image and text decisions from licence-checked sources, then hard
typed questions from open-weight teachers and a soft-target stage; the shipped adapter is the weight-space average of the last two stages' adapters. Results: `results/imajev-1.0/`, `results/benchmarks/`,
`results/imajev-9b/` (earlier version).
