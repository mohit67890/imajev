# imajev-2b

The full model card lives on Hugging Face: https://huggingface.co/mohit67890/imajev-2b

Summary: Qwen3.5-2B + LoRA (r16/α32, language layers) + 255-code decision readout; the latency tier (p50 83 ms raw, 238 ms as shipped with four option orders, per decision on
one H100 under load). ImajevBench v2.0-lite test 71.7%, JevBench public hard 60.4% (our runs, as shipped). Trained on image, text, image+state and
two-image decisions (new sources labelled by the 9B), then hard typed questions from open-weight teachers and a soft-target stage; the
shipped adapter is the weight-space average of the last two stages' adapters. Abstains
too rarely on ImajevBench's Unknown items; start with the 4B unless the 2B's footprint is the point.
Results: `results/imajev-1.0/`, `results/benchmarks/`.
