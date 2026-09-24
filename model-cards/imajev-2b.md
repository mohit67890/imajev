# imajev-2b

The full model card lives on Hugging Face: https://huggingface.co/mohit67890/imajev-2b

Summary: Qwen3.5-2B + LoRA (r16/α32, language layers) + 255-code decision readout; the latency tier (p50 68 ms per decision on
one H100). ImajevBench v2.0-lite test 70.3%, JevBench public hard 56.8% (our runs). Trained on image, text, image+state and
two-image decisions (new sources labelled by the 9B), then a final stage of hard typed questions from open-weight teachers. Abstains
too rarely on ImajevBench's Unknown items; start with the 4B unless the 2B's footprint is the point.
Results: `results/imajev-1.0/`, `results/benchmarks/`.
