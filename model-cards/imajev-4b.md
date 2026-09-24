# imajev-4b

The full model card lives on Hugging Face: https://huggingface.co/mohit67890/imajev-4b

Summary: Qwen3.5-4B + LoRA (r16/α32, language layers) + 255-code decision readout; the recommended default (p50 85 ms per
decision on one H100; runs on a Mac via MLX). ImajevBench v2.0-lite test 82.4%, statistically tied with the 9B; JevBench public
hard 67.6% (our runs). Trained on image and text decisions (new photo sources labelled by the 9B, plus photo-vs-record and
two-photo pairs), then a final stage of hard typed questions from open-weight teachers. Results: `results/imajev-1.0/`, `results/benchmarks/`.
