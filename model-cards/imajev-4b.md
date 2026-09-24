# imajev-4b

The full model card lives on Hugging Face: https://huggingface.co/mohit67890/imajev-4b

Summary: Qwen3.5-4B + LoRA (r16/α32, language layers) + 255-code decision readout; the recommended default (p50 96 ms raw, 350 ms as shipped with four option orders, per
decision on one H100 under load; runs on a Mac via MLX). ImajevBench v2.0-lite test 82.4%, within noise of the 9B; JevBench public
hard 70.3% (our runs, as shipped). Trained on image and text decisions (new photo sources labelled by the 9B, plus photo-vs-record and
two-photo pairs), then hard typed questions from open-weight teachers and a soft-target stage; the shipped adapter is the weight-space
average of the last two stages' adapters. Results: `results/imajev-1.0/`, `results/benchmarks/`.
