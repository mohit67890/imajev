# imajev-2b

The full model card lives on Hugging Face: https://huggingface.co/mohit67890/imajev-2b

Summary: Qwen3.5-2B + LoRA (r16/α32, language layers) + 255-code decision readout; the latency tier (~80 ms per request on a Mac
Studio via MLX). Trained on image, text, image+state and two-image decisions, with the 9B as teacher. Results: `results/imajev-2b/`.
