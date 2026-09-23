# Contributing

- Open an issue before a large change; small fixes can go straight to a pull request.
- Run `pytest` before submitting. Tests that need model weights or private data are skipped automatically when those are absent.
- Do not commit weights, datasets, prediction files or credentials. Adapters live on Hugging Face; data stays out of git.
- Every evaluation number added to `results/` must name the model, adapter, calibration, interface (direct scoring vs generation),
  rotation count and hardware it was measured on.
- Licence contributions under Apache-2.0 (see `LICENSE`).
