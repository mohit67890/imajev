# Torch Prefix-KV Parity: Findings (2026-10-06, 4B base bf16, transformers 5.18.0)

Gepusht: `f9face4` auf `fork/torch-batch`. Prefix bleibt per Parity-Gate
sicher deaktiviert bis delta <= 0.02 + argmax-identisch. Kein Speedup quoten.

## Bewiesen (GPU-Bisect, 4B-Base, text-only)

1. **0D-Graph-Crash (GEFIXT, f9face4):** `graphs.run()` liefert 1D `[H]`,
   Batch-Reader `hidden[i]` → 0D → explodiert in `F.linear`.
   Fix: `.unsqueeze(0)` am B==1-Graph-Branch in `torch_decision.py`.
   MLX hat keine CUDA-Graphs → torch-only.
2. **Probe-Paar (GEFIXT):** `compiled[:2]` blind mischt Typen (noul+choice
   teilen keinen Prefix). Jetzt same-type via `_select_probe_pair`.
3. **Unsuitable-vs-Parity (GEFIXT):** `PrefixUnsuitable` → fallback OHNE
   disable; nur echte Mismatches (`ParityError`) disablen.
4. **Positionen OK:** `_suffix_positions` == seriell für text-only (Stub+GPU).
5. **Maske OK:** Suffix mit/ohne attention_mask identisch (0.093).
6. **Cache-Mutation GEFUNDEN:** Suffix-Call mutiert übergebenen Cache TROTZ
   `use_cache=False` (shared-copy-reuse delta 0.63 vs frische Kopie 0.07).
   → Frisches `_repeat_cache` pro Branch ist PFLICHT, kein Opt.
7. **Repeat(1) OK:** `deepcopy` exakt, `repeat_interleave(1)` No-Op.
8. **Padding-Hypothese (OFFEN, stärkste Spur):** Suffix-Längen 19/45/100 →
   delta 14.8/13.3/0.016. Je mehr durch Prefix läuft, desto größer der
   Fehler. Verdacht: Prefix-Prefill läuft OHNE attention_mask, seriell MIT
   (left-pad) → divergiert sobald Prompts ungleich lang. Runde 1–4 mit
   gleichen Längen: 0.067 (Residuum ungeklärt, evtl. bf16).

## Nächster Schritt

- `_prepare_rendered_example` Masken zurückgeben lassen; Prefix-Prefill MIT
  Maske; Gate-Prompts exakt nachstellen (rotiert, ungleiche Längen).
- Residuum 0.067 bei gleichen Längen danach neu messen (bf16 vs fp32?).
- Erst bei `validated_text=true` + argmax-identisch: Speedup messen.

## Repro-Skripte (`/tmp`, nicht im Repo)

- `gpu_bisect{,2,3,4,5}.py`, `parity_bisect.py`, `livegate_proof.py`
- Gate-Overlays: `/tmp/livegate_*.json`

## Zahlen (nüchtern)

- Result-Cache (identischer Request): cold 1345ms → warm 0.7ms (~1900×).
- Prefix-KV: aktuell 0× (deaktiviert). Unquotable bis Parity grün.
