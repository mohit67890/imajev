## v1.1 test set, identical 35,528 decisions

| | Base 2B | 2B v1.1 | 9B v1.1 | **2B v2 raw** | 2B v2 calibrated |
|---|---:|---:|---:|---:|---:|
| Accuracy (abstention credited) | 51.5% | 87.3% | 89.0% | **86.1%** | – |
| Image decisions (19,232) | 61.2% | 86.1% | 87.1% | **84.7%** | – |
| Text decisions (16,296) | 40.0% | 88.7% | 91.2% | **87.8%** | – |
| MMLU in test (never trained) | 33.6% | 53.5% | 73.7% | **43.0%** | – |
| False abstention on answerable | 20.9% | 1.6% | 1.4% | **3.6%** | – |
| ECE (10 bins) | 0.155 | 0.016 | 0.007 | **0.151** | 0.033 |
| Brier | 0.243 | 0.079 | 0.068 | **0.130** | 0.099 |
| Auto-accept at ≥0.95 | – | 61.2% @ 98.6% | 64.2% @ 98.8% | **13.8% @ 98.6%** | 58.0% @ 95.0% |

Paired v1.1 → v2: 682 fixed, 1,094 broken.

Per source (v1.1 → v2, Δ):

| Source | n | 2B v1.1 | 2B v2 | Δ |
|---|---:|---:|---:|---:|
| mmlu | 1000 | 53.5% | 43.0% | -10.5% |
| koniq | 300 | 74.7% | 69.0% | -5.7% |
| esci | 600 | 65.3% | 60.8% | -4.5% |
| vizwiz_quality | 629 | 86.0% | 83.1% | -2.9% |
| vsr | 814 | 86.6% | 83.8% | -2.8% |
| masked_evidence | 870 | 82.2% | 79.4% | -2.8% |
| unkvqa | 584 | 42.6% | 40.6% | -2.1% |
| marqo_gs | 299 | 94.6% | 92.6% | -2.0% |
| vizwiz | 954 | 93.2% | 91.6% | -1.6% |
| fashion200k | 1500 | 89.7% | 88.1% | -1.5% |
| abo | 1874 | 96.4% | 94.9% | -1.5% |
| state_aware | 1200 | 84.9% | 83.5% | -1.4% |
| tdiuc | 1412 | 93.9% | 92.6% | -1.3% |
| sqid_esci | 300 | 64.0% | 62.7% | -1.3% |
| dude | 993 | 92.3% | 91.3% | -1.0% |
| tallyqa | 600 | 68.0% | 67.0% | -1.0% |
| aokvqa | 1142 | 80.4% | 79.4% | -1.0% |
| snli | 2000 | 90.0% | 89.0% | -0.9% |
| bool_abstain | 600 | 97.5% | 96.7% | -0.8% |
| defects | 862 | 86.8% | 86.0% | -0.8% |
| vg_attributes | 1200 | 78.6% | 77.8% | -0.8% |
| boolq | 2000 | 88.0% | 87.3% | -0.7% |
| banking77 | 1000 | 91.7% | 91.3% | -0.4% |
| civil_comments | 2000 | 96.5% | 96.0% | -0.4% |
| clinc150 | 1000 | 91.2% | 90.9% | -0.3% |
| squad2 | 2000 | 99.7% | 99.4% | -0.3% |
| gqa | 1500 | 82.1% | 81.9% | -0.3% |
| nvidia_aegis | 1196 | 87.8% | 87.5% | -0.3% |
| fever | 1000 | 97.2% | 97.1% | -0.1% |
| vqav2 | 1446 | 94.8% | 95.0% | +0.1% |
| textvqa | 653 | 97.2% | 97.4% | +0.2% |
| massive | 1000 | 86.1% | 87.2% | +1.1% |
| goemotions | 1000 | 83.9% | 86.1% | +2.2% |

## New-source test rows, teacher-labelled (41,533 decisions; accuracy is agreement with the 9B teacher, not with humans)

| Source | n | Base 2B | **2B v2** | teacher-unknown rows | v2 abstains on them |
|---|---:|---:|---:|---:|---:|
| commons_photos | 3482 | 77.4% | **95.6%** | 1134 | 1093 |
| openimages_v2 | 4586 | 74.6% | **94.0%** | 1075 | 1018 |
| pd12m | 14606 | 76.5% | **96.8%** | 5263 | 5159 |
| stackexchange | 7366 | 47.8% | **95.8%** | 1962 | 1869 |
| support_reviews | 5300 | 61.3% | **97.2%** | 1631 | 1594 |
| wikipedia_paragraphs | 6193 | 60.2% | **97.5%** | 1724 | 1674 |
| all | 41533 | 66.9% | **96.4%** | | |

## v1 image exam (identical 24,221 cases)

| Panel | n | Base 2B | v1 | 2B v1.1 | 9B v1.1 | **2B v2** |
|---|---:|---:|---:|---:|---:|---:|
| held-out sources | 4,989 | 39.4% | 49.8% | 50.7% | 55.9% | **54.4%** |
| trained sources | 19,232 | 57.6% | 85.8% | 86.0% | 87.1% | **84.7%** |
| · heldout_abstention | 1,500 | 66.1% | 62.5% | 65.2% | 73.3% | **72.0%** |
| · heldout_countqa | 1,167 | 26.6% | 34.8% | 35.6% | 39.9% | **32.0%** |
| · heldout_fashionpedia | 983 | 30.1% | 48.8% | 47.7% | 58.1% | **51.6%** |
| · heldout_livewild | 1,162 | 24.8% | 51.3% | 51.6% | 52.2% | **56.6%** |
| · heldout_pairs | 177 | 45.8% | 36.2% | 37.9% | 26.6% | **54.2%** |

Held-out abstention targets n=1622: v1.1 900, 9B 980, v2 1107 correct.

## Text panels (never trained on)

| Panel | n | 2B v1.1 | 9B v1.1 | **2B v2** (raw ECE / calibrated ECE) |
|---|---:|---:|---:|---:|
| typed | 2000 | 58.1% (ECE 0.032) | 66.2% (ECE 0.045) | **59.1%** (0.052 / 0.135) |
| sst5 | 2210 | 47.5% (ECE 0.026) | 51.0% (ECE 0.062) | **44.2%** (0.022 / 0.100) |
| irrelevance | 2823 | 64.3% (ECE 0.082) | 79.6% (ECE 0.017) | **55.5%** (0.072 / 0.263) |
| · mmlu_heldout | 2000 | 51.8% | – | **40.4%** |

## Reasoning dev set on the pod (full manifest, v2 only; v1.1/9B baselines on the authored 240 are in reports/v2-datasets/reasoning-dev-baselines.md)

| Family | n | 2B v2 |
|---|---:|---:|
| agent_trace_observability | 1500 | 39.9% |
| customer_service | 1500 | 63.8% |
| invoice_processing | 1500 | 55.5% |
| security_incidents | 1500 | 63.1% |
| long_policy | 30 | 30.0% |
| multi_hop | 30 | 63.3% |
| temporal_numeric | 30 | 13.3% |
| ambiguous | 30 | 10.0% |
| tradeoff | 30 | 36.7% |
| trap | 30 | 76.7% |
| probability | 30 | 26.7% |
| routing | 30 | 80.0% |
| typed-decisions dev-selection | 6000 | 55.6% |
| authored (240) | 240 | 42.1% |
