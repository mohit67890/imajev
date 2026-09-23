# v2 teacher labelling (9B v1.1, calibrated), 23 Sept 2026, 4x H100, ~2.3 h at ~130 decision-orders/s

Candidates: 326,384 records / 366,967 decisions from commons_photos, openimages_v2, stackexchange, wikipedia_paragraphs, support_reviews.
Keep rule: argmax agrees across the served and rotated-by-one option orders AND top probability >= 0.6 (or unknown >= 0.5).

| | records | decisions |
|---|---:|---:|
| kept | 247,304 | 271,036 (73.9%) |
| dropped: low confidence | | 60,927 |
| dropped: order disagreement | | 29,657 |
| dropped: weak unknown | | 5,347 |
| kept with target unknown | | 76,400 (28.2% of kept) |

Kept by family: multi_question 47,761 · contains_claim 22,486 · topic 20,645 · intent 18,342 · entity_type 15,730 · next_action_routing 13,783 ·
main_object 13,506 · object_present 11,949 · sentiment 11,525 · colour_of_x 11,206 · adequacy_of_answer 10,761 · material 9,513 ·
pii_present 9,166 · stance 8,616 · severity_urgency 8,283 · indoor_outdoor 7,374 · count_bucket 7,285 · text_visible 5,962 ·
two_image_same_subject 5,608 · scene_type 5,066 · two_image_which 3,674 · numeric_comparison 2,448 · photo_quality 347.
Mean teacher confidence on kept decisions 0.86. PD12M (179,997 candidates) labelled in a second pass (see phase 2).

## PD12M pass (second labelling pass, same teacher and keep rule)
179,997 candidates → **144,871 kept (80.5%)**, 35.3% of kept with target unknown (PD12M asks about absent objects for half its
object questions and contradicts the image in 30% of its states). Dropped: low confidence 18,468 · order disagreement 12,384 · weak unknown 4,274.
Kept by family: main_object 21,682 · object_present 20,893 · colour_of_x 19,985 · scene_type 16,667 · material 15,313 · indoor_outdoor 12,297 ·
count_bucket 11,420 · text_visible 9,973 · two_image_same_subject 9,401 · two_image_which 6,698 · photo_quality 542.

## Assembled manifest `decision-v2.jsonl` (sha256 5cf6f388…)
940,153 records: v1.1 547,978 rows (594,771 decisions; 418,598 training rows regularised toward the base 2B at alpha 0.5, the rest
pass through: not_listed rows and non-train partitions) + v2 392,175 rows (415,907 decisions; 39,166 held-out test rows with
`pseudo_label_test`; unknown share 30.7%). 634,233 unique groups, 337,301 unique images, audit ok. Full report: `mixture-audit-v2.json`.
