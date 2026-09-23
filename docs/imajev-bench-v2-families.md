# imajev-bench v2 families and item templates

These are the 12 families announced before collection. Each family targets at least 30 test clusters
in its track, with no more than 3 items per photo. Author items from these templates and **vary
the wording**. A template is a pattern to follow, not a sentence to copy: repeated phrasings become
shortcuts.

**Balance rules** (checked by `lint`):
- In each family, no single answer is correct on more than 55% of items.
- The correct option's position is random.
- Unknown is the answer on 15–25% of items in each family, except `answerability`, which is
  roughly 50/50.
- Wording such as "if not shown, answer unknown" appears on answerable and Unknown items at similar
  rates, or on neither.

## Visual track (state holds only a neutral instruction)

| Family | Pattern | Unknown arises when | Distractors |
| --- | --- | --- | --- |
| `text_reading` | Which price, time or name is printed for X? | X's value is covered, cropped or illegible | Other values printed in the same scene |
| `counting` | How many separate X are visible? (ordinal bins or exact choice) | X is too occluded to count, where the rule says so | Off-by-one counts and the count of a similar object |
| `attribute_state` | Is X open or closed, on or off? Which colour is X? | X's relevant part is hidden | The state of a neighbouring object |
| `spatial_relation` | Is X left of, above or inside Y? | X or Y is not in view | The mirrored relation |
| `answerability` | Is a year, unit or brand printed on X? The answer is yes (with the value) or no. | Items are balanced between visible and absent | — |

## Joint track (a written rule plus images)

| Family | Pattern | Notes |
| --- | --- | --- |
| `threshold_rule` | Approve if the visible total or count is at most N | Put N on both sides of the true value across items, not always at the boundary |
| `multi_clause_rule` | Approve only if A and B hold / if A or B holds | Mix which clause fails |
| `two_image_comparison` | Does Image 1 have more X than Image 2? | Refer to images by label; include order-swap variants with the reference updated |
| `rule_exception` | A later clause overrides an earlier one ("except when …") | The exception applies in about half the items |

## Text track (state only, no images)

| Family | Pattern | Unknown arises when |
| --- | --- | --- |
| `numerical_reconciliation` | Totals, date spans, unit conversions | A needed number is missing |
| `policy_precedence` | Priorities, overrides, ties | An undefined tie between two matching rules |
| `missing_conflicting_evidence` | Disagreeing sources, pending values | No authoritative source is designated |

## Contrast sets

Aim for 25–35% of visual and joint clusters. Record each set member in the item spec:

```json
"contrast": {"set_id": "scene-017", "role": "original"}
"contrast": {"set_id": "scene-017", "role": "variant", "relation": "change"}
"contrast": {"set_id": "scene-017", "role": "variant", "relation": "same"}
```

A set can vary in three ways:
- **Image variants:** the photographer's variant shots.
- **Rule variants:** the same image with a stricter or looser rule.
- **Order variants:** the two images swapped, with the ordinal references updated.

At least a third of variants must be `same`. The original must not always be the one that sits on
the threshold.

## Item spec example

```json
{
  "dataset": "imajev-bench-v2-pilot", "version": "0.1.0", "split_salt": "v2-pilot-2026-10",
  "split_fractions": {"dev": 0.3, "calibration": 0.15, "test": 0.55}, "max_records_per_image": 3,
  "sources": [
    {"id": "p017a", "path": "incoming/scene-017-a.jpg", "source_cluster": "scene-017",
     "provenance": {"source": "commissioned", "source_page": "internal:photo-batch-1", "creator": "Photographer 3",
                    "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/"}}
  ],
  "items": [
    {"id": "v2-0001", "track": "visual", "family": "text_reading", "images": ["p017a"],
     "state": {"instruction": "Use only the supplied photograph."},
     "field": {"id": "decision", "type": "choice", "question": "What price is printed next to the soup?",
               "options": [{"value": "6.50"}, {"value": "7.50"}, {"value": "5.00"}, {"value": "19.00"}]},
     "draft_gold": "6.50", "evidence": "Top-left list: Soup 6.50",
     "answerability": "visible", "difficulty": "easy",
     "contrast": {"set_id": "scene-017", "role": "original"}}
  ]
}
```

Build it with `python -m imajev_bench assemble --spec spec.json --output data/imajev-bench/v2-pilot`,
then run `lint`.
