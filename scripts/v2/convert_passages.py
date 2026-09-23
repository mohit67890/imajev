"""Turn decision-v2 TEXT passages into candidate decision records for the 9B teacher.

Input   data/decision-v2-raw/<source>/passages.jsonl   (written by scripts/v2/acquire_*.py)
Output  data/decision-v2/<source>/records.jsonl        (the spec's v2 record schema)
        data/decision-v2/<source>/conversion.json      (counts, shares, what was dropped)

Every record has `target: null` and `pseudo_label: "pending"`; the labels come later from
`scripts/v2/pseudo_label.py`.  The converter is deterministic from `--seed`: rerunning it
reproduces the file byte for byte.

    PYTHONPATH=src:scripts .venv/bin/python scripts/v2/convert_passages.py stackexchange
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v1_text.common import stable_partition                      # noqa: E402
from v2.common import verified_license                           # noqa: E402
from v2.templates import passage as T                            # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "decision-v2-raw"
OUT = ROOT / "data" / "decision-v2"
LIC = "data/decision-v2/licenses"
PARTITION_SEED = "decision-v2-text"

# ------------------------------------------------------------------------------- source config

SITE_TOPIC = {
    "cooking": "cooking and recipes", "diy": "home repair and DIY", "travel": "travel and transport",
    "money": "personal finance", "law": "law and legal procedure", "gardening": "gardening and plants",
    "fitness": "fitness and exercise", "workplace": "workplace and careers",
    "parenting": "parenting and childcare", "pets": "pets and animal care", "photo": "photography",
    "security": "computer security", "outdoors": "outdoor recreation and camping",
    "bicycles": "cycling and bicycles", "music": "music", "movies": "film and television",
    "health": "health and medicine", "interpersonal": "mental health and relationships",
    "woodworking": "woodworking", "boardgames": "board games and tabletop",
    "academia": "academic life and research", "writers": "books and writing",
    "mechanics": "car maintenance and repair", "ux": "user experience design",
    "graphicdesign": "graphic design", "homebrew": "home brewing and fermentation",
    "sustainability": "environment and sustainability", "vegetarianism": "vegetarian and vegan eating",
    "martialarts": "martial arts and self defence", "lifehacks": "everyday household tips",
    "expatriates": "visas and border formalities", "sports": "sports and athletics",
}
DOMAIN_TOPIC = {
    "restaurant": "restaurant reservations", "restaurants": "restaurant reservations",
    "restaurant-search": "restaurant reservations", "food-ordering": "food delivery orders",
    "hotel": "hotel booking", "hotels": "hotel booking", "flights": "flight booking",
    "flights_1": "flight booking", "taxi": "taxi and ride booking", "ridesharing": "taxi and ride booking",
    "train": "train travel", "trains": "train travel", "buses": "train travel",
    "rentalcars": "car rental", "movies": "movie tickets", "media": "film and television",
    "music": "music streaming", "events": "event and ticket booking", "sports": "sports information",
    "travel": "tourist attractions", "attraction": "tourist attractions", "hospital": "health and medicine",
    "weather": "weather information", "calendar": "appointment scheduling", "services": "appointment scheduling",
    "banks": "banking support", "payment": "banking support", "homes": "hotel booking",
    "police": "law and legal procedure", "rentalcar": "car rental",
    "alarm": "appointment scheduling", "messaging": "appointment scheduling",
    "ridesharing_1": "taxi and ride booking", "buses_1": "train travel",
}
WIKI_TOPIC_KEYWORDS = {
    "history": ["century", "empire", "kingdom", "dynasty", "historian", "medieval", "ancient", "revolution"],
    "military history": ["regiment", "battalion", "infantry", "warship", "squadron", "troops", "wehrmacht", "artillery"],
    "politics and government": ["election", "parliament", "senator", "minister", "constituency", "governor", "legislature", "political party"],
    "sports and athletics": ["football", "soccer", "league", "championship", "olympic", "tournament", "cricket", "basketball", "athlete"],
    "music": ["album", "band", "singer", "guitarist", "record label", "soundtrack", "composer", "song"],
    "film and television": ["film", "television", "sitcom", "screenplay", "episode", "directed by", "actress", "actor"],
    "books and writing": ["novel", "poet", "poem", "novelist", "literary", "magazine", "publisher"],
    "biology and life sciences": ["species", "genus", "moth", "beetle", "endemic", "mammal", "larvae", "subspecies", "plant"],
    "physics and astronomy": ["galaxy", "asteroid", "orbit", "spacecraft", "telescope", "quantum", "particle", "star system"],
    "chemistry": ["compound", "molecule", "chemical", "acid", "polymer", "synthesis", "ion"],
    "earth science and geography": ["village", "municipality", "district", "province", "river", "mountain", "island", "population", "county"],
    "mathematics": ["theorem", "equation", "algebra", "geometry", "topology", "integer", "matrix"],
    "travel and transport": ["railway", "locomotive", "highway", "airport", "airline", "station", "motorway"],
    "architecture and construction": ["cathedral", "architect", "castle", "bridge", "tower", "constructed in", "listed building"],
    "business and commerce": ["company", "corporation", "subsidiary", "headquarters", "manufacturer", "revenue", "founded in"],
    "education and teaching": ["school", "university", "college", "campus", "students", "faculty"],
    "law and legal procedure": ["court", "statute", "supreme court", "plaintiff", "legislation", "judicial"],
    "religion and philosophy": ["church", "theology", "buddhist", "hindu", "islam", "christian", "temple", "philosopher"],
    "language and linguistics": ["dialect", "phonology", "grammar", "alphabet", "linguistic", "spoken language"],
    "health and medicine": ["disease", "patients", "symptoms", "syndrome", "clinical", "hospital", "virus", "treatment"],
    "software and computing": ["software", "algorithm", "programming", "protocol", "operating system", "database"],
    "agriculture": ["crop", "cultivar", "livestock", "harvest", "farming"],
}

SOURCES = {
    "stackexchange": {
        "kind": "qa",
        "source_split": "dump",
        "license": [("stackexchange/archive-org-item-description.txt", "CC-BY-SA-4.0")],
    },
    "wikipedia_paragraphs": {
        "kind": "article",
        "source_split": "20231101.en",
        "license": [("wikipedia_paragraphs/terms-of-use-licensing-section.txt", "CC-BY-SA-4.0")],
    },
    "support_reviews": {
        "kind": "conversation",
        "source_split": "train",
        "license": [
            ("support_reviews/schema_guided.LICENSE.txt", "CC-BY-SA-4.0"),
            ("support_reviews/multiwoz.LICENSE", "MIT"),
            ("support_reviews/taskmaster2.TM-2-2020.README.md", "CC-BY-4.0"),
        ],
        "license_by_subsource": {
            "schema_guided": "support_reviews/schema_guided.LICENSE.txt",
            "multiwoz": "support_reviews/multiwoz.LICENSE",
            "taskmaster2": "support_reviews/taskmaster2.TM-2-2020.README.md",
        },
    },
}

FAMILY_WEIGHTS = {
    "qa": {"topic": 14, "intent": 12, "stance": 8, "adequacy_of_answer": 14, "sentiment": 8,
           "severity_urgency": 10, "contains_claim": 12, "entity_type": 6,
           "next_action_routing": 8, "pii_present": 4, "numeric_comparison": 4},
    "article": {"topic": 16, "intent": 10, "stance": 6, "adequacy_of_answer": 4, "sentiment": 8,
                "severity_urgency": 6, "contains_claim": 16, "entity_type": 16,
                "next_action_routing": 8, "pii_present": 4, "numeric_comparison": 6},
    "conversation": {"topic": 10, "intent": 14, "stance": 4, "adequacy_of_answer": 12, "sentiment": 12,
                     "severity_urgency": 12, "contains_claim": 12, "entity_type": 6,
                     "next_action_routing": 10, "pii_present": 4, "numeric_comparison": 4},
}
# families whose unknown variant is honest for this kind of passage
UNKNOWN_OK = {
    "qa": T.UNKNOWN_CAPABLE,
    "article": T.UNKNOWN_CAPABLE,
    "conversation": T.UNKNOWN_CAPABLE - {"adequacy_of_answer"},
}
UNKNOWN_ATTEMPT_RATE = 0.235      # tuned so the realised share lands inside the spec's 15-25 %
MULTI_RECORD_RATE = 0.25
DECISION_COUNTS = ([2] * 55) + ([3] * 32) + ([4] * 10) + ([5] * 3)   # mean 2.61 decisions per passage

# --------------------------------------------------------------------------------- extraction

SENTENCE = re.compile(r"(?<=[.!?])\s+")
CAP_RUN = re.compile(r"\b([A-Z][a-zA-Z'’-]{2,}(?:\s+(?:of|the|de|van|and|&)?\s*[A-Z][a-zA-Z'’-]{2,}){0,2})")
STOP_ENTITY = {
    "The", "This", "That", "There", "These", "Those", "When", "What", "Where", "Which", "While",
    "However", "Although", "Because", "Since", "After", "Before", "Also", "Then", "They", "Their",
    "You", "Your", "Yes", "No", "But", "For", "And", "It", "If", "In", "On", "As", "At", "So",
    "My", "We", "Our", "His", "Her", "Its", "One", "Some", "Any", "All", "Thanks", "Hi", "Hello",
    "User", "System", "Assistant", "Edit", "Note", "Answer", "Question",
}
CURRENCY = re.compile(r"([$£€])\s?(\d[\d,]*(?:\.\d+)?)")
PERCENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?(?:%|per cent|percent)")
UNITED = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s?(kg|kilograms?|g|grams?|lbs?|pounds?|oz|ounces?|km|kilometres?|kilometers?|"
    r"miles?|metres?|meters?|cm|mm|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|litres?|liters?|ml|watts?|volts?)\b",
    re.I)
UNIT_QUANTITY = {
    "kg": "weight in kilograms", "g": "weight in grams", "lb": "weight in pounds",
    "oz": "weight in ounces", "km": "distance in kilometres", "mile": "distance in miles",
    "metre": "length in metres", "meter": "length in metres", "cm": "length in centimetres",
    "mm": "length in millimetres", "minute": "duration in minutes", "min": "duration in minutes",
    "hour": "duration in hours", "hr": "duration in hours", "day": "number of days",
    "week": "number of weeks", "month": "number of months", "year": "number of years",
    "litre": "volume in litres", "liter": "volume in litres", "ml": "volume in millilitres",
    "watt": "power in watts", "volt": "voltage in volts", "gram": "weight in grams",
    "kilogram": "weight in kilograms", "pound": "weight in pounds", "ounce": "weight in ounces",
    "kilometre": "distance in kilometres", "kilometer": "distance in kilometres",
}
CURRENCY_NAME = {"$": "USD", "£": "GBP", "€": "EUR"}


def entities(text: str, title: str = "") -> list[str]:
    found: list[str] = []
    if title and title[:1].isupper() and len(title) <= 60 and " " in title:
        found.append(title)
    for match in CAP_RUN.finditer(text):
        value = re.sub(r"\s+", " ", match.group(1)).strip()
        head = value.split()[0]
        if head in STOP_ENTITY or value.isupper() and len(value) <= 3:
            continue
        start = match.start()
        preceding = text[max(0, start - 2):start]
        if " " not in value and (start == 0 or preceding.strip().endswith((".", "!", "?", "\n"))):
            continue                                  # a sentence-initial ordinary word
        if value not in found:
            found.append(value)
        if len(found) >= 12:
            break
    return found


def numbers(text: str) -> list[dict]:
    out: list[dict] = []
    for sign, raw in CURRENCY.findall(text):
        out.append({"raw": sign + raw, "value": float(raw.replace(",", "")),
                    "unit": CURRENCY_NAME.get(sign, ""), "quantity": "price"})
    for raw in PERCENT.findall(text):
        out.append({"raw": raw + "%", "value": float(raw.replace(",", "")),
                    "unit": "per cent", "quantity": "percentage"})
    for raw, unit in UNITED.findall(text):
        stem = unit.lower().rstrip("s") if not unit.lower().endswith("ss") else unit.lower()
        quantity = UNIT_QUANTITY.get(stem)
        if not quantity:
            continue
        out.append({"raw": f"{raw} {unit}", "value": float(raw.replace(",", "")),
                    "unit": unit.lower(), "quantity": quantity})
    seen, unique = set(), []
    for n in out:
        if n["raw"] in seen or not (0 < n["value"] < 1_000_000):
            continue
        seen.add(n["raw"])
        unique.append(n)
    return unique[:8]


def clauses(text: str) -> list[str]:
    out = []
    for sentence in SENTENCE.split(text.replace("\n", " ")):
        s = re.sub(r"\s+", " ", sentence).strip()
        if not (40 <= len(s) <= 180) or s.endswith("?") or s.startswith(("User:", "System:", "Assistant:")):
            continue
        if not s[:1].isalpha():
            continue
        out.append(s.rstrip("."))
        if len(out) >= 8:
            break
    return out


def wiki_topics(lead: str) -> list[str]:
    low = lead.lower()
    scored = []
    for label, keys in WIKI_TOPIC_KEYWORDS.items():
        hits = sum(1 for k in keys if k in low)
        if hits:
            scored.append((hits, label))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < 2:
        return [scored[0][1]] if scored and scored[0][0] == 1 and len(scored) == 1 else []
    return [label for _, label in scored[:2]]


# ------------------------------------------------------------------------------------- inputs

def build_inputs(source: str, row: dict, kind: str) -> dict:
    if source == "stackexchange":
        body, answer, title = row["body"], row["answer"], row["title"]
        hints = [SITE_TOPIC[row["site"]]] if row["site"] in SITE_TOPIC else []
        tags = row.get("tags", [])
        group = row["site"]
    elif source == "wikipedia_paragraphs":
        body, answer, title = row["lead"], "", row["title"]
        hints = wiki_topics(row["lead"])
        tags = []
        group = "wikipedia"
    else:
        body, title = row["transcript"], ""
        answer = body.rsplit("\n", 1)[-1].split(": ", 1)[-1] if "\n" in body else ""
        hints = [DOMAIN_TOPIC[d.lower()] for d in row.get("domains", []) if d.lower() in DOMAIN_TOPIC]
        tags = row.get("domains", [])
        group = row["subsource"]
    return {
        "id": row["id"], "source": source, "kind": kind, "title": title, "passage": body,
        "answer": answer, "topic_hints": list(dict.fromkeys(hints)), "tags": tags,
        "entities": entities(body, title), "numbers": numbers(body),
        "clauses": clauses(body), "foreign_clauses": [], "_bucket": group,
    }


def weighted(rng, weights: dict) -> str:
    total = sum(weights.values())
    draw = rng.random() * total
    for key, weight in weights.items():
        draw -= weight
        if draw <= 0:
            return key
    return next(iter(weights))


# ------------------------------------------------------------------------------------ records

def convert(source: str, seed: str, limit: int | None, max_passages: int | None = None) -> tuple[list[dict], dict]:
    config = SOURCES[source]
    kind = config["kind"]
    licences = {path: verified_license(Path(LIC) / path, spdx) for path, spdx in config["license"]}
    by_sub = config.get("license_by_subsource")

    rows = [json.loads(line) for line in (RAW / source / "passages.jsonl").open() if line.strip()]
    if limit:
        rows = rows[:limit]
    if max_passages and len(rows) > max_passages:
        # Deterministic thinning that keeps the site / sub-source proportions of the raw file.
        order = sorted(range(len(rows)), key=lambda i: hashlib.sha256(rows[i]["id"].encode()).hexdigest())
        keep = set(order[:max_passages])
        rows = [r for i, r in enumerate(rows) if i in keep]
    inputs = [build_inputs(source, row, kind) for row in rows]

    # Claims that are NOT in a passage come from a different passage of a different bucket
    # (a different Stack Exchange site / a different support corpus), so they stay topically
    # plausible without being about this text.
    by_bucket: dict[str, list[str]] = collections.defaultdict(list)
    for item in inputs:
        by_bucket[item["_bucket"]].extend(item["clauses"][:3])
    buckets = sorted(by_bucket)
    flat = [c for b in buckets for c in by_bucket[b]]
    for n, item in enumerate(inputs):
        others = [b for b in buckets if b != item["_bucket"]] or buckets
        source_pool = by_bucket[others[n % len(others)]] or flat
        if not source_pool:
            item["foreign_clauses"] = []
            continue
        own = set(item["clauses"])
        start = (n * 7919) % len(source_pool)
        foreign = []
        for k in range(min(len(source_pool), 60)):
            clause = source_pool[(start + k) % len(source_pool)]
            if clause not in own and clause not in foreign:
                foreign.append(clause)
            if len(foreign) >= 3:
                break
        item["foreign_clauses"] = foreign

    subsource_of = {row["id"]: row.get("subsource") for row in rows}
    records: list[dict] = []
    stats = collections.Counter()
    families = collections.Counter()
    partitions = collections.Counter()
    variants = collections.Counter()
    claim_balance = collections.Counter()
    decisions = 0
    unknown_decisions = 0
    claim_toggle = 0

    for item in inputs:
        rng = random.Random(f"{seed}\0{source}\0{item['id']}")
        want = rng.choice(DECISION_COUNTS)
        weights = dict(FAMILY_WEIGHTS[kind])
        if not item["topic_hints"]:
            weights.pop("topic", None)
        if not item["clauses"]:
            # both halves of contains_claim need the passage's own clauses to stay 50/50,
            # and stance needs a claim drawn from the passage
            weights.pop("contains_claim", None)
            weights.pop("stance", None)
        if not item["foreign_clauses"]:
            weights.pop("contains_claim", None)
        if not item["entities"]:
            weights["entity_type"] = max(1, weights["entity_type"] // 4)
        if not item["numbers"]:
            weights["numeric_comparison"] = max(1, weights["numeric_comparison"] // 4)
        if kind == "conversation" and not item["answer"]:
            weights.pop("adequacy_of_answer", None)

        chosen: list[T.Question] = []
        used_ids: set[str] = set()
        for _ in range(want * 3):
            if len(chosen) >= want:
                break
            family = weighted(rng, weights)
            unknown = family in UNKNOWN_OK[kind] and rng.random() < UNKNOWN_ATTEMPT_RATE
            if family == "adequacy_of_answer" and kind == "article":
                unknown = True                 # there is no answer to rate; asking is honest-unknown
            if family == "contains_claim":
                made = T.contains_claim(item, rng, absent=bool(claim_toggle % 2))
                if made:
                    claim_toggle += 1
            else:
                made = T.make_one(family, item, rng, unknown=unknown)
            for q in made:
                if q.field["id"] in used_ids:
                    continue
                used_ids.add(q.field["id"])
                chosen.append(q)
        if not chosen:
            stats["passages_without_questions"] += 1
            continue

        partition = stable_partition(item["id"], seed=PARTITION_SEED)
        licence = licences[by_sub[subsource_of[item["id"]]]] if by_sub else licences[config["license"][0][0]]
        groups: list[list[T.Question]] = []
        if len(chosen) >= 2 and rng.random() < MULTI_RECORD_RATE:
            groups.append(chosen[:5])
        else:
            groups = [[q] for q in chosen]

        for k, group in enumerate(groups):
            withhold = any(q.notes.get("withhold_answer") for q in group)
            has_adequacy = any(q.family == "adequacy_of_answer" for q in group)
            with_answer = bool(item["answer"]) and not withhold and (has_adequacy or rng.random() < 0.4)
            if kind == "conversation":
                with_answer = False            # the transcript already ends with the agent's reply
            state, variant = T.build_state(item, rng, with_answer=with_answer)
            record = {
                "id": f"{source}:{item['id']}:{k}",
                "source": source,
                "source_group": item["id"],
                "partition": partition,
                "family": group[0].family if len(group) == 1 else "multi_question",
                "source_split": config["source_split"],
                "license": licence,
                "images": [],
                "request": {
                    "schema_version": "1.0",
                    "request_id": f"{source}-{item['id']}-{k}"[:128],
                    "state": state,
                    "fields": [q.field for q in group],
                },
                "target": None,
                "abstention_cause": None,
                "pseudo_label": "pending",
                "template_id": ";".join(q.template_id for q in group),
                "state_variant": variant,
                "field_families": {q.field["id"]: q.family for q in group},
                "unknown_by_construction": {q.field["id"]: q.unknown_by_construction for q in group},
            }
            if partition == "test":
                record["pseudo_label_test"] = True
            records.append(record)
            partitions[partition] += 1
            variants[variant] += 1
            for q in group:
                decisions += 1
                families[q.family] += 1
                unknown_decisions += int(q.unknown_by_construction)
                if q.family == "contains_claim":
                    claim_balance["absent" if q.notes.get("claim_absent") else "present"] += 1

    report = {
        "source": source,
        "passages": len(inputs),
        "records": len(records),
        "decisions": decisions,
        "decisions_per_passage": round(decisions / max(1, len(inputs)), 3),
        "unknown_by_construction": unknown_decisions,
        "unknown_share": round(unknown_decisions / max(1, decisions), 4),
        "families": dict(sorted(families.items())),
        "partitions": dict(sorted(partitions.items())),
        "state_variants": dict(sorted(variants.items())),
        "contains_claim_balance": dict(sorted(claim_balance.items())),
        "multi_question_records": sum(1 for r in records if r["family"] == "multi_question"),
        "notes": dict(stats),
    }
    return records, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=sorted(SOURCES))
    ap.add_argument("--seed", default="decision-v2")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-passages", type=int, help="deterministically thin the raw passages to this many")
    args = ap.parse_args()

    records, report = convert(args.source, args.seed, args.limit, args.max_passages)
    out = OUT / args.source
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=True, allow_nan=False) + "\n" for r in records))
    (out / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
