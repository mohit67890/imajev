#!/usr/bin/env python3
"""Build the deterministic, synthetic imajev-bench infrastructure pilot.

This creates review candidates, not approved benchmark data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SEED = 20260922
VERSION = "pilot-v0.1"
SIZE = (720, 480)
# This generator intentionally reuses semantic templates. Until independently
# authored templates exist, assigning any synthetic group to calibration/test
# would overstate the strength of the split.
SPLITS = ("dev",) * 100
COLORS = ((39, 93, 173), (216, 82, 62), (40, 145, 104), (226, 158, 48))


def request(request_id: str, state: dict, field: dict) -> dict:
    return {
        "schema_version": "1.0",
        "request_id": request_id,
        "state": state,
        "fields": [field],
        "execution": {"mode": "inspect", "allow_external_fallback": False},
    }


def boolean(question: str) -> dict:
    return {"id": "decision", "type": "boolean", "question": question}


def choice(question: str, values: list[tuple[str, str]]) -> dict:
    return {
        "id": "decision", "type": "choice", "question": question,
        "options": [{"value": value, "description": desc} for value, desc in values],
    }


def ordinal(question: str, maximum: int = 6) -> dict:
    return {
        "id": "decision", "type": "ordinal", "question": question,
        "levels": [{"value": i, "description": str(i)} for i in range(1, maximum + 1)],
    }


def canvas(title: str, split: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", SIZE, (244, 243, 239))
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((22, 18, 698, 462), 18, fill="white", outline=(55, 61, 71), width=3)
    draw.text((48, 38), title, font=ImageFont.load_default(size=24), fill=(28, 32, 39))
    return im, draw


def save_image(im: Image.Image, path: Path, root: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, format="PNG", optimize=False, compress_level=9)
    data = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(data).hexdigest()}


def provenance(group_index: int, variant: str, family: str, split: str, **extra: object) -> dict:
    return {
        "generator": "scripts/imajev_bench/build_pilot.py", "generator_version": "1",
        "seed": SEED, "group_index": group_index, "variant": variant,
        "template_id": f"{family}-{split}-layout-{group_index % 4}",
        "scene_id": f"synthetic-{family}-{group_index:03d}", "synthetic": True,
        "label_basis": "deterministic generator assertion; provisional pending human review", **extra,
    }


def boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Return whether two axis-aligned boxes have positive-area overlap."""
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def record(rid: str, gid: str, track: str, family: str, split: str, images: list[dict], req: dict,
           gold: bool | int | str | None, prov: dict) -> dict:
    return {"id": rid, "group_id": gid, "track": track, "family": family, "split": split,
            "images": images, "request": req, "gold": gold, "annotation_status": "draft",
            "provenance": prov}


def text_groups() -> list[dict]:
    rows = []
    for i, split in enumerate(SPLITS):
        gid, rid = f"txt-{i:03d}", f"txt-{i:03d}-a"
        family = ("threshold_routing", "evidence_policy", "priority_ordinal", "document_rule")[i % 4]
        mode = i % 10
        if family == "threshold_routing":
            observed, threshold = 3 + (i * 7) % 18, 5 + (i * 11) % 16
            state = {"case": {"verified_units": observed}, "rule": f"Approve iff verified_units is at least {threshold}. The value is complete."}
            field, gold = boolean("Approve this case?"), observed >= threshold
        elif family == "evidence_policy":
            status = ("verified", "self_reported", "missing", "conflicting", "verified")[mode % 5]
            state = {"evidence_status": status, "rule": "Return eligible only for verified evidence; ineligible for self-reported evidence; missing or conflicting evidence is insufficient."}
            field = choice("What is the eligibility decision?", [("eligible", "Verified evidence"), ("ineligible", "Disallowed evidence")])
            gold = {"verified": "eligible", "self_reported": "ineligible", "missing": None, "conflicting": None}[status]
        elif family == "priority_ordinal":
            family_index = i // 4
            impact = 1 + (family_index * 2) % 3
            urgency = 1 + (family_index * 5) % 3
            complete = mode not in (2, 7)
            state = {"impact": impact if complete else None, "urgency": urgency,
                     "rule": "Priority is impact + urgency - 1, capped at 5. Both inputs are required."}
            field, gold = ordinal("Assign priority from 1 through 5.", 5), min(5, impact + urgency - 1) if complete else None
        else:
            total, cap = 35 + (i * 17) % 170, 50 + (i * 13) % 150
            receipt = mode not in (3, 8)
            state = {"claimed_total": total, "receipt_verified": receipt,
                     "rule": f"Reimburse only when a receipt is verified and total is no more than {cap}. If receipt verification is absent, evidence is insufficient."}
            field = choice("Choose the reimbursement outcome.", [("reimburse", "Pay claim"), ("deny", "Do not pay claim")])
            gold = ("reimburse" if total <= cap else "deny") if receipt else None
        req = request(rid, state, field)
        rows.append(record(rid, gid, "text", family, split, [], req, gold, provenance(i, "base", family, split)))
    return rows


def visual_groups(root: Path) -> list[dict]:
    rows = []
    for i, split in enumerate(SPLITS):
        gid, rid = f"vis-{i:03d}", f"vis-{i:03d}-a"
        family = ("object_count", "spatial_relation", "receipt_reading", "ui_state")[i % 4]
        family_index = i // 4
        im, draw = canvas({"object_count": "Inventory tray", "spatial_relation": "Shelf diagram", "receipt_reading": "Purchase receipt", "ui_state": "Control panel"}[family], split)
        unknown = i % 10 in (0, 7)
        if family == "object_count":
            count = 1 + (family_index * 5) % 6
            for k in range(count):
                x, y = 100 + (k % 3) * 180, 130 + (k // 3) * 150
                draw.rounded_rectangle((x, y, x + 100, y + 74), 12, fill=COLORS[k % 4], outline=(20, 25, 30), width=3)
            if unknown: draw.rectangle((75, 105, 645, 410), fill=(120, 125, 132)); draw.text((260, 245), "TRAY COVERED", fill="white", font=ImageFont.load_default(size=22))
            field, gold = ordinal("How many packages are in the tray? The complete tray view is required.", 6), None if unknown else count
        elif family == "spatial_relation":
            red_left = family_index % 2 == 0
            ax, bx = (150, 470) if red_left else (470, 150)
            draw.ellipse((ax, 175, ax + 110, 285), fill=(211, 55, 53), outline=(30, 30, 30), width=4)
            draw.rectangle((bx, 175, bx + 110, 285), fill=(39, 93, 173), outline=(30, 30, 30), width=4)
            if unknown: draw.rectangle((90, 140, 640, 325), fill=(135, 139, 145)); draw.text((260, 220), "VIEW BLOCKED", fill="white", font=ImageFont.load_default(size=22))
            field, gold = boolean("Is the red circle left of the blue square?"), None if unknown else red_left
        elif family == "receipt_reading":
            amounts = [7 + (family_index * 3) % 23, 11 + (family_index * 5) % 31, 5 + (family_index * 7) % 17]
            total = sum(amounts)
            draw.text((100, 110), "NORTH MARKET", font=ImageFont.load_default(size=24), fill=(20, 20, 20))
            for k, amount in enumerate(amounts): draw.text((115, 175 + k * 52), f"ITEM {chr(65+k)}", font=ImageFont.load_default(size=20), fill=(25, 25, 25)); draw.text((490, 175 + k * 52), f"${amount}.00", font=ImageFont.load_default(size=20), fill=(25, 25, 25))
            draw.line((100, 340, 610, 340), fill=(30, 30, 30), width=2); draw.text((385, 365), f"TOTAL  ${total}.00", font=ImageFont.load_default(size=24), fill=(15, 15, 15))
            if unknown: draw.rectangle((85, 155, 630, 425), fill=(92, 96, 102)); draw.text((285, 270), "INK SMEARED", fill="white", font=ImageFont.load_default(size=22))
            field = choice("Is the printed TOTAL field below $60? Do not reconstruct an unreadable total from other fields.", [("below", "Total is below $60"), ("not_below", "Total is $60 or more")]); gold = None if unknown else ("below" if total < 60 else "not_below")
        else:
            enabled = family_index % 2 == 1
            draw.rounded_rectangle((120, 140, 600, 360), 16, fill=(235, 238, 243), outline=(60, 67, 78), width=3)
            draw.text((165, 190), "Automatic backup", font=ImageFont.load_default(size=25), fill=(25, 28, 33))
            draw.rounded_rectangle((440, 185, 555, 245), 30, fill=(45, 156, 94) if enabled else (145, 150, 158))
            knob = 505 if enabled else 445; draw.ellipse((knob, 190, knob + 50, 240), fill="white")
            draw.text((165, 285), "Device registered", font=ImageFont.load_default(size=20), fill=(65, 70, 78))
            if unknown: draw.rectangle((425, 165, 575, 260), fill=(115, 119, 125)); draw.text((438, 200), "OBSCURED", fill="white", font=ImageFont.load_default(size=18))
            field, gold = boolean("Is automatic backup visibly enabled?"), None if unknown else enabled
        img = save_image(im, root / "assets" / f"asset-{i:03d}.png", root)
        req = request(rid, {"image_roles": ["evidence"], "instruction": "Use only visible content; return insufficient evidence if the requested region cannot be read."}, field)
        rows.append(record(rid, gid, "visual", family, split, [img], req, gold, provenance(i, "base", family, split, occluded=unknown)))
    return rows


def joint_image(root: Path, i: int, split: str, family: str, condition: str) -> dict:
    im, draw = canvas(family.replace("_", " ").title(), split)
    if family == "count_threshold":
        count = {"low": 2, "mid": 5, "high": 7}.get(condition, 0)
        for k in range(count):
            x, y = 90 + (k % 4) * 145, 130 + (k // 4) * 145
            draw.rounded_rectangle((x, y, x + 82, y + 65), 10, fill=COLORS[(k + i) % 4], outline=(25, 28, 32), width=3)
    elif family == "receipt_cap":
        total = {"low": 35, "mid": 65, "high": 95}.get(condition, 0)
        draw.text((120, 125), "SERVICE RECEIPT", font=ImageFont.load_default(size=26), fill=(25, 25, 25))
        draw.text((120, 210), f"REFERENCE  {3000+i}", font=ImageFont.load_default(size=20), fill=(35, 35, 35))
        draw.line((110, 285, 610, 285), fill=(30, 30, 30), width=2)
        draw.text((310, 320), f"TOTAL  ${total}.00", font=ImageFont.load_default(size=27), fill=(20, 20, 20))
    elif family == "spatial_policy":
        positions = {"low": (150, 175, 470, 175), "mid": (470, 175, 150, 175),
                     "high": (300, 135, 300, 295)}
        rx, ry, bx, by = positions.get(condition, positions["low"])
        red_box, blue_box = (rx, ry, rx + 100, ry + 100), (bx, by, bx + 100, by + 100)
        assert not boxes_overlap(red_box, blue_box), (family, condition, red_box, blue_box)
        draw.ellipse(red_box, fill=(211, 55, 53), outline=(25, 25, 25), width=4)
        draw.rectangle(blue_box, fill=(39, 93, 173), outline=(25, 25, 25), width=4)
    else:
        draw.rounded_rectangle((130, 115, 590, 390), 22, fill=(218, 221, 224), outline=(38, 44, 50), width=5)
        draw.rectangle((180, 155, 540, 345), fill=(239, 239, 235), outline=(85, 90, 95), width=3)
        draw.text((265, 215), f"LOT {1000+i}", font=ImageFont.load_default(size=25), fill=(40, 42, 46))
        if condition == "mid":
            draw.line((225, 300, 330, 270), fill=(78, 62, 45), width=8); draw.line((245, 315, 350, 282), fill=(96, 74, 52), width=5)
        elif condition == "high":
            draw.line((445, 155, 410, 215, 468, 255, 425, 330), fill=(42, 42, 42), width=7)
    if condition == "unknown":
        draw.rectangle((75, 105, 645, 410), fill=(98, 104, 112)); draw.text((255, 240), "REQUIRED VIEW COVERED", fill="white", font=ImageFont.load_default(size=20))
    return save_image(im, root / "assets" / f"joint-{i:03d}-{family}-{condition}.png", root)


def joint_groups(root: Path) -> list[dict]:
    rows = []
    conditions = ("low", "mid", "high", "unknown")
    for i, split in enumerate(SPLITS):
        gid = f"jnt-{i:03d}"
        family = ("count_threshold", "receipt_cap", "spatial_policy", "policy_conditioned_inspection")[i % 4]
        images = {condition: joint_image(root, i, split, family, condition) for condition in conditions}
        if family == "count_threshold":
            rules = {"lenient": "Accept when the complete tray contains at least 4 packages.", "strict": "Accept when the complete tray contains at least 6 packages."}
            values = {"low": 2, "mid": 5, "high": 7}
        elif family == "receipt_cap":
            rules = {"lenient": "Accept when the readable printed TOTAL is no more than $80.", "strict": "Accept when the readable printed TOTAL is no more than $50."}
            values = {"low": 35, "mid": 65, "high": 95}
        elif family == "spatial_policy":
            rules = {"lenient": "Accept only when the red circle is left of the blue square.", "strict": "Accept only when the red circle is right of the blue square."}
            values = {"low": "left", "mid": "right", "high": "same horizontal position"}
        else:
            rules = {"lenient": "Accept if there is no visible structural crack. Cosmetic scuffs are allowed.", "strict": "Accept only if there is neither a visible structural crack nor a cosmetic scuff."}
            values = {"low": "intact", "mid": "scuff", "high": "crack"}
        for condition in conditions:
            for rule_name, rule_text in rules.items():
                variant = f"{condition}-{rule_name}"; rid = f"{gid}-{variant}"
                if condition == "unknown": gold = None
                elif family == "count_threshold": gold = "accept" if values[condition] >= (4 if rule_name == "lenient" else 6) else "reject"
                elif family == "receipt_cap": gold = "accept" if values[condition] <= (80 if rule_name == "lenient" else 50) else "reject"
                elif family == "spatial_policy": gold = "accept" if values[condition] == ("left" if rule_name == "lenient" else "right") else "reject"
                elif condition == "high" or (condition == "mid" and rule_name == "strict"): gold = "reject"
                else: gold = "accept"
                state = {"image_roles": ["required decision evidence"], "rule": rule_text,
                         "scope": "This is the only acceptance criterion. The complete specified view must be visible."}
                field = choice("Choose the decision from the image and rule.", [("accept", "Meets the rule"), ("reject", "Does not meet the rule")])
                req = request(rid, state, field)
                rows.append(record(rid, gid, "joint", family, split, [images[condition]], req, gold,
                                   provenance(i, variant, family, split, condition=condition, rule_variant=rule_name)))
    return rows


def validate(rows: list[dict]) -> dict:
    from vision_decision.contracts import Request
    assert len({r["id"] for r in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        Request.model_validate(row["request"])
        assert row["annotation_status"] == "draft"
        assert len(row["request"]["fields"]) == 1
        groups[row["group_id"]].append(row)
        field = row["request"]["fields"][0]
        gold = row["gold"]
        if gold is not None:
            if field["type"] == "boolean": assert type(gold) is bool
            elif field["type"] == "choice": assert gold in {o["value"] for o in field["options"]}
            else: assert type(gold) is int and gold in {x["value"] for x in field["levels"]}
    assert len(groups) == 300
    for track in ("text", "visual", "joint"):
        track_groups = {g: rs for g, rs in groups.items() if rs[0]["track"] == track}
        assert len(track_groups) == 100
        assert Counter(rs[0]["split"] for rs in track_groups.values()) == {"dev": 100}
    joint = [rs for rs in groups.values() if rs[0]["track"] == "joint"]
    assert all(len(rs) == 8 for rs in joint)
    assert all(len({r["split"] for r in rs}) == 1 for rs in joint)
    answer_counts = {track: Counter("unknown" if r["gold"] is None else str(r["gold"]).lower() for r in rows if r["track"] == track) for track in ("text", "visual", "joint")}
    assert all(counts["unknown"] > 0 for counts in answer_counts.values())
    # Each family must exercise multiple outcomes; an unknown-only or constant family
    # is not a useful review candidate even when the aggregate track looks balanced.
    for family in {r["family"] for r in rows}:
        outcomes = {"unknown" if r["gold"] is None else str(r["gold"]).lower() for r in rows if r["family"] == family}
        assert len(outcomes) >= 2, (family, outcomes)
    return {"records": len(rows), "groups": len(groups), "groups_by_track_split": {
        track: dict(Counter(rs[0]["split"] for rs in groups.values() if rs[0]["track"] == track))
        for track in ("text", "visual", "joint")}, "gold_distribution": {k: dict(v) for k, v in answer_counts.items()}}


def build(output: Path) -> None:
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output; choose a new --output path: {output}")
    output.mkdir(parents=True)
    random.seed(SEED)
    rows = text_groups() + visual_groups(output) + joint_groups(output)
    summary = validate(rows)
    records_path = output / "records.jsonl"
    records_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    asset_hashes = sorted((p.relative_to(output).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in (output / "assets").glob("*.png"))
    manifest = {
        "dataset_version": VERSION, "status": "draft", "purpose": "synthetic infrastructure pilot and human review candidates",
        "claims_excluded": ["human-reviewed labels", "hidden independent benchmark", "production-ready ranking set"],
        "split_warning": "All records are development-only because generator family and semantic template classes overlap. No calibration or test holdout is claimed.",
        "generator": "scripts/imajev_bench/build_pilot.py", "generator_seed": SEED,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "determinism": "Pillow raster output and JSON serialization are deterministic for the pinned environment; hashes are recorded.",
        "records_file": "records.jsonl", "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "asset_count": len(asset_hashes), "assets": [{"path": p, "sha256": h} for p, h in asset_hashes], **summary,
    }
    (output / "generator_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/imajev-bench/pilot-v0.1"))
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__": main()
