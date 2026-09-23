"""Descriptive dataset audit; declared group counts are not independence proof."""
from collections import Counter, defaultdict
from .runner import digest


def audit(records):
    images, inputs, templates, groups = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(list)
    for row in records:
        groups[row["group_id"]].append(row)
        for asset in row["images"]:
            images[asset["sha256"]].add(row["group_id"])
        request = {k: v for k, v in row["request"].items() if k != "request_id"}
        inputs[digest({"request": request, "images": [x["sha256"] for x in row["images"]]})].add(row["group_id"])
        template = row["provenance"].get("template_id")
        if template:
            templates[template].add(row["split"])
    parent = {g: g for g in groups}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    for connected in list(images.values()) + list(inputs.values()):
        if connected:
            first = next(iter(connected))
            for group in connected:
                parent[find(group)] = find(first)
    clusters = defaultdict(list)
    for group in groups:
        clusters[find(group)].append(group)
    return {
        "records": len(records), "declared_groups": len(groups),
        "exact_content_connected_clusters": len(clusters),
        "warning": "Clusters detect exact reuse only; neither group IDs nor unique pixels prove statistical independence or fresh templates.",
        "annotation_status": dict(Counter(r["annotation_status"] for r in records)),
        "track_records": dict(Counter(r["track"] for r in records)),
        "track_groups": dict(Counter(track for rows in groups.values() for track in {r["track"] for r in rows})),
        "mixed_track_groups": sum(len({r["track"] for r in rows}) > 1 for rows in groups.values()),
        "split_records": dict(Counter(r["split"] for r in records)),
        "family_records": dict(Counter(r["family"] for r in records)),
        "type_records": dict(Counter(r["request"]["fields"][0]["type"] for r in records)),
        "unknown_records": sum(r["gold"] is None for r in records),
        "unique_image_hashes": len(images),
        "image_hashes_shared_across_groups": sum(len(v) > 1 for v in images.values()),
        "identical_inputs_shared_across_groups": sum(len(v) > 1 for v in inputs.values()),
        "templates_crossing_splits": sorted(k for k, v in templates.items() if len(v) > 1),
        "linked_group_clusters": [sorted(v) for v in clusters.values() if len(v) > 1],
    }
