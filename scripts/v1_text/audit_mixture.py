"""Fail-closed audit for a decision-v1-text or mixed JSONL manifest."""
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path
from .common import verified_license
from decision_data import expand_fields,render

HELDOUT={"mmlu","sst5","sst_5","massive_test","typed_decisions_test"}

def audit(rows):
    errors=[];ids=set();groups={};images=collections.defaultdict(set);image_partitions=collections.defaultdict(set);counts=collections.Counter();license_files={};license_cache={}
    if not rows:errors.append("manifest is empty")
    for n,r in enumerate(rows):
        rid=r.get("id",f"row-{n}");source=str(r.get("source",""));partition=r.get("partition");counts[(source,partition)]+=1
        if partition not in {"train","dev","calibration","test"}:errors.append(f"{rid}: invalid partition {partition!r}")
        if rid in ids:errors.append(f"{rid}: duplicate id")
        ids.add(rid);group=f"{source}:{r.get('source_group',rid)}";groups.setdefault(group,set()).add(partition)
        lic=r.get("license")
        try:
            if not isinstance(lic,dict):raise ValueError("license is not an evidence object")
            key=(lic.get("evidence"),lic.get("spdx"))
            if key not in license_cache:license_cache[key]=verified_license(Path(lic.get("evidence","")),lic.get("spdx"))
            checked=license_cache[key]
            if any(lic.get(k)!=checked.get(k) for k in ("spdx","evidence","evidence_sha256","receipt")):raise ValueError("record does not match verified receipt")
            license_files[source]=(lic.get("spdx"),lic.get("evidence"),lic.get("evidence_sha256"))
        except Exception as exc:errors.append(f"{rid}: unverified/noncommercial license: {exc}")
        family=str(r.get("family","")).lower().replace("-","_")
        if (any(h in family or h in source.lower().replace("-","_") for h in HELDOUT) or r.get("heldout_family") is True or str(r.get("source_split", "")).lower() == "test") and partition!="test":errors.append(f"{rid}: held-out family outside test")
        if partition=="calibration" and r.get("images"): # tracked separately below, explicit for report
            counts[("_control","calibration_images")]+=1
        for im in r.get("images",[]):
            key=im.get("sha256",im.get("image",""));images[(key,partition)].add(r.get("image_role","relevant"));image_partitions[key].add(partition)
        for im in r.get("images",[]):
            path=Path(im.get("image",""))
            if not path.is_file():errors.append(f"{rid}: missing image file {path}")
        request=r.get("request",{});fields=request.get("fields",[])
        if not fields:errors.append(f"{rid}: no questions")
        if len(fields)>1 and not isinstance(r.get("targets"),dict):errors.append(f"{rid}: multi-question target map missing")
        try:
            for item in expand_fields(r):render(item)
        except Exception as exc:errors.append(f"{rid}: invalid request/target/distribution: {exc}")
    for g,parts in groups.items():
        if len(parts)>1:errors.append(f"{g}: group leaks across partitions {sorted(parts, key=str)}")
    leaking=[sha for sha,parts in image_partitions.items() if len(parts)>1]
    if leaking:errors.append(f"{len(leaking)} image hashes leak across partitions")
    bad_controls=[sha for sha,roles in images.items() if "irrelevant" in roles and "relevant" not in roles]
    if bad_controls:errors.append(f"{len(bad_controls)} irrelevant-control images never appear as relevant")
    return {"ok":not errors,"errors":errors,"counts":{f"{a}/{b}":n for (a,b),n in sorted(counts.items(), key=lambda item: str(item[0]))},
            "licenses":{k:{"spdx":v[0],"evidence":v[1],"sha256":v[2]} for k,v in sorted(license_files.items())},
            "unique_groups":len(groups),"unique_images":len(images)}

def main():
    p=argparse.ArgumentParser();p.add_argument("manifest",type=Path);p.add_argument("--report",type=Path);a=p.parse_args()
    result=audit([json.loads(x) for x in a.manifest.open() if x.strip()]);text=json.dumps(result,indent=2)+"\n"
    if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(text)
    print(text,end="");raise SystemExit(0 if result["ok"] else 1)
if __name__=="__main__":main()
