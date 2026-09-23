#!/usr/bin/env python3
"""Build human-review candidates from inspected photos and authored text cases."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import random
import shutil
import sys
from pathlib import Path
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).parent))
from build_exposure_inventory import contains
from imajev_bench.schema import validate_records
from imajev_bench.audit import audit


def choice(question, values):
    return {"id":"decision","type":"choice","question":question,"options":[{"value":x,"description":x} for x in values]}


def boolean(question):
    return {"id":"decision","type":"boolean","question":question}


def spec(field,gold,evidence,joint,unknown_field,unknown_evidence):
    return {"field":field,"gold":gold,"evidence":evidence,"joint":joint,"unknown_field":unknown_field,"unknown_evidence":unknown_evidence}


PHOTO_CASES={
"commons-menu-board-joshcool776":spec(
    choice("Under the MONDAY heading, which opponent is printed after Arsenal vs?",["Cardiff","Fulham","Chelsea","Tottenham"]),"Cardiff","Upper-middle board: MONDAY, Arsenal vs Cardiff.",
    [( {"candidate_opponent":name,"rule":"Publish the candidate only if it is the opponent printed for Arsenal under MONDAY on this board. Otherwise withhold. Require visible text, not knowledge of actual fixtures."},gold) for name,gold in [("Cardiff","accept"),("Fulham","reject")]],
    choice("Which four-digit year is explicitly printed for these fixtures? Return unknown when no year is readable; do not infer it from a calendar.",["2008","2009","2010","2011"]),"Dates and weekdays appear, but no four-digit fixture year is readable."),
"commons-hibernian-menu-board":spec(
    choice("What euro price is printed beside Homemade Rustic Soup?",["5.50","6.50","7.50","19.00"]),"6.50","Soup price at upper right is 6.50.",
    [({"order":["Homemade Rustic Soup","Evening Roast Joint"],"budget_eur":cap,"rule":"Accept when the sum of the two printed item prices is no greater than the budget. Use displayed prices only, with no additional charges."},gold) for cap,gold in [(20,"reject"),(21,"accept")]],
    choice("What euro delivery charge does this photographed board specify? If the board gives no delivery charge, return unknown.",["0.00","2.00","3.50","5.00"]),"The board lists menu prices but no delivery charge; do not assume delivery is free."),
"commons-cta-speed-limit":spec(
    choice("Which number is printed as the speed limit on the foreground sign?",["5","10","15","20"]),"10","Foreground sign reads Speed limit 10 mph.",
    [({"sample_value_mph":value,"rule":"This is a fictional record-validation task. Accept if sample_value_mph is at most the number on the mph sign; reject if greater. Do not assess real-world permission or safe operation."},gold) for value,gold in [(8,"accept"),(12,"reject")]],
    choice("Which station name is explicitly readable in this photograph? If no station name is readable, return unknown.",["Central","North","South","West"]),"No station name is visible on the numeric sign or readable elsewhere."),
"commons-caringbah-speed-limit-50":spec(
    choice("What number is inside the red circle?",["30","40","50","60"]),"50","Large black 50 inside the red circle.",
    [({"sample_value":value,"rule":"For a fictional data check, compare numbers only. Accept if sample_value is no greater than the number inside the red circle. Reject otherwise. No unit or traffic-law inference is required."},gold) for value,gold in [(45,"accept"),(55,"reject")]],
    choice("What speed unit is explicitly printed on the sign face? If no unit is printed, return unknown rather than inferring from location.",["mph","km/h","m/s"]),"The sign face gives 50 without a printed unit."),
"commons-hirschegg-no-parking-sign":spec(
    choice("What does the large triangular sign's pictogram depict?",["a car suspended from a hook","a bicycle","a pedestrian","a bus"]),"a car suspended from a hook","The triangular sign depicts a car lifted by lines attached to a hook.",
    [({"required_pictogram":symbol,"rule":"Accept this photograph for an icon catalogue only if the large triangular sign depicts required_pictogram. Reject if it depicts a different listed object. This is image categorization, not parking advice."},gold) for symbol,gold in [("a car suspended from a hook","accept"),("a bicycle","reject")]],
    choice("What monetary towing fee is stated on the signs? Return unknown if no fee is readable.",["50","100","150","200"]),"Neither visible sign supplies a monetary fee."),
"commons-deodorant-back-label":spec(
    boolean("Does the visible INGREDIENTS list include NATURAL BEESWAX (CERA ALBA)?"),True,"NATURAL BEESWAX (CERA ALBA) is a readable ingredient line.",
    [({"required_literal":literal,"rule":"Accept a catalogue record only when required_literal appears as a complete ingredient phrase on the photographed INGREDIENTS list; reject if the complete readable list does not contain it. Ignore spelling case. Do not infer product suitability or safety."},gold) for literal,gold in [("NATURAL BEESWAX (CERA ALBA)","accept"),("OLIVE OIL","reject")]],
    choice("What percentage of the product is coconut oil, according to this label? Return unknown if no percentage is printed.",["1%","5%","10%","20%"]),"Ingredients are listed without percentage quantities."),
}


def image_checks(asset, directory, database):
    checks=[]
    for name,value in [("downloaded",asset['original_receipt']['sha256']),("benchmark_jpeg",asset['image']['sha256'])]:
        checks.append({"kind":"image_sha256","basis":name,"value":value,"matched":contains(database,'image_sha256',value)})
    # Check the legacy converter's documented 1MP and 400k RGB/JPEG90 variants.
    with Image.open(directory/"originals"/(asset['asset_id']+'.bin')) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        for pixels in (1_000_000,400_000):
            scale=min(1,math.sqrt(pixels/(im.width*im.height)))
            resized=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),Image.Resampling.LANCZOS)
            blob=io.BytesIO();resized.save(blob,'JPEG',quality=90)
            value=hashlib.sha256(blob.getvalue()).hexdigest()
            checks.append({"kind":"image_sha256","basis":f"legacy_jpeg90_{pixels}px","value":value,"matched":contains(database,'image_sha256',value)})
    source=asset['source']
    for value in (source['source_id'],source['file_page'],source['download_url'],asset['original_receipt']['resolved_url']):
        for kind in ('source_id','image_ref'):
            checks.append({"kind":kind,"value":value,"matched":contains(database,kind,value)})
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acquisitions',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--text-cases',type=Path,required=True)
    p.add_argument('--photo-cases',type=Path,help='Additional pixel-inspected provisional specifications keyed by source ID')
    args=p.parse_args()
    cases=dict(PHOTO_CASES)
    if args.photo_cases:
        additional=json.loads(args.photo_cases.read_text())
        if set(additional)&set(cases):raise ValueError('Additional photo cases must not replace existing specifications')
        cases.update(additional)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'assets').mkdir()
    rows=[];sources=[];excluded=[];counter=0
    def add(group,track,family,field,gold,state,evidence,images,provenance):
        nonlocal counter
        counter+=1;rid=f"fresh-{counter:04d}"
        field=json.loads(json.dumps(field))
        if field['type']=='choice':random.Random(9217+counter).shuffle(field['options'])
        request={"request_id":rid,"state":state,"fields":[field]}
        rows.append({"id":rid,"group_id":group,"track":track,"family":family,"split":"dev","images":images,
            "request":request,"gold":gold,"annotation_status":"draft","provenance":{**provenance,"draft_evidence":evidence,
            "label_origin":"assistant-authored provisional label; no human annotation","review_protocol":"imajev-bench-blind-review-v1"}})
    seen_sources=set()
    for directory in args.acquisitions:
        acquisition=json.loads((directory/'acquisition.json').read_text())
        for asset in acquisition['assets']:
            sid=asset['source']['source_id']
            if sid in seen_sources:continue
            seen_sources.add(sid)
            if sid not in cases:
                excluded.append({"source_id":sid,"reason":"No pixel-inspected task specification"});continue
            checks=image_checks(asset,directory,args.inventory)
            if any(x['matched'] for x in checks):
                excluded.append({"source_id":sid,"reason":"Recorded local overlap","checks":checks});continue
            short=hashlib.sha256(sid.encode()).hexdigest()[:12];group='scene-'+short
            dest=args.output/'assets'/f'{short}.jpg';shutil.copyfile(directory/asset['image']['path'],dest)
            image={"path":f'assets/{short}.jpg',"sha256":hashlib.sha256(dest.read_bytes()).hexdigest()}
            source={**asset,"local_image":image,"overlap_checks":checks,"acquisition_root":str(directory)};sources.append(source)
            case=cases[sid]
            provenance={"source":"Wikimedia Commons","source_id":sid,"source_page":asset['source']['file_page'],"creator":asset['source']['creator'],"license":asset['source']['license'],"license_url":asset['source']['license_url'],"source_group":sid,"synthetic":False,"source_receipt":str(directory/'receipts'/(asset['asset_id']+'.json'))}
            add(group,'visual','visible_evidence',case['field'],case['gold'],{"instruction":"Answer from the supplied photograph only."},case['evidence'],[image],provenance)
            for state,gold in case['joint']:
                add(group,'joint','written_rule',choice("Does this case meet the supplied rule?",['accept','reject']),gold,state,case['evidence']+' Apply the explicit rule in state.',[image],provenance)
            add(group,'visual','insufficient_evidence',case['unknown_field'],None,{"instruction":"Use explicit visible evidence. Do not infer missing facts from source, location, prior knowledge or filenames."},case['unknown_evidence'],[image],provenance)
    text=json.loads(args.text_cases.read_text())
    for i,case in enumerate(text):
        add(f'text-case-{i+1:02d}','text',case['family'],case['field'],case['gold'],case['state'],case['evidence'],[],{"source":"newly authored benchmark scenario","synthetic":True,"author":"Sol drafting agent","source_group":f'text-case-{i+1:02d}'})
    rows=validate_records(rows,args.output)
    records=args.output/'records.jsonl';records.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest={"status":"draft_human_review_pending","records_sha256":hashlib.sha256(records.read_bytes()).hexdigest(),"builder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"sources":sources,"excluded":excluded,"audit":audit(rows),"inventory":str(args.inventory),"independence_note":"Related visual and joint items share one scene group. All are development candidates; no hidden/test or pretraining-clean claim."}
    (args.output/'collection.json').write_text(json.dumps(manifest,indent=2))
    attribution=['# Attributions and source receipts','','Photographs retain their per-file licenses. No photo is relicensed by the benchmark. Display copies have the changes noted below.','']
    for a in sources:
        s=a['source'];attribution.append(f"- [{s['source_id']}]({s['file_page']}) — {s['creator']}, [{s['license']}]({s['license_url']}); {a['changes']}.")
    (args.output/'ATTRIBUTION.md').write_text('\n'.join(attribution)+'\n')
    print(json.dumps({"records":len(rows),"source_photos":len(sources),"groups":len({r['group_id'] for r in rows}),"excluded":excluded},indent=2))


if __name__=='__main__':main()
