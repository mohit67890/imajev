"""Portable blind review packets; exports annotations, never approval claims."""
import base64
import hashlib
import html as html_module
import json
import mimetypes
import random
from collections import defaultdict
from pathlib import Path

from .runner import digest
from .schema import model_payload

PROTOCOL = "imajev-bench-blind-review-v1"
FORMAT_VERSION = "0.0.1"


def _packet_order(rows, reviewer_id, seed):
    """Separate variants from the same group while retaining reproducibility."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    rng = random.Random(hashlib.sha256(f"{seed}\0{reviewer_id or ''}".encode()).digest())
    group_ids = sorted(groups)
    rng.shuffle(group_ids)
    for group_id in group_ids:
        rng.shuffle(groups[group_id])
    ordered = []
    for round_number in range(max(map(len, groups.values()), default=0)):
        ordered.extend(groups[group_id][round_number] for group_id in group_ids if round_number < len(groups[group_id]))
    return ordered


def build_review(records, root, output, reviewer_id=None, seed=0):
    """Build a reproducible, self-contained packet for an independent reviewer."""
    root, output = Path(root), Path(output)
    if reviewer_id is not None and (not isinstance(reviewer_id, str) or not reviewer_id.strip() or reviewer_id != reviewer_id.strip()):
        raise ValueError("reviewer_id must be a trimmed non-empty string")
    rows, hashes = [], []
    for record in records:
        images = []
        for asset in record["images"]:
            path = (root / asset["path"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Image escapes dataset root")
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            images.append("data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode())
        input_sha256 = digest(model_payload(record))
        hashes.append((record["id"], input_sha256))
        rows.append({k: record[k] for k in ("id", "track", "group_id", "request")} | {"images": images, "input_sha256": input_sha256})
    dataset_sha256 = hashlib.sha256(json.dumps(sorted(hashes), separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    rows = _packet_order(rows, reviewer_id, seed)
    for row in rows:
        del row["group_id"]
    payload = base64.b64encode(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).decode()
    config = base64.b64encode(json.dumps({"format_version": FORMAT_VERSION, "protocol": PROTOCOL, "dataset_sha256": dataset_sha256, "reviewer_id": reviewer_id, "seed": seed}, separators=(",", ":")).encode()).decode()
    credits = set()
    for record in records:
        p = record.get("provenance", {})
        for source in p.get("image_sources", [p]):
            if record["images"] and source.get("license"):
                credits.add(" | ".join(str(source.get(k, "")) for k in ("creator", "license", "license_url", "source_page")))
    credit_html = ""
    if credits:
        credit_html = '<details><summary>Image credits and licenses — consult after independent review</summary><p>Display copies were orientation-corrected, resized and encoded as JPEG; metadata was removed. Per-image licenses continue to apply.</p><ul>' + ''.join('<li>' + html_module.escape(c) + '</li>' for c in sorted(credits)) + '</ul></details>'
    html = TEMPLATE.replace("__RECORDS_B64__", payload).replace("__CONFIG_B64__", config).replace("__CREDITS__", credit_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(html)
    return output


TEMPLATE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>imajev-bench · Blind review packet</title>
<!-- Record data is base64 encoded; this also protects the historic \u003c/script> boundary. -->
<style>
body{font:16px/1.5 system-ui,sans-serif;background:#f5f6f8;color:#17202d;max-width:1050px;margin:32px auto;padding:0 24px}h1{font-size:30px;margin-bottom:4px}.muted{color:#536174}.panel{background:white;border:1px solid #dbe0e7;border-radius:12px;padding:24px;margin:20px 0}button,select,input,textarea{font:inherit;padding:9px;border:1px solid #abb6c4;border-radius:6px}button{cursor:pointer;background:#fff}button:focus,select:focus,input:focus,textarea:focus{outline:3px solid #adcafa}.toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.images{display:flex;gap:12px;flex-wrap:wrap}.images img{max-width:100%;width:450px;object-fit:contain;border:1px solid #ddd}.images figure{margin:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f8;padding:16px}textarea{display:block;width:95%;min-height:85px;margin:10px 0}.banner{padding:14px;background:#fff1ce;border-radius:8px}.options label{display:block;padding:10px;border-bottom:1px solid #ddd}#progress{font-weight:600}.danger{color:#842029}
</style>
<h1>imajev-bench</h1><p class="muted">Independent blind annotation packet</p>
<p class="banner">Reference labels, record grouping, and generator metadata are hidden. This page records one reviewer's judgments; it does not approve benchmark items. Drafts stay in this browser profile until exported or explicitly reset.</p>
<div class="toolbar"><label for="reviewer">Reviewer ID</label><input id="reviewer" autocomplete="off" placeholder="Your unique audit ID"><label for="mode">Review mode</label><select id="mode"><option value="independent_review">Independent</option><option value="adjudication">Adjudication</option></select><label for="track">Track</label><select id="track"><option value="">All</option><option>text</option><option>visual</option><option>joint</option></select><button id="export">Export reviews</button><button id="reset" class="danger">Reset local draft</button></div>
<div class="panel"><div class="toolbar"><button id="prev">Previous</button><span id="progress"></span><button id="next">Next</button></div><p id="meta" class="muted"></p><div class="images" id="images"></div><h2>State and policy</h2><pre id="state"></pre><h2 id="question"></h2><div class="options" id="options"></div><label for="evidence">Evidence and rule clause</label><textarea id="evidence" placeholder="Describe visible evidence and the rule supporting your judgment. Explain missing evidence when unknown."></textarea><div class="toolbar"><button id="save">Save judgment</button><button id="flag">Flag / reject item</button></div><label for="flag-reason">Flag reason (required when flagging)</label><textarea id="flag-reason" placeholder="Unreadable image, ambiguity, leakage, mismatched evidence, or another protocol problem."></textarea><span id="notice" role="status"></span></div>
__CREDITS__
<script>
const records=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('__RECORDS_B64__'),c=>c.charCodeAt(0))));
const config=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('__CONFIG_B64__'),c=>c.charCodeAt(0))));
let selection=records,position=0,reviews={},flags={};const el=id=>document.getElementById(id);const storageKey=id=>`${config.protocol}:${config.dataset_sha256}:${id}`;
function loadDraft(){const id=el('reviewer').value.trim();reviews={};flags={};if(!id)return;try{const saved=JSON.parse(localStorage.getItem(storageKey(id))||'{}');reviews=saved.reviews||{};flags=saved.flags||{}}catch(_){el('notice').textContent=' Stored draft could not be read.'}}
function persist(){const id=el('reviewer').value.trim();if(id)localStorage.setItem(storageKey(id),JSON.stringify({reviews,flags}))}
function render(){const row=selection[position];if(!row){el('progress').textContent='0 / 0';return}const packetIndex=records.indexOf(row)+1;el('progress').textContent=`${position+1} / ${selection.length} · ${Object.keys(reviews).length} judged · ${Object.keys(flags).length} flagged`;el('meta').textContent=`Item ${packetIndex} · ${row.track}`;el('state').textContent=typeof row.request.state==='string'?row.request.state:JSON.stringify(row.request.state,null,2);const f=row.request.fields[0];el('question').textContent=f.question;el('images').replaceChildren();row.images.forEach((src,i)=>{const fig=document.createElement('figure'),img=document.createElement('img'),cap=document.createElement('figcaption');img.src=src;img.alt=`Image ${i+1}`;cap.textContent=`Image ${i+1}`;fig.append(img,cap);el('images').append(fig)});const options=f.type==='boolean'?[{value:true,description:f.yes_description||'Yes'},{value:false,description:f.no_description||'No'}]:f.type==='ordinal'?f.levels:f.options;el('options').replaceChildren();[...options,{value:null,description:'The requested answer cannot be established from the supplied evidence'}].forEach(o=>{const label=document.createElement('label'),input=document.createElement('input');input.type='radio';input.name='value';input.value=JSON.stringify(o.value);input.checked=reviews[row.id]!==undefined&&JSON.stringify(reviews[row.id].value)===input.value;label.append(input,document.createTextNode(' '+(o.value===null?'Unknown':String(o.value))+' — '+(o.description||'')));el('options').append(label)});el('evidence').value=reviews[row.id]?.evidence||'';el('flag-reason').value=flags[row.id]?.reason||'';el('notice').textContent=flags[row.id]?' This item is flagged and has no ordinary judgment.':'';el('prev').disabled=position===0;el('next').disabled=position===selection.length-1}
el('reviewer').value=config.reviewer_id||'';if(config.reviewer_id)el('reviewer').readOnly=true;loadDraft();el('reviewer').onchange=()=>{loadDraft();render()};
el('save').onclick=()=>{const row=selection[position],id=el('reviewer').value.trim(),chosen=document.querySelector('input[name=value]:checked');if(!id||!chosen||!el('evidence').value.trim()){el('notice').textContent=' Add reviewer ID, decision, and evidence.';return}delete flags[row.id];reviews[row.id]={id:row.id,input_sha256:row.input_sha256,reviewer_id:id,value:JSON.parse(chosen.value),evidence:el('evidence').value.trim()};persist();render();el('notice').textContent=' Judgment saved locally.'};
el('flag').onclick=()=>{const row=selection[position],id=el('reviewer').value.trim(),reason=el('flag-reason').value.trim();if(!id||!reason){el('notice').textContent=' Add reviewer ID and a flag reason.';return}delete reviews[row.id];flags[row.id]={id:row.id,reviewer_id:id,input_sha256:row.input_sha256,reason};persist();render();el('notice').textContent=' Item flagged; no ordinary judgment will be exported for it.'};
el('prev').onclick=()=>{position--;render()};el('next').onclick=()=>{position++;render()};el('track').onchange=()=>{selection=records.filter(r=>!el('track').value||r.track===el('track').value);position=0;render()};
el('reset').onclick=()=>{const id=el('reviewer').value.trim();if(!id){el('notice').textContent=' Enter the reviewer ID whose local draft should be reset.';return}if(confirm('Permanently clear this reviewer draft from this browser?')){localStorage.removeItem(storageKey(id));reviews={};flags={};render();el('notice').textContent=' Local draft reset.'}};
el('export').onclick=()=>{const id=el('reviewer').value.trim();if(!id){el('notice').textContent=' Add reviewer ID before export.';return}const data={format_version:config.format_version,protocol:config.protocol,input_sha256:config.dataset_sha256,reviewer_id:id,purpose:el('mode').value,reviews:Object.values(reviews),flags:Object.values(flags)};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`imajev-bench-review-${id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();
</script></html>'''
