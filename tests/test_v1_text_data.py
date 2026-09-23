import hashlib,json,random,sys
from pathlib import Path

sys.path.insert(0,"scripts")
from decision_data import expand_fields,render,with_noise_state
from v1_text.common import fit_partition,stable_partition,verified_license
from v1_text.audit_mixture import audit
from v1_text.build_controls import add_controls
from v1_text.assemble_mixture import stratified_train
from v1_text.convert_classification import convert
from v1_text.convert_typed_decisions import convert as convert_typed
from v1_text.filter_synthetic_agreement import agreed

def evidence(tmp_path):
    p=tmp_path/"LICENSE";p.write_text("Apache License Version 2.0 fixture")
    (tmp_path/"LICENSE.receipt.json").write_text(json.dumps({"evidence_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"spdx":"Apache-2.0","commercial_use_reviewed":True}))
    return p

def test_license_gate_and_deterministic_group_split(tmp_path):
    p=evidence(tmp_path);assert verified_license(p,"Apache-2.0")["spdx"]=="Apache-2.0"
    (tmp_path/"LICENSE").write_text("changed")
    try:verified_license(p,"Apache-2.0");assert False
    except ValueError:pass
    assert stable_partition("customer-7")==stable_partition("customer-7")
    assert fit_partition("customer-7") in {"train","dev","calibration"}

def test_cc_by_3_license_is_commercially_admitted(tmp_path):
    p=tmp_path/"LICENSE";p.write_text("Creative Commons Attribution 3.0 fixture")
    (tmp_path/"LICENSE.receipt.json").write_text(json.dumps({"evidence_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"spdx":"CC-BY-3.0","commercial_use_reviewed":True}))
    assert verified_license(p,"CC-BY-3.0")["spdx"]=="CC-BY-3.0"

def test_classification_converter_uses_string_state_and_heldout(tmp_path):
    src=tmp_path/"rows.jsonl";src.write_text('{"id":"a","text":"hello","label":"greet"}\n{"id":"b","text":"bye","label":"leave"}\n')
    rows=convert(src,evidence(tmp_path),"Apache-2.0",source_name="massive_test",heldout=True)
    assert {r["partition"] for r in rows}=={"test"} and rows[0]["request"]["state"]=="hello"

def test_classification_preserves_official_eval_and_splits_train(tmp_path):
    src=tmp_path/"rows.jsonl";src.write_text('\n'.join(json.dumps(x) for x in [
      {"id":"a","text":"a","label":"x","split":"test"},{"id":"b","text":"b","label":"y","split":"validation"},{"id":"c","text":"c","label":"x","split":"train"}])+"\n")
    rows=convert(src,evidence(tmp_path),"Apache-2.0",source_name="intent")
    assert rows[0]["partition"]=="test" and rows[1]["partition"]=="dev" and rows[2]["partition"] in {"train","dev","calibration"}

def test_multiquestion_expansion_and_soft_unknown():
    req={"schema_version":"1.0","request_id":"r","state":"plain text","fields":[
      {"id":"a","type":"boolean","question":"True?"},{"id":"b","type":"choice","question":"Kind?","options":[{"value":"x"},{"value":"y"}]}]}
    r={"id":"r","request":req,"targets":{"a":True,"b":None},"target_distributions":{"a":{"true":.7,"false":.2,"unknown":.1},"b":{"x":.1,"unknown":.9}},"abstention_causes":{"b":"insufficient_evidence"}}
    a,b=expand_fields(r);assert a["target"] is True and b["target"] is None
    assert render(a)[3]==[.7,.2,.1] and render(b)[3]==[.1,0,.9]

def test_single_question_target_map_is_normalized_and_bad_soft_labels_fail():
    r={"id":"r","request":{"schema_version":"1.0","request_id":"r","state":"text","fields":[{"id":"a","type":"boolean","question":"True?"}]},"targets":{"a":True},"target_distributions":{"a":{"true":.9,"unknown":.1}}}
    one=expand_fields(r)[0];assert one["target"] is True and render(one)[3]==[.9,0,.1]
    one["target_distribution"]={"true":.5,"bogus":.5}
    try:render(one);assert False
    except ValueError:pass

def test_published_typed_decisions_shape_preserves_test_and_gold(tmp_path):
    row={"id":"case","workflow":"tickets","split":"test","state":json.dumps({"body":"refund"}),
      "questions":json.dumps({"route":{"type":"choice","instructions":"Route?","criteria":{"billing":"Money","support":"Technical"}},"urgent":{"type":"noul","instructions":"Urgent?","criteria":{"true":"Today","false":"Later"}},"score":{"type":"score","instructions":"Priority?","criteria":["low","high"]}}),
      "gold":json.dumps({"route":{"type":"choice","label":"billing","probabilities":{"billing":.8,"support":.2}},"urgent":{"type":"noul","label":"false","probabilities":{"false":.7,"true":.3}},"score":{"type":"score","label":"1","probabilities":{"0":.1,"1":.9}}})}
    src=tmp_path/"typed.jsonl";src.write_text(json.dumps(row)+"\n")
    converted=convert_typed(src,evidence(tmp_path))[0]
    assert converted["partition"]=="test" and converted["targets"]=={"route":"billing","urgent":False,"score":1}
    assert converted["request"]["state"]=={"body":"refund"} and len(expand_fields(converted))==3 and converted["family"]=="tickets"

def test_typed_converter_requires_official_split_and_flattens_instructions(tmp_path):
    row={"id":"x","workflow":"w","state":"plain","questions":json.dumps({"q":{"type":"noul","instructions":{"question":"Allowed?","today":"now"}}}),"gold":json.dumps({"q":{"type":"noul","label":"true","probabilities":{"false":.1,"true":.9}}})}
    src=tmp_path/"rows.jsonl";src.write_text(json.dumps(row)+"\n")
    try:convert_typed(src,evidence(tmp_path));assert False
    except ValueError as exc:assert "official split" in str(exc)
    converted=convert_typed(src,evidence(tmp_path),split="validation")[0]
    assert converted["partition"]=="dev" and converted["request"]["fields"][0]["question"]=="question: Allowed?\ntoday: now"

def test_noise_default_is_fifteen_percent():
    base={"id":"x","target":True,"abstention_cause":None,"request":{"state":{},"fields":[{"id":"q","type":"boolean","question":"Q?"}]}}
    class R:
      def random(self):return .149
      def choice(self,x):return x[0]
    assert with_noise_state(base,R())["request"]["state"]

def test_controls_are_deterministic_and_auditable(tmp_path):
    lic=verified_license(evidence(tmp_path),"Apache-2.0")
    image_path=tmp_path/"x.jpg";image_path.write_bytes(b"fixture")
    image={"id":"im","source":"vision","source_group":"im","family":"vision","license":lic,"partition":"train","images":[{"image":str(image_path),"sha256":"abc"}],"request":{"schema_version":"1.0","request_id":"im","fields":[{"id":"q","type":"boolean","question":"Q?"}],"state":{}},"target":True}
    text={"id":"tx","source":"text","source_group":"tx","family":"text","license":lic,"partition":"train","images":[],"request":{"schema_version":"1.0","request_id":"tx","fields":[{"id":"q","type":"boolean","question":"Q?"}],"state":"text"},"target":True}
    rows=add_controls([text],[image],text_image_rate=1,image_state_rate=1);assert rows==add_controls([text],[image],text_image_rate=1,image_state_rate=1)
    assert rows[1]["image_role"]=="irrelevant" and audit(rows)["ok"]

def test_audit_rejects_heldout_leak_and_unpaired_irrelevant(tmp_path):
    lic=verified_license(evidence(tmp_path),"Apache-2.0")
    row={"id":"x","source":"sst5","source_group":"g","family":"sst5","license":lic,"partition":"train","images":[{"sha256":"x"}],"image_role":"irrelevant","request":{"fields":[{"id":"q"}]},"target":1}
    result=audit([row]);assert not result["ok"] and any("held-out" in x for x in result["errors"])

def test_stratified_cap_keeps_eval_and_multiple_train_sources():
    rows=[{"id":f"a{i}","source":"a","family":"x","partition":"train"} for i in range(9)]
    rows += [{"id":f"b{i}","source":"b","family":"x","partition":"train"} for i in range(2)]
    rows += [{"id":"held","source":"a","family":"x","partition":"test"}]
    selected=stratified_train(rows,4,"s")
    assert len([r for r in selected if r["partition"]=="train"])==4 and {r["source"] for r in selected if r["partition"]=="train"}=={"a","b"}
    assert any(r["id"]=="held" for r in selected)

def test_synthetic_filter_requires_two_exact_blind_agreements():
    rows=[{"id":"yes","target":False,"blind_labels":[{"q":True},{"q":True}]},{"id":"no","blind_labels":[1,2]}]
    kept=agreed(rows);assert [r["id"] for r in kept]==["yes"] and kept[0]["targets"]=={"q":True} and "target" not in kept[0]
    try:agreed([{"id":"nan","blind_labels":[float("nan"),float("nan")]}]);assert False
    except ValueError:pass

def test_decision_quota_counts_fields_not_requests():
    rows=[{"id":"multi","source":"a","family":"x","partition":"train","request":{"fields":[1,2,3]}},
          {"id":"single","source":"b","family":"x","partition":"train","request":{"fields":[1]}}]
    selected=stratified_train(rows,2,"s")
    assert [r["id"] for r in selected]==["single"]

def test_audit_rejects_empty_unknown_partition_and_cross_partition_image(tmp_path):
    assert not audit([])["ok"]
    lic=verified_license(evidence(tmp_path),"Apache-2.0");p=tmp_path/"im";p.write_bytes(b"x")
    def row(i,part,role):return {"id":i,"source":"s","source_group":i,"family":"f","license":lic,"partition":part,"images":[{"image":str(p),"sha256":"same"}],"image_role":role,"request":{"schema_version":"1.0","request_id":i,"state":{},"fields":[{"id":"q","type":"boolean","question":"Q?"}]},"target":True}
    assert not audit([row("a","train","relevant"),row("b","dev","irrelevant")])["ok"]
    assert not audit([row("x","weird","relevant")])["ok"]


def test_audit_rejects_official_test_relabelled_train(tmp_path):
    source = tmp_path / 'classification.json'
    source.write_text(json.dumps([{'id': 'a', 'text': 'hello', 'label': 'greet', 'split': 'test'},
                                  {'id': 'b', 'text': 'bye', 'label': 'leave', 'split': 'test'}]))
    rows = convert(source, evidence(tmp_path), 'Apache-2.0', source_name='massive')
    rows[0]['partition'] = 'train'
    report = audit(rows)
    assert not report['ok'] and any('held-out' in error for error in report['errors'])
