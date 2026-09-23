import hashlib
import json
import sys

sys.path.insert(0, 'scripts')
from v1_text import summarize_acquisition as registry


def test_registry_fails_closed_on_unverified_or_corrupt_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, 'ROOT', tmp_path)
    monkeypatch.setattr(registry, 'EXPECTED', ['banking77'])
    payload = tmp_path / 'train.json'
    payload.write_text('[1]')
    evidence = tmp_path / 'data/decision-v1-text/licenses/banking77/LICENSE'
    evidence.parent.mkdir(parents=True)
    evidence.write_text('Apache License Version 2.0 fixture')
    receipt = evidence.with_name('LICENSE.receipt.json')
    receipt.write_text(json.dumps({'spdx': 'Apache-2.0', 'commercial_use_reviewed': True,
        'evidence_sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}))
    entry = {'source': 'banking77', 'training_approved': True, 'acquisition_status': 'complete',
             'files': [{'path': 'train.json', 'bytes': 3,
                        'sha256': hashlib.sha256(payload.read_bytes()).hexdigest()}]}
    report = tmp_path / 'inventory.json'
    def result(verify=True):
        report.write_text(json.dumps({'sources': [entry]}))
        return registry.consolidate([report], verify)['sources'][0]
    assert result()['training_admitted']
    assert not result(False)['training_admitted']
    entry['failures'] = ['required test split failed']
    assert not result()['training_admitted']
    del entry['failures']
    payload.write_text('[2]')
    assert not result()['training_admitted']
    payload.write_text('[1]')
    evidence.write_text('changed evidence')
    assert not result()['training_admitted']
