#!/usr/bin/env bash
# Dry run of the LABEL-THEN-TRAIN path (reports/phase3/labeltrain-dryrun.md; runbook reports/phase3/LABELTRAIN-RUNBOOK.md).
# No Azure, no GPU, no upload, no pod: the "Azure session at stop time" is a copy of the earlier dry run's session
# (data/p3/dryrun/session, from scripts/p3/dryrun/run_all.sh) with its last results removed; the pod is cloud/p3/bootstrap_labeltrain.sh
# in LT_FAKE mode with keyless mock vLLM servers (scripts/p3/mock_azure.py --vllm, which refuses any credential header).
# Steps: extra queue -> orchestrator (servers, pod teacher + consistency sample, local writer, variants, leftovers via the inbox,
# writer -> teacher switch, finalize + pack, servers stopped, direct-image baseline pass) -> pull -> audit -> Kimi sample over both
# teachers -> build_manifest from both dirs (+ direct chart / safety rows) -> label + train bundle checks -> fake train stage -> ALL_DONE.
# Never touches data/p3/{stream,teacher,variants,mine-pulled} (the live session). Outputs: data/p3/dryrun/labeltrain/.
set -u
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
DRY=${LT_DRY:-$HOME/.cache/imajev-labeltrain-dry}; rm -rf $DRY; mkdir -p $DRY
ln -sfn "$REPO" $DRY/code; C=$DRY/code; cd $C          # the orchestrator assumes space-free paths (as on the pod)
PY=$C/.venv/bin/python; D=$C/data/p3/dryrun/labeltrain; S=$C/data/p3/dryrun/session; WS=$DRY/ws; M=$WS/labeltrain
step(){ echo; echo "######## $(TZ=Asia/Kolkata date +%H:%M:%S) IST  $*"; }
pass(){ echo "PASS  $*"; }; failc(){ echo "FAIL  $*"; }
[ -s $S/teacher/results.jsonl ] || { echo "FAIL  needs data/p3/dryrun/session: run MLX=0 bash scripts/p3/dryrun/run_all.sh first"; exit 1; }
for port in 8100 8101 8102 8200; do lsof -iTCP:$port -sTCP:LISTEN > /dev/null 2>&1 && { echo "FAIL  port $port is busy"; exit 1; }; done
rm -rf $D; mkdir -p $D/logs

step "1 fixture: the Azure session at stop time (copy of the dry-run session, last 80 non-parent results removed)"
mkdir -p $D/azure; cp -R $S/teacher $S/stream $S/variants $D/azure/; cp -R $S/pulled $D/azure/mine-pulled
$PY - $D/azure <<'PY'
import json, sys
from pathlib import Path
a = Path(sys.argv[1]); res = [json.loads(l) for l in open(a / "teacher/results.jsonl") if l.strip()]
parents = {json.loads(l)["parent_id"] for l in open(a / "variants/parents.jsonl") if l.strip()}
drop = [r["id"] for r in reversed(res) if r["id"] not in parents and r.get("origin") != "variant"][:80]
keep = [r for r in res if r["id"] not in set(drop)]
(a / "teacher/results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
st = json.load(open(a / "teacher/status.json")); st.update(state="running", t=__import__("time").time())
json.dump(st, open(a / "teacher/status.json", "w"))
print(f"azure fixture: {len(keep)} results kept, {len(drop)} removed (they become leftovers); status 'running'")
PY
$PY scripts/p3/azure_teacher.py finalize --out $D/azure/teacher > $D/logs/azure-finalize.json
mkdir -p $D/pool; cp -R $C/data/p3/dryrun/pool/. $D/pool/
# direct constructed image groups (charts, safety) + a held-out chart file, from the sample pool's image rows
$PY - $D <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); rows = [json.loads(l) for f in sorted((d / "pool/shards").glob("*.jsonl")) for l in open(f) if l.strip()]
img = [r for r in rows if r.get("images") and not r["pool"]["heldout_flagged_bucket"]][:24]
out = {"charts": [], "safety": []}
for i, r in enumerate(img):
    grp = "charts" if i % 2 == 0 else "safety"
    c = {k: v for k, v in r.items() if k != "pool"}
    c.update(id=f"dry-{grp}-{i}", family=f"{'chart_bar_value' if grp == 'charts' else 'safety_hazard_visible'}", dataset=grp,
             source="I", gold_kind="constructed", gold=bool(i % 3), unknown_reason=None, parent_id=None,
             field={"type": "noul", "question": f"Dry-run {grp} question {i}: is the highlighted value above the threshold {i + 3}?"},
             state={"note": f"dry-run {grp} item {i}", "threshold": i + 3},
             provenance={"licence": "generated", "group_id": f"dry-{grp}-img-{i // 2}"})
    out[grp].append(c)
(d / "direct").mkdir(exist_ok=True)
for g, rs in out.items():
    (d / "direct" / f"I-{g}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs[1:]))
held = dict(out["charts"][0], id="dry-heldout-chart-0", state={"note": "held-out chart", "threshold": 99},
            field={"type": "noul", "question": "Dry-run held-out chart: is the highlighted value above the threshold 99?"},
            provenance={"licence": "generated", "group_id": "dry-held"})
(d / "pool/heldout-fresh-charts.jsonl").write_text(json.dumps(held) + "\n")
print("direct: " + ", ".join(f"I-{g}.jsonl {len(rs) - 1}" for g, rs in out.items()) + "; held-out charts 1")
PY

step "2 make_extra_queue.py extra (replays the coordinator; never overlaps Azure)"
$PY scripts/p3/make_extra_queue.py extra --out-dir $D/labeltrain --mine-pulled $D/azure/mine-pulled --azure-queue $D/azure/stream/queue.jsonl \
  --azure-variants-queue $D/azure/stream/queue-variants.jsonl --azure-results $D/azure/teacher/results.jsonl \
  --azure-status $D/azure/teacher/status.json --pool-dir $D/pool --coordinator-status $D/azure/stream/coordinator-status.json \
  --consistency-n 40 --strict 2>&1 | tail -1 && pass "extra queue built, coordinator replay matches (--strict)" || failc "extra queue"
$PY -c "import json;s=json.load(open('$D/labeltrain/extra-summary.json'));assert s['direct_constructed_leftovers']>0 and s['direct_constructed_leftovers']+s['queue_extra']==s['waiting_total'];print('PASS  split: waiting',s['waiting_total'],'-> direct (no teacher)',s['direct_constructed_leftovers'],'+ teacher',s['queue_extra'])" || failc "constructed split"

step "3 the pod (fake): bootstrap_labeltrain.sh with mock vLLM servers (3 GPUs: teacher 0-1, writer 2)"
SERVE="$PY $C/scripts/p3/mock_azure.py --vllm --served {name} --port {port} --policy realistic --accuracy 0.88 --parse-error-rate 0.008 --truncation-rate 0.012 --latency 0.05 --gold-from $D/pod-p3/stream-pod/queue-*.jsonl --gold-from $D/pool/shards/*.jsonl"
( env LT_FAKE=1 LT_WS=$WS LT_CODE=$C LT_P3=$D/pod-p3 LT_PY=$PY LT_NGPU=3 LT_LABEL_DIR=$D/labeltrain LT_SERVE_CMD="$SERVE" LT_GPU_CHECK=0 \
    WRITER_GPU=2 PER_SERVER_CONC=6 LOOP_SECS=2 SNAP_SECS=5 WAIT_SECS=2 VARIANTS_INTERVAL=3 CONSISTENCY_MIN=0.8 POD_USD_PER_HOUR=25 \
    TEACHER_EXTRA_ARGS="--poll-secs 1 --status-secs 2 --gate-items 60" DIRECT_DIR=$D/direct \
    LT_DIRECT_MINE_CMD="$PY scripts/p3/mine.py --backend fake --fake-accuracy 0.55 --shards '{shards}' --out {out} --worker-id direct-gpu{gpu}" \
    RUN_LOG=$WS/p3/run/run.log TRAIN_CMD="echo BOOTSTRAP_DONE; mkdir -p $WS/p3/run; echo '(fake) pod_run_train.sh ALL_DONE' >> $WS/p3/run/run.log" \
    bash $C/cloud/p3/bootstrap_labeltrain.sh > $D/logs/orchestrator.log 2>&1 ) & ORCH=$!
waitfor(){ local i; for i in $(seq 1 $2); do [ -f $M/$1 ] && return 0; [ -f $M/FAILED ] && return 1; sleep 1; done; return 1; }
waitfor SERVERS_READY 120 && pass "SERVERS_READY: $(cat $M/SERVERS_READY)" || { failc "servers"; tail -20 $D/logs/orchestrator.log; }

step "4 Azure leftovers: refused while Azure runs; built after it stops; delivered through the pod inbox"
$PY scripts/p3/make_extra_queue.py leftovers --out-dir $D/labeltrain --azure-queue $D/azure/stream/queue.jsonl \
  --azure-variants-queue $D/azure/stream/queue-variants.jsonl --azure-results $D/azure/teacher/results.jsonl \
  --azure-status $D/azure/teacher/status.json > $D/logs/leftovers-refused.log 2>&1 \
  && failc "leftovers built while Azure 'running'" || pass "leftovers refused while Azure 'running': $(tail -c 120 $D/logs/leftovers-refused.log)"
$PY -c "import json,time;p='$D/azure/teacher/status.json';s=json.load(open(p));s.update(state='cap_reached',t=time.time());json.dump(s,open(p,'w'))"
$PY scripts/p3/make_extra_queue.py leftovers --out-dir $D/labeltrain --azure-queue $D/azure/stream/queue.jsonl \
  --azure-variants-queue $D/azure/stream/queue-variants.jsonl --azure-results $D/azure/teacher/results.jsonl \
  --azure-status $D/azure/teacher/status.json 2>&1 | tail -1
cp $D/labeltrain/inbox/* $M/inbox/          # = scp data/p3/labeltrain/inbox/* imajev-lt:labeltrain/inbox/
waitfor LEFTOVERS_INSTALLED 60 && pass "LEFTOVERS_INSTALLED: $(cat $M/LEFTOVERS_INSTALLED)" || failc "leftovers not installed"

step "5 labelling runs to LABEL_DONE, servers stopped, direct baseline, TRAIN_WAITING (a pilot-gate pause is resumed as the lead would)"
for i in $(seq 1 900); do
  [ -f $M/TRAIN_WAITING ] || [ -f $M/FAILED ] && break
  if [ -f $M/PILOT_GATE_PAUSED ] && [ ! -f $D/logs/gate-resumed ]; then
    pass "PILOT_GATE_PAUSED raised: $(sed -n 3p $D/pod-p3/teacher-pod/PILOT_GATE_ALERT.txt)"
    $PY scripts/p3/pod_teacher.py resume --out $D/pod-p3/teacher-pod && touch $D/logs/gate-resumed; fi
  sleep 1; done
[ -f $M/TRAIN_WAITING ] && pass "TRAIN_WAITING" || { failc "no TRAIN_WAITING"; tail -30 $D/logs/orchestrator.log; }
grep -E "MARKER|WATCHDOG|WARNING|FAILED|FP8-vs-bf16|writer done" $D/logs/orchestrator.log | cut -c1-230
tail -n 2 $M/throughput.log | cut -c1-300

step "6 pull (the lead: scp p3-label-results.tgz) and audit"
mkdir -p $D/pulled && tar xzf $WS/p3-label-results.tgz -C $D/pulled && pass "p3-label-results.tgz extracted ($(du -sh $WS/p3-label-results.tgz | cut -f1))"
$PY scripts/p3/make_extra_queue.py audit --azure-results $D/azure/teacher/results.jsonl --pod-results $D/pulled/teacher-pod/results.jsonl \
  > $D/logs/audit.json && pass "audit: no id judged by both teachers" || failc "audit overlap"
$PY scripts/p3/dryrun/labeltrain_checks.py pod $D || failc "pod checks (exit $?)"

step "7 Kimi sample over both teachers (sample only; no reviewer calls)"
mkdir -p $D/review
$PY scripts/p3/kimi_review.py sample --teacher $D/azure/teacher/kept.jsonl $D/pulled/teacher-pod/kept.jsonl $D/azure/variants/constructed.jsonl \
  $D/pulled/variants-pod/constructed.jsonl --out $D/review/sample.jsonl --n 80 --n-image 16 --force > $D/logs/kimi-sample.log 2>&1 \
  && pass "kimi sample: $(tail -n 1 $D/logs/kimi-sample.log | cut -c1-200)" || failc "kimi sample (see logs/kimi-sample.log)"

step "8 build_manifest from BOTH teacher dirs (+ both variant dirs, direct chart/safety rows), publish decision-p3ltdry"
$PY scripts/p3/build_manifest.py --teacher-dirs $D/azure/teacher $D/pulled/teacher-pod \
  --verified $D/azure/variants/constructed.jsonl $D/pulled/variants-pod/constructed.jsonl \
  --direct-constructed $D/direct/I-charts.jsonl $D/direct/I-safety.jsonl \
  --direct-leftovers $D/labeltrain/direct-constructed-leftovers.jsonl $D/labeltrain/direct-constructed-leftovers-late.jsonl \
  --review-dev $C/data/p3/dryrun/review/dev-slice.jsonl --drop-ids $C/data/p3/dryrun/review/drop-ids.json \
  --share-hard-image 0.15 --share-replay-image 0.20 --pool-dir $D/pool --out-dir $D/manifests --publish decision-p3ltdry \
  > $D/logs/build_manifest.log 2>&1 && pass "build_manifest rc 0" || { failc "build_manifest (see logs/build_manifest.log)"; tail -5 $D/logs/build_manifest.log; }
grep -E "teacher dirs|direct constructed|direct image rows|final mixture|verified:|mixture|gate lane|^.{8} OK|FAILED" $D/logs/build_manifest.log | cut -c1-260
$PY scripts/p3/dryrun/labeltrain_checks.py manifest $D || failc "manifest checks (exit $?)"

step "9 bundles: p3-label (code + label files + direct candidates) and p3-train (manifests, no pool) must pass the benchmark check"
mkdir -p $D/bundles
$PY scripts/p3/transfer.py build --preset code --name p3ltdry-label --out-dir $D/bundles --include data/p3/dryrun/labeltrain/labeltrain=data/p3/labeltrain \
  --exclude 'data/p3/labeltrain/direct-*' --exclude 'data/p3/labeltrain/inbox/*' \
  --candidates "data/p3/dryrun/labeltrain/direct/I-*.jsonl" 2>&1 | tail -1
$PY scripts/p3/transfer.py build --preset train --name p3ltdry-train --out-dir $D/bundles \
  --manifest decision-p3ltdry --manifest decision-p3ltdry-lane255 --manifest decision-p3ltdry-labelled --manifest decision-p3ltdry-heldout-fresh \
  --manifest decision-p3ltdry-heldout-fresh-large-choice --manifest decision-p3ltdry-heldout-flagged --manifest decision-p3ltdry-human-dev \
  --manifest decision-p3ltdry-heldout-charts --manifest decision-p2b --manifest decision-v2-state-probe --manifest decision-v2-pairs-probe \
  --manifest decision-v1.1-irrelevance-test 2>&1 | tail -1
$PY - $D/bundles <<'PY'
import sys, shutil
sys.path.insert(0, "scripts/p3")
import transfer as T
from pathlib import Path
out = Path(sys.argv[1])
for name, want in (("p3ltdry-label", "data/p3/labeltrain/queue-extra.jsonl"), ("p3ltdry-train", "data/manifests/decision-p3ltdry-heldout-charts.jsonl")):
    ext = out / f"x-{name}"; shutil.rmtree(ext, ignore_errors=True)
    m = T.verify_and_extract(name, out, ext)
    ok = m["check"]["status"] == "PASS" and (ext / want).exists() and (ext / "scripts/p3/pod_teacher.py").exists()
    print(f"{'PASS' if ok else 'FAIL'}  {name}: check {m['check']['status']}, {len(m['files'])} files, {m['archive_bytes'] / 1e6:.1f} MB, has {want}")
    shutil.rmtree(ext, ignore_errors=True)
PY
[ $? = 0 ] || failc "bundle verify (a bundle was not built or did not verify)"

step "10 train stage (fake): the lead touches TRAIN_BUNDLE_READY"
touch $M/TRAIN_BUNDLE_READY
waitfor ALL_DONE 60 && pass "ALL_DONE: $(cat $M/ALL_DONE)" || failc "no ALL_DONE"
wait $ORCH; echo "orchestrator exit $?"
ls $M | tr '\n' ' '; echo

step "11 cleanup"
rm -f $C/data/manifests/decision-p3ltdry*.jsonl $C/data/manifests/decision-p3ltdry*-images.json
for f in $M/pids/*.pid; do [ -f $f ] && kill -TERM -- -$(cat $f) 2>/dev/null; done
echo "left in data/manifests: $(ls $C/data/manifests | grep -c p3ltdry || true)"
