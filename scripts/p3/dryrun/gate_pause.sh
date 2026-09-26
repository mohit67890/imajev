#!/usr/bin/env bash
# Pilot-gate pause/resume exercise: the same queue through a mock with 6% parse errors; the gate (100 items) must pause dispatch,
# send nothing while paused, and resume after `azure_teacher.py resume`.
set -u; cd "$(dirname "$0")/../../.."
D=data/p3/dryrun; S=$D/session; G=$D/gate; PY=.venv/bin/python; PORT=8912
rm -rf $G; mkdir -p $G/logs
$PY scripts/p3/mock_azure.py --port $PORT --key-file $S/key --policy realistic --gold-from "$D/pool/shards/*.jsonl" \
   --latency 0.4 --accuracy 0.88 --parse-error-rate 0.06 --truncation-rate 0.01 > $G/logs/mock.log 2>&1 & MOCK=$!
sleep 1
$PY scripts/p3/azure_teacher.py instances --set 8 --out $G/teacher > /dev/null
$PY scripts/p3/azure_teacher.py run --deployment imajev-teacher:8 --queue $S/stream/queue.jsonl --out $G/teacher \
   --base-url http://127.0.0.1:$PORT/openai/v1 --key-file $S/key --per-instance-concurrency 4 --poll-secs 1 --status-secs 2 \
   --gate-items 100 > $G/logs/teacher.log 2>&1 & TEACH=$!
for i in $(seq 1 120); do grep -q '"decision": "paused"' $G/teacher/pilot-gate.json 2>/dev/null && break; sleep 1; done
echo "gate: $(python3 -c "import json;d=json.load(open('$G/teacher/pilot-gate.json'));print(d['decision'], d['why'])")"
sleep 6                                   # in-flight requests drain
n1=$(wc -l < $G/teacher/solves.jsonl); r1=$(python3 -c "import json;print(json.load(open('$G/teacher/status.json'))['state'])")
sleep 10
n2=$(wc -l < $G/teacher/solves.jsonl); r2=$(python3 -c "import json;print(json.load(open('$G/teacher/status.json'))['state'])")
echo "while paused: solves $n1 -> $n2 over 10 s; state $r1 / $r2; alert file: $(test -s $G/teacher/PILOT_GATE_ALERT.txt && echo present)"
$PY scripts/p3/azure_teacher.py resume --out $G/teacher
wait $TEACH; echo "teacher exit $?"
kill $MOCK; wait $MOCK 2>/dev/null
$PY scripts/p3/azure_teacher.py instances --set 0 --out $G/teacher > /dev/null
python3 -c "import json;d=json.load(open('$G/teacher/pilot-gate.json'));s=json.load(open('$G/teacher/status.json'));print('final gate', d['decision'], '| state', s['state'], '| results', s['results'])"
ls $G/teacher
