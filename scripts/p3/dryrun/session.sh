#!/usr/bin/env bash
# Dry run of the streamed session (reports/phase3/dry-run.md; called by run_all.sh): mock Azure (realistic), coordinator via local rsync, teacher on
# 8 simulated instances with the pilot gate, variants without a writer, then fake mining workers started after the consumers.
set -u; cd "$(dirname "$0")/../../.."
D=data/p3/dryrun; S=$D/session; PY=.venv/bin/python; PORT=${PORT:-8911}
rm -rf $S $D/pod/mine; mkdir -p $S/logs $S/stream
echo "fake-key-for-the-mock-$(date +%s)" > $S/key; chmod 600 $S/key
$PY scripts/p3/mock_azure.py --port $PORT --key-file $S/key --policy realistic --gold-from "$D/pool/shards/*.jsonl" \
   --gold-from "$S/stream/queue*.jsonl" --latency ${LATENCY:-0.4} --accuracy 0.88 --parse-error-rate 0.008 --truncation-rate 0.012 \
   > $S/logs/mock.log 2>&1 & MOCK=$!
sleep 1
$PY scripts/p3/azure_teacher.py instances --set 8 --out $S/teacher
$PY scripts/p3/stream_coordinator.py --pull-from-dir $D/pod/mine --local-dir $S/pulled --queue $S/stream/queue.jsonl \
   --status $S/stream/coordinator-status.json --quotas $D/pool/quotas.json --teacher-status $S/teacher/status.json \
   --pool-shards "$D/pool/shards/*.jsonl" --variants-queue $S/stream/queue-variants.jsonl --interval 3 --pull-scores-at-end \
   > $S/logs/coord.log 2>&1 & COORD=$!
$PY scripts/p3/azure_teacher.py run --deployment imajev-teacher:8 --queue $S/stream/queue.jsonl --queue $S/stream/queue-variants.jsonl \
   --out $S/teacher --base-url http://127.0.0.1:$PORT/openai/v1 --key-file $S/key --data-root data --per-instance-concurrency 4 \
   --poll-secs 2 --status-secs 5 --gate-items 200 --follow > $S/logs/teacher.log 2>&1 & TEACH=$!
$PY scripts/p3/variants.py --no-writer --results $S/teacher/results.jsonl --teacher-out $S/teacher --out $S/variants \
   --queue $S/stream/queue-variants.jsonl --pool-shards "data/p3/pool/shards/*.jsonl" --upstream-queues $S/stream/queue.jsonl \
   --follow --interval 5 --stop-when-closed --close > $S/logs/variants.log 2>&1 & VAR=$!
sleep 5
for g in 0 1; do $PY scripts/p3/mine.py --backend fake --fake-accuracy 0.65 --shards "$D/pool/shards/*.jsonl" --out $D/pod/mine \
   --worker-id gpu$g --chunk 32 > $S/logs/mine-gpu$g.log 2>&1 & done
wait $COORD; echo "coordinator exit $?"
wait $VAR; echo "variants exit $?"
wait $TEACH; echo "teacher exit $?"
kill $MOCK; wait $MOCK 2>/dev/null
$PY scripts/p3/azure_teacher.py instances --set 0 --out $S/teacher
$PY scripts/p3/azure_teacher.py spend --out $S/teacher
