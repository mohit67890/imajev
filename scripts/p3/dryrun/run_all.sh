#!/usr/bin/env bash
# Phase-3 end-to-end dry run (reports/phase3/dry-run.md). No real Azure, no GPU, no uploads, at most ONE local model process
# (the optional MLX smoke, MLX=1). Everything lands in data/p3/dryrun/ and the scratch dir $OUT; the dry-run manifests are
# published as data/manifests/decision-p3dry* and removed at the end.
set -u; cd "$(dirname "$0")/../../.."
PY=.venv/bin/python; D=data/p3/dryrun; S=$D/session; R=$D/review; OUT=${OUT:-$D/bundles}; mkdir -p $D/logs
step(){ echo; echo "######## $(TZ=Asia/Kolkata date +%H:%M:%S) IST  $*"; }
step "1 sample pool";   rm -rf $D/pool $D/pod $D/mine-mlx $D/gate $R $D/manifests $D/r2; $PY scripts/p3/dryrun/make_sample.py
step "2 MLX smoke (one model process, 24 items)"
if [ "${MLX:-1}" = 1 ]; then $PY scripts/p3/mine.py --backend mlx --shards "$D/pool/shards/*.jsonl" --out $D/mine-mlx --limit 8 \
   --worker-id mlx0 --chunk 8 > $D/logs/mine-mlx.log 2>&1; tail -3 $D/logs/mine-mlx.log; $PY scripts/p3/dryrun/checks.py mlx; fi
step "3 streamed session: mock Azure x8, coordinator (local rsync), teacher + pilot gate, variants (no writer), fake mining x2"
bash scripts/p3/dryrun/session.sh; $PY scripts/p3/azure_teacher.py finalize --out $S/teacher > $D/logs/finalize.json
$PY scripts/p3/dryrun/checks.py mine session
step "4 pilot gate pause / resume (6% parse errors)"; bash scripts/p3/dryrun/gate_pause.sh; $PY scripts/p3/dryrun/checks.py gate
step "5 take-flagged (flagged held-out half)"
$PY scripts/p3/assemble_pool.py take-flagged --pool-dir $D/pool --mined $S/pulled/*.flagged.jsonl > $D/logs/take-flagged.log 2>&1
tail -1 $D/logs/take-flagged.log; $PY scripts/p3/dryrun/checks.py flagged
step "6 Kimi review against the mock (sample 150, review 100 then resume)"
mkdir -p $R
$PY scripts/p3/kimi_review.py sample --teacher $S/teacher/kept.jsonl $S/variants/constructed.jsonl --out $R/sample.jsonl --n 150 --n-image 30
$PY scripts/p3/mock_azure.py --port 8913 --key-file $S/key --policy realistic --gold-from "$D/pool/shards/*.jsonl" \
   --gold-from "$S/stream/queue*.jsonl" --gold-from "$S/variants/constructed.jsonl" --latency 0.2 --accuracy 0.9 > $R/mock.log 2>&1 & MOCK=$!
sleep 1
for lim in 100 0; do $PY scripts/p3/kimi_review.py review --sample $R/sample.jsonl --out $R/kimi.jsonl \
   --url http://127.0.0.1:8913/openai/v1/chat/completions --key-file $S/key --max-usd 3 --workers 8 --limit $lim | tail -1; done
kill $MOCK; wait $MOCK 2>/dev/null
step "7 review page smoke on :8790 (headless)"
$PY scripts/p3/review_page.py --sample $R/sample.jsonl --kimi $R/kimi.jsonl --queue $R/owner-queue.jsonl --owner $R/owner.jsonl \
   --n-disagree 12 --n-random 8 > $R/page.log 2>&1 & PAGE=$!
for i in $(seq 1 40); do curl -s -o /dev/null http://127.0.0.1:8790/api/progress && break; sleep 0.25; done
$PY scripts/p3/dryrun/page_smoke.py; kill $PAGE; wait $PAGE 2>/dev/null
lsof -iTCP:8790 -sTCP:LISTEN > /dev/null && echo "FAIL  page still listening" || echo "PASS  review page stopped"
step "8 review report"
$PY scripts/p3/review_report.py --sample $R/sample.jsonl --kimi $R/kimi.jsonl --queue $R/owner-queue.jsonl --owner $R/owner.jsonl \
   --report $R/review.md --dev-out $R/dev-slice.jsonl
$PY scripts/p3/dryrun/checks.py review
step "9 build_manifest, both lanes, publish decision-p3dry*"
$PY scripts/p3/build_manifest.py --verified $S/teacher/kept.jsonl $S/variants/constructed.jsonl --review-dev $R/dev-slice.jsonl \
   --drop-ids $R/drop-ids.json --heldout-results $S/teacher/heldout-results.jsonl --pool-dir $D/pool --out-dir $D/manifests \
   --publish decision-p3dry > $D/logs/build_manifest.log 2>&1; echo "build_manifest rc $?"
grep -E "verified:|mixture|gate lane|OK|FAILED" $D/logs/build_manifest.log; $PY scripts/p3/dryrun/checks.py manifest
step "10 pod-side CPU steps on the published names: stage-0 checks, prep_manifests, round 2"
awk '/python - \$V \$FRESH \$FLAGGED \$HUMAN <<.PY./{f=1;next} /^PY$/{if(f){f=0;exit}} f' cloud/p3/pod_run_train.sh > $D/logs/pod_check.py
$PY $D/logs/pod_check.py decision-p3dry decision-p3dry-heldout-fresh decision-p3dry-heldout-flagged decision-p3dry-human-dev \
   && echo "PASS  pod stage-0 manifest check" || echo "FAIL  pod stage-0 manifest check"
$PY cloud/p3/prep_manifests.py --version decision-p3dry --out $D/logs/manifests.json > /dev/null && cat $D/logs/manifests.json
mkdir -p $D/r2/pool; awk '/python - \$SOURCE "\$MINE_SHARDS" \$O\/r2\/pool/{f=1;next} /^PY$/{if(f){f=0;exit}} f' cloud/p3/pod_run_train.sh > $D/r2/split.py
$PY $D/r2/split.py decision-p3dry-labelled "$D/pool/shards/*.jsonl" $D/r2/pool 8
$PY scripts/p3/mine.py --backend fake --fake-accuracy 0.7 --shards "$D/r2/pool/*.jsonl" --out $D/r2/mine --worker-id r2-gpu0 > $D/r2/mine.log 2>&1
$PY cloud/p3/round2_manifest.py --mined "$D/r2/mine/*.flagged.jsonl" --source decision-p3dry-labelled --base decision-p3dry-r255 \
   --exclude decision-p3dry-heldout-fresh,decision-p3dry-heldout-flagged,decision-p3dry-human-dev --out decision-p3dry-r2 --codes 255 \
   --min-new 20 && echo "PASS  round-2 manifest" || echo "FAIL  round-2 manifest"
step "11 transfer.py: training bundle (benchmark check), local verify+extract, negative control, gatekit"
rm -rf $OUT; mkdir -p $OUT
$PY scripts/p3/transfer.py build --preset train --name p3dry-train --out-dir $OUT \
  --manifest decision-p3dry --manifest decision-p3dry-lane255 --manifest decision-p3dry-labelled --manifest decision-p3dry-heldout-fresh \
  --manifest decision-p3dry-heldout-fresh-large-choice --manifest decision-p3dry-heldout-flagged --manifest decision-p3dry-human-dev \
  --manifest decision-p2b --manifest decision-v2-state-probe --manifest decision-v2-pairs-probe --manifest decision-v1.1-irrelevance-test \
  --candidates "$D/pool/shards/*.jsonl" 2>&1 | tail -1
$PY - "$OUT" <<'PY'
import sys, shutil
sys.path.insert(0, "scripts/p3")
import transfer as T
from pathlib import Path
out = Path(sys.argv[1]); ext = out / "extract"; shutil.rmtree(ext, ignore_errors=True)
m = T.verify_and_extract("p3dry-train", out, ext)
ok = m["check"]["status"] == "PASS" and (ext / "data/manifests/decision-p3dry.jsonl").exists() and (ext / "scripts/p3/mine.py").exists()
print(f"{'PASS' if ok else 'FAIL'}  training bundle verify + extract ({len(m['files'])} files, {m['archive_bytes'] / 1e9:.2f} GB)")
shutil.rmtree(ext, ignore_errors=True)
PY
$PY scripts/p3/transfer.py build --preset train --name p3dry-neg --out-dir $OUT --manifest decision-p3dry-human-dev \
   --include data/manifests/decision-p2b-jevstyle-dev.jsonl --list > /dev/null 2>&1 \
   && echo "FAIL  negative control built" || echo "PASS  negative control: authored JevBench-style dev refused in a training bundle (exit 2)"
$PY scripts/p3/transfer.py build --preset gatekit --name p3dry-gatekit --out-dir $OUT 2>&1 | tail -1
$PY - "$OUT" <<'PY'
import sys
sys.path.insert(0, "scripts/p3")
import transfer as T
from pathlib import Path
try:
    T.verify_and_extract("p3dry-gatekit", Path(sys.argv[1]), Path(sys.argv[1]) / "x"); print("FAIL  gatekit extracted without --gatekit")
except T.BundleError:
    print("PASS  gatekit refuses extraction without --gatekit")
PY
step "12 cleanup of the published dry-run manifests"
rm -f data/manifests/decision-p3dry*.jsonl data/manifests/decision-p3dry*-images.json; echo "left in data/manifests: $(ls data/manifests | grep -c p3dry || true)"
