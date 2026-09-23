#!/usr/bin/env bash
# imajevBench v2.0-lite local-model runs + release checks on one GPU. Stages write DONE markers; reruns skip finished stages.
#   serve → direct (option scoring, torch) → gen (structured generation via vLLM) → calfold (2B logits) → pack
set -uo pipefail
cd ; . venv/bin/activate
export PYTHONPATH=src:scripts HF_HOME=hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=out; mkdir -p $O; skip(){ [ -f $O/$1.DONE ]; }; log(){ echo "$(date +%H:%M:%S) $*"; }
path(){ python -c "import json;print(json.load(open('$1'))['path'])"; }
B2=$(path artifacts/model.json); B4=$(path artifacts/model-qwen4b.json); B9=$(path artifacts/model-qwen9b.json)
GE2=$(path artifacts/model-gemma-e2b.json); GE4=$(path artifacts/model-gemma.json)
A2=adapters/imajev-2b; A9=adapters/imajev-9b
R=bench/records/records-eval.jsonl; ROOT=bench

if ! skip serve; then log "stage serve: public-repo server on CUDA + Jev API calls"
  (python scripts/playground/server.py --backend torch --adapter $A2 --calibration $A2/calibration.json --host 127.0.0.1 --port 8765 > $O/server.log 2>&1 &)
  for i in $(seq 1 60); do curl -s -m 2 http://127.0.0.1:8765/v1/models >/dev/null && break; sleep 3; done
  curl -s http://127.0.0.1:8765/v1/models | tee $O/serve-models.json; echo
  curl -s -X POST http://127.0.0.1:8765/v1/systemone -H 'content-type: application/json' -d @examples/text-ticket.json | tee $O/serve-text.json; echo
  python - <<'PY' | tee $O/serve-image.json
import base64,json,os,urllib.request
img=base64.b64encode(open('bench/assets/'+sorted(os.listdir('bench/assets'))[0],'rb').read()).decode()
body={"state":{"listing":{"colour":"blue"}},"questions":{"colour_matches":{"type":"noul","question":"Does the photo show the listed colour?"},"scene":{"type":"choice","question":"What is shown?","options":["shop counter","street","kitchen","other"]}},"images":["data:image/jpeg;base64,"+img]}
req=urllib.request.Request('http://127.0.0.1:8765/v1/systemone',data=json.dumps(body).encode(),headers={'content-type':'application/json'})
print(urllib.request.urlopen(req,timeout=120).read().decode()[:1200])
PY
  pkill -f "scripts/playground/server.py" || true; sleep 2; touch $O/serve.DONE; fi

run(){ name=$1; shift; [ -f $O/direct-$name/completion.json ] && { log "  $name done"; return; }; rm -rf $O/direct-$name
  python scripts/imajev_bench/run_local_v2.py --records $R --root $ROOT --split test --profile benchmark-neutral --rotations full --warmup 1 --repeats 1 --allow-draft --backend torch --device cuda --gpu-coordinated --output $O/direct-$name "$@" > $O/direct-$name.log 2>&1 || { tail -n 5 $O/direct-$name.log; log "FAILED direct $name"; return; }
  log "  $name: $(python -c "import json;d=json.load(open('$O/direct-$name/completion.json'));print({k:d[k] for k in list(d)[:5]})" 2>/dev/null)"; }
if ! skip direct; then log "stage direct: option scoring (torch, bf16)"
  run imajev2b-full     --model imajev2b-v21 --adapter $A2 --base-path $B2 --condition full
  run imajev2b-nostate  --model imajev2b-v21 --adapter $A2 --base-path $B2 --condition no_state
  run qwen2b-full       --model qwen2b --adapter none --base-path $B2 --condition full
  run qwen4b-full       --model qwen4b --adapter none --base-path $B4 --condition full
  run imajev9b-full     --model imajev9b --adapter $A9 --base-path $B9 --condition full
  run qwen9b-full       --model qwen9b --adapter none --base-path $B9 --condition full
  touch $O/direct.DONE; fi

gen(){ name=$1; model_path=$2; served=$3; shift 3; [ -f $O/gen-$name/completion.json ] && { log "  $name done"; return; }; rm -rf $O/gen-$name
  log "  vLLM serve $served"; (vllm-venv/bin/python -m vllm.entrypoints.openai.api_server --model $model_path --served-model-name $served --port 8000 --max-model-len 8192 --limit-mm-per-prompt '{"image":2}' --gpu-memory-utilization 0.85 "$@" > $O/vllm-$name.log 2>&1 &)
  for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:8000/v1/models | grep -q "$served" && break; sleep 5; done
  if ! curl -s -m 2 http://127.0.0.1:8000/v1/models | grep -q "$served"; then log "FAILED vllm $name (see vllm-$name.log)"; pkill -f "vllm.entrypoints" || true; sleep 5; return; fi
  OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=local python -m imajev_bench api-run --records $R --root $ROOT --split test --allow-draft --allow-test-exposure --provider openai --model $served --purpose evaluation --workers 8 --output $O/gen-$name > $O/gen-$name.log 2>&1 || { tail -n 5 $O/gen-$name.log; log "FAILED gen $name"; }
  pkill -f "vllm.entrypoints" || true; sleep 8; log "  $name: $(tail -n 1 $O/gen-$name.log | cut -c1-160)"; }
if ! skip gen; then log "stage gen: structured generation (vLLM, json_schema, default reasoning)"
  gen qwen2b   $B2  qwen3.5-2b
  gen qwen4b   $B4  qwen3.5-4b
  gen qwen9b   $B9  qwen3.5-9b
  gen gemma-e2b $GE2 gemma-4-e2b-it
  gen gemma-e4b $GE4 gemma-4-e4b-it
  touch $O/gen.DONE; fi

if ! skip calfold; then log "stage calfold: final 2B logits on the text calibration fold + MMLU + typed + SST-5"
  for spec in "decision-v2.1-calibration calibration calfold" "decision-v1.1-irrelevance-test-text test mmlu" "decision-v1.1-heldout-text test typed" "decision-v1.1-heldout-text-unlicensed test sst5"; do set -- $spec
    python scripts/evaluate_decision_model_torch.py --output $O/2b-$3 --model $B2 --adapter $A2 --version $1 --partition $2 --device cuda --workers 8 > $O/2b-$3.log 2>&1 || { tail -n 5 $O/2b-$3.log; log "FAILED $3"; continue; }
    log "  $3: $(wc -l < $O/2b-$3/predictions.jsonl) predictions"; done
  touch $O/calfold.DONE; fi

log "pack"; cd /workspace && tar czf release-test-results.tgz out && ls -la release-test-results.tgz; echo ALL_DONE
