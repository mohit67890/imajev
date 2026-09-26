#!/usr/bin/env bash
# Live view of the DecisionBench pod run: reports/benchmarks/decisionbench/dash_db.sh [host] [seconds]; copies dash_db_remote.py to the pod first.
H=${1:-imajev-db2}; S=${2:-20}
scp -q "$(dirname "$0")/dash_db_remote.py" $H:dash_db_remote.py
while :; do out=$(ssh -n -o ConnectTimeout=20 $H 'python3 dash_db_remote.py' 2>&1); clear; echo "$out"; sleep $S; done
