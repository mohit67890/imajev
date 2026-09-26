#!/usr/bin/env bash
# Phase-3 terminal dashboard from the Mac: cloud/p3/dash.sh [-w [seconds]] [host]. -w refreshes (default 60 s). Read-only.
H=imajev-p3; W=0; S=60
while [ $# -gt 0 ]; do case $1 in -w) W=1; [[ ${2:-} =~ ^[0-9]+$ ]] && { S=$2; shift; } ;; *) H=$1 ;; esac; shift; done
scp -q "$(dirname "$0")/dash.py" $H:dash.py
while :; do out=$(ssh -n -o ConnectTimeout=20 $H 'python3 dash.py --usd-per-hour ${POD_USD_PER_HOUR:-27.92}' 2>&1); [ $W = 1 ] && clear; echo "$out"; [ $W = 1 ] || break; sleep $S; done
