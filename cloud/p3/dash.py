#!/usr/bin/env python3
"""Phase-3 terminal dashboard, runs ON the pod: python3 dash.py [--usd-per-hour 27.92]. Read-only."""
import glob, json, os, re, subprocess, sys, time
from datetime import datetime, timedelta, timezone
IST = timezone(timedelta(hours=5, minutes=30)); O = "p3/run"
USD = float(sys.argv[sys.argv.index("--usd-per-hour") + 1]) if "--usd-per-hour" in sys.argv else 27.92
B = "\033[1m"; D = "\033[2m"; G = "\033[32m"; Y = "\033[33m"; R = "\033[31m"; C = "\033[36m"; Z = "\033[0m"
def ist(ts=None): return datetime.fromtimestamp(ts or time.time(), IST).strftime("%H:%M:%S")
def jl(p):
    try: return [json.loads(l) for l in open(p) if l.strip()]
    except FileNotFoundError: return []
def jread(p):
    try: return json.load(open(p))
    except Exception: return None
start = os.path.getmtime("p3.env") if os.path.exists("p3.env") else time.time()
el = time.time() - start
print(f"{B}imajev-4b phase 3{Z}  {ist()} IST   pod up {el/3600:.2f} h   ~${el/3600*USD:.0f} at ${USD}/h")
# stages
lines = open(f"{O}/run.log").read().splitlines() if os.path.exists(f"{O}/run.log") else []
stages = [(l[:12], l.split("stage ", 1)[1]) for l in lines if " IST stage " in l]
done = {os.path.basename(p)[:-5] for p in glob.glob(f"{O}/*.DONE")}
failed = os.path.exists(f"{O}/FAILED") or any("FAILED" in l for l in lines[-3:])
alldone = lines and lines[-1].strip() == "ALL_DONE"
print(f"{B}stages{Z}  " + "  ".join(f"{G}✓{Z}{s}" if s in done else f"{D}·{s}{Z}" for s in
      ["checks", "prep", "parity", "pilot", "train-r1", "sel-r1", "round2", "eval", "pick"]) +
      (f"  {G}{B}ALL_DONE{Z}" if alldone else f"  {R}{B}FAILED{Z}" if failed else ""))
cur = [x for x in stages if not (x[1].startswith("parity") and "parity" in done)]
if cur: print(f"        now: {C}{cur[-1][1]}{Z} (since {cur[-1][0]})")
rc = open(f"{O}/parity/rc").read().strip() if os.path.exists(f"{O}/parity/rc") else None
if rc is not None: print(f"        parity: {G+'PASS' if rc == '0' else R+'FAIL rc '+rc}{Z}")
mi = jread(f"{O}/manifests.json") or {}
if mi: print(f"        data: {mi.get('rows'):,} rows {mi.get('partitions')}, 255-option train rows {mi.get('rows_255')}, ordinal dev {mi.get('ordinal_dev_rows')}")
# lanes / runs
def show(name, logp, cfg="", max_steps=None):
    rows = jl(logp)
    if not rows: return
    steps = [r for r in rows if "of" in r]; devs = [r for r in rows if "dev_loss" in r]
    last = steps[-1] if steps else {}; of = last.get("of", 0); st = last.get("step", 0)
    if max_steps: of = min(of, max_steps)   # a pilot lane is the full run's first max_steps steps (its log says "of <full run>")
    recent = steps[-10:]; loss = sum(r["loss"] for r in recent) / max(1, len(recent)) if recent else float("nan")
    secs = sum(r["seconds"] for r in steps[-20:]) / max(1, len(steps[-20:])) if steps else 0
    eta = (of - st) * secs if of else 0
    bar = int(30 * st / of) if of else 0
    line = f"  {B}{name:<9}{Z}{D}{cfg:<26}{Z} [{'█'*bar}{'░'*(30-bar)}] {st:>5}/{of:<5} loss {loss:5.3f}  {secs:4.1f}s/step  eta {int(eta//3600)}h{int(eta%3600//60):02d}m"
    if devs:
        d = devs[-1]; sl = d.get("dev_slices", {}); im = sl.get("image", {}).get("accuracy"); tx = sl.get("text", {}).get("accuracy")
        line += f"\n  {'':<9}{D}dev@{d['step']}:{Z} acc {d['dev_accuracy']*100:5.1f}  loss {d['dev_loss']:.3f}"
        if im is not None: line += f"  img {im*100:4.1f} txt {tx*100:4.1f}"
        if "dev2_accuracy" in d: line += f"  ordinal {d['dev2_accuracy']*100:4.1f}"
        if len(devs) > 1: line += f"   {D}(step0: acc {devs[0]['dev_accuracy']*100:.1f} loss {devs[0]['dev_loss']:.3f}){Z}"
    print(line)
lanes = [("baseline", "W0 · 255 · r16"), ("ordinal", "W0.15 · 255 · r16"), ("rank64", "W0 · 255 · r16→64"), ("codes256", "W0 · 256 · r16")]
if any(os.path.exists(f"{O}/pilot/{l}/train/log.jsonl") for l, _ in lanes):
    print(f"{B}pilot{Z}  (100 steps × 8 micro-batches per lane, 2 GPUs each; metrics = dev at 75 & 100)")
    for l, cfg in lanes:
        mark = f"{G}done{Z}" if os.path.exists(f"{O}/pilot-{l}.DONE") else (f"{R}FAILED{Z}" if os.path.exists(f"{O}/pilot-{l}.FAILED") else "")
        show(l, f"{O}/pilot/{l}/train/log.jsonl", cfg + ("  " + mark if mark else ""), max_steps=100)
pd = jread(f"{O}/pilot-decision.json")
if pd: print(f"  {B}decision:{Z} ordinal W {pd.get('ordinal_weight')}  codes {pd.get('readout_codes')}  rank {pd.get('expand_lora_rank') or 16}  lane→r1 {pd.get('lane') or 'COMBINED (restart)'}")
lr = jread(f"{O}/lane-resume.json")
if lr: print(f"  {D}{lr.get('summary','')}{Z}")
for tag in ("r1", "r2"):
    if os.path.exists(f"{O}/{tag}/train/log.jsonl"):
        print(f"{B}train {tag}{Z}"); show(tag, f"{O}/{tag}/train/log.jsonl", "all 8 GPUs")
ck = sorted(os.path.basename(p) for p in glob.glob(f"{O}/ckpts/*"))
if ck: print(f"  {D}snapshots:{Z} " + " ".join(ck))
for f, lab in (("r1-pick.json", "r1 pick"), ("pick.json", "final pick")):
    j = jread(f"{O}/{f}")
    if j: print(f"  {B}{lab}:{Z} {j.get('pick')}  {D}{j.get('reason','')[:120]}{Z}")
def pacc(d, key):
    """accuracy from a panel's predictions.jsonl (live, even mid-panel); '' when not started."""
    f = f"{d}/panels/{key}/predictions.jsonl"
    if not os.path.exists(f): return None, 0
    c = n = 0
    for l in open(f):
        if l.strip():
            n += 1; c += bool(json.loads(l).get("correct"))
    return (100.0 * c / n if n else None), n
def fmt(v, w=6, dec=1):
    return f"{v:>{w}.{dec}f}" if isinstance(v, (int, float)) else f"{'·':>{w}}"
ev = sorted(os.path.basename(e[:-1]) for e in glob.glob(f"{O}/eval/*/"))
order = ["shipped"] + [n for n in ev if n.startswith("r1")] + [n for n in ev if n.startswith("r2")] + [n for n in ev if n.startswith("soup")]
ev = [n for n in order if n in ev]
if ev:
    P = [("heldout_fresh", "fresh"), ("heldout_flagged", "flagd"), ("human_dev", "human"), ("heldout_charts", "chart"), ("heldout_docimg", "docim"),
         ("heldout_inventory", "invnt"), ("heldout_safety", "safty"), ("heldout_geometry", "geom"), ("heldout_screens", "scrns"),
         ("p2b_test", "p2b"), ("state_probe", "state"), ("pairs_probe", "pairs"), ("irrelevance", "irrel")]
    print(f"{B}eval · held-out panels{Z} (acc %, live from predictions; · = not started)")
    print(f"  {'ckpt':<11}" + "".join(f"{h:>6}" for _, h in P) + f"  {D}done{Z}")
    for n in ev:
        d = f"{O}/eval/{n}"; st = jread(f"{d}/status.json") or {}
        cells = []
        for key, _ in P:
            v, cnt = pacc(d, key); ok = st.get(key, {}).get("ok")
            cells.append((f"{G}" if ok else f"{Y}" if cnt else "") + fmt(v) + (Z if cnt else ""))
        ndone = sum(1 for v in st.values() if isinstance(v, dict) and v.get("ok"))
        now = ""
        try:
            last = open(f"{O}/eval/{n}.all.log").read().rstrip("\n").rsplit("\n", 1)[-1]
            now = re.sub(r"^\d\d:\d\d:\d\d\s+", "", last).strip()[:60]
            if last.strip().startswith("done") or "SHIPPABLE" in last: now = last.strip()[9:60]
        except Exception: pass
        print(f"  {B}{n:<11}{Z}" + "".join(cells) + f"  {D}{ndone:>2}  last: {now}{Z}")
    gr = jread("cloud/p2c_gates_reference.json") or {}; R4 = gr.get("sizes", {}).get("4b", {}); TL = gr.get("tolerances", {})
    bounds = (f"gates (4b): IJB ≥ {R4.get('imajevbench_acc',0)-TL.get('imajevbench_acc_points',0):.1f}  vis ≥ {R4.get('visual',[0])[0]-TL.get('track_items',0)}  "
              f"joint ≥ {R4.get('joint',[0])[0]-TL.get('track_items',0)}  state ≥ {R4.get('state_probe',0)-TL.get('probe_points',0):.1f}  pairs ≥ {R4.get('pairs_probe',0)-TL.get('probe_points',0):.1f}  "
              f"irrel ≥ {R4.get('irrelevance',0)-TL.get('irrelevance_points',0):.1f}  unknown-correct ≥ {R4.get('unknown_correct_rate')}  false-abstain ≤ {R4.get('false_abstention_rate',0)+TL.get('false_abstention_points',0):.2f}") if R4 else ""
    print(f"{B}eval · benchmarks & gates{Z} (JevBench single raw acc; DecisionBench 3k tracking only)  {D}{bounds}{Z}")
    print(f"  {'ckpt':<11}{'IJB':>7}{'vis':>5}{'joint':>6} {'jev-easy':>9}{'orig':>6}{'hard':>6} {'DB-acc':>7}{'ord':>6}{'reas':>6}  {'gates':<8} fail / calibration")
    for n in ev:
        d = f"{O}/eval/{n}"; sm = jread(f"{d}/summary.json") or {}; gt = jread(f"{d}/gates.json")
        tr = sm.get("imajevbench")
        if tr is None:
            for f in [f"{d}/imajevbench/score", f"{d}/imajevbench/score.json"] + glob.glob(f"{d}/imajevbench/score/*.json"):
                j = jread(f) if os.path.isfile(f) else None
                if j and "capability" in j:
                    tr = {t: [sum(x["correct"] for x in v["families"].values()), sum(x["total"] for x in v["families"].values())] for t, v in j["capability"]["tracks"].items()}; break
        ijb = (100.0 * sum(v[0] for v in tr.values()) / sum(v[1] for v in tr.values())) if tr else None
        vis = tr.get("visual", [None])[0] if tr else None; jnt = tr.get("joint", [None])[0] if tr else None
        jb = {}
        for tier in ("easy", "original", "hard"):
            j = jread(f"{d}/jevbench/single-raw/{tier}/summary.json")
            if j: jb[tier] = 100 * j["accuracy"]
        db = jread(f"p3/tracking-only/{n}/decisionbench.json") or {}
        if gt is None: gs = f"{D}pending{Z} "; fails = ""
        else:
            gs = f"{G}{B}SHIP{Z}    " if gt.get("shippable") else f"{R}{B}FAIL{Z}    "
            fails = " ".join(c["gate"].split(".", 1)[1] + f"({c['value']}<{c['bound']})" if c.get("op") == ">=" else c["gate"].split(".", 1)[1] + f"({c['value']}>{c['bound']})" for c in gt["checks"] if not c["pass"])
        cal = (sm.get("calibration") or {}).get("single", {}).get("choice", "")
        print(f"  {B}{n:<11}{Z}{fmt(ijb, 7)}{fmt(vis, 5, 0)}{fmt(jnt, 6, 0)} {fmt(jb.get('easy'), 9)}{fmt(jb.get('original'), 6)}{fmt(jb.get('hard'), 6)} "
              f"{fmt(db.get('primary_accuracy'), 7)}{fmt(db.get('ordinal'), 6)}{fmt(db.get('reasoning'), 6)}  {gs}{R if fails else ''}{fails}{Z} {D}{cal}{Z}")
# gpus
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
    print(f"{B}gpus{Z}   " + " ".join(f"{i}:{int(u):>3}%/{int(m)//1024}G" for i, (u, m, t) in enumerate(x.split(", ") for x in q)))
except Exception: pass
du = subprocess.run(["df", "-h", "/workspace"], capture_output=True, text=True).stdout.splitlines()[-1].split()
print(f"{B}disk{Z}   {du[2]} used of {du[1]}")
tail = [l for l in lines if not re.search(r"^\s*$", l)][-4:]
print(f"{B}log{Z}    " + f"\n       ".join(l[:150] for l in tail))
