"""Phase-3 pod: the rolling eval lane (EVAL_DURING_TRAIN=1 in cloud/p3/pod_run_train.sh; OFF by default, see the runbook).

One GPU is kept out of training and evaluates checkpoints (scripts/p3/eval_checkpoint.py --panels all) as they appear, so the
final eval stage only covers what is left. State lives in one directory (<O>/lane):
  claims/<name>/   the lane took <name> (mkdir, atomic, under lock)      <name>.ok / <name>.failed   its outcome
  PAUSE            no new claims (the r1 selection stage runs)          STOP   no new claims; exit when the current one ends
  lock, run.lock   claim lock; one lane process at a time
Eligible, in this order: `shipped` (the baseline, first), then r1-/r2- snapshots by (round, step). The LAST snapshot of a round
(step >= of) is left to the pipeline: r1's is needed right away by the round-1 selection (eligible again once that has finished,
--final-ok r1=<O>/sel-r1.DONE), r2's only exists when training ends. Soups are never eligible (the fallback runs after the pick).
`ok` means the eval exited 0, summary.json exists and no panel is marked failed in status.json; anything else is `failed` and the
final queue evaluates it again (cached panels are reused).

    python cloud/p3/eval_lane.py run --state <O>/lane --ckpts <O>/ckpts --eval-root <O>/eval --gpu 7 --shipped <adapter> \
        --final-ok r1=<O>/sel-r1.DONE --parent-pid $$ --eval-args "--model ... --tracking-root ..."
    python cloud/p3/eval_lane.py pause|resume|stop --state <O>/lane           (pause/stop print the checkpoint being evaluated)
    python cloud/p3/eval_lane.py left --state <O>/lane --names a,b,c [--lane-exited]   (not ok and not running: for the final queue)
    python cloud/p3/eval_lane.py status --state <O>/lane --name a
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIXES = ("r1-", "r2-")


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime(time.time() + 5.5 * 3600)), "IST lane:", *a, flush=True)


@contextmanager
def locked(path: Path, blocking: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def meta(name: str, ckpts: Path) -> dict | None:
    """{'round', 'step', 'of', 'order'} of a finished snapshot, as pod_run_train.sh ckpt_meta computes them; None if not ready."""
    if name == "shipped":
        return {"round": 0, "step": 0, "of": 0, "order": 0.0}
    d = ckpts / name
    if not name.startswith(PREFIXES) or name.endswith(".tmp") or not (d / "ckpt.json").exists() or not (d / "adapter_model.safetensors").exists():
        return None
    m = json.loads((d / "ckpt.json").read_text())
    rnd = 2 if name.startswith("r2") else 1
    step, of = m.get("step") or 0, m.get("of") or 1
    return {"round": rnd, "step": step, "of": of, "order": rnd + step / max(1, of)}


def eligible(ckpts: Path, final_ok: set[int], shipped: bool = True) -> list[str]:
    names = []
    if ckpts.is_dir():
        for d in ckpts.iterdir():
            m = meta(d.name, ckpts)
            if m and (m["step"] < m["of"] or m["round"] in final_ok):
                names.append((m["round"], m["step"], d.name))
    return (["shipped"] if shipped else []) + [n for _, _, n in sorted(names)]


def status(state: Path, name: str) -> str:
    if (state / f"{name}.ok").exists():
        return "ok"
    if (state / f"{name}.failed").exists():
        return "failed"
    return "running" if (state / "claims" / name).is_dir() else "unclaimed"


def running(state: Path) -> list[str]:
    d = state / "claims"
    return sorted(p.name for p in d.iterdir() if status(state, p.name) == "running") if d.is_dir() else []


def claim_next(state: Path, ckpts: Path, final_ok: set[int], shipped: bool = True) -> str | None:
    with locked(state / "lock"):
        if (state / "PAUSE").exists() or (state / "STOP").exists():
            return None
        (state / "claims").mkdir(parents=True, exist_ok=True)
        for n in eligible(ckpts, final_ok, shipped):
            try:
                (state / "claims" / n).mkdir()
                return n
            except FileExistsError:
                continue
    return None


def set_flag(state: Path, flag: str, on: bool = True) -> list[str]:
    """PAUSE / STOP under the claim lock, so no claim can slip in after it; returns what the lane is evaluating now."""
    with locked(state / "lock"):
        p = state / flag
        if on:
            p.touch()
        elif p.exists():
            p.unlink()
        return running(state)


def reset(state: Path):
    """At lane start: forget claims of a previous (dead) lane that did not end ok; ok results stay."""
    for n in os.listdir(state / "claims") if (state / "claims").is_dir() else []:
        if status(state, n) != "ok":
            os.rmdir(state / "claims" / n)
            (state / f"{n}.failed").unlink(missing_ok=True)
    for f in ("PAUSE", "STOP"):
        (state / f).unlink(missing_ok=True)


def eval_ok(out: Path) -> bool:
    if not (out / "summary.json").exists():
        return False
    st = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    return all(v.get("ok", True) for v in st.values())


def left(state: Path, names: list[str], lane_exited: bool = False) -> list[str]:
    """Names the final queue must evaluate: not ok, and not being evaluated by a live lane (after the lane has exited, a claim
    without an outcome is a lane that died mid-way: evaluate it again)."""
    busy = ("ok",) if lane_exited else ("ok", "running")
    return [n for n in names if status(state, n) not in busy]


def eval_command(a, name: str) -> list[str]:
    m = meta(name, a.ckpts)
    ckpt = a.shipped if name == "shipped" else str(a.ckpts / name)
    cmd = shlex.split(a.eval_cmd) + ["--ckpt", ckpt, "--name", name, "--out", str(a.eval_root / name), "--gpu", str(a.gpu),
                                     "--port", str(8765 + int(a.gpu)), "--panels", "all", "--round", str(m["round"]),
                                     "--step", str(m["step"]), "--order", str(m["order"])] + shlex.split(a.eval_args)
    if name == "shipped":
        cmd += ["--fixed-calibration", str(Path(a.shipped) / "calibration.json")]
    return cmd


def parent_alive(pid: int) -> bool:
    if not pid:
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def final_rounds(specs: list[str]) -> set[int]:
    """--final-ok r1=<marker>: round 1's last snapshot becomes eligible once <marker> exists."""
    out = set()
    for s in specs:
        r, _, marker = s.partition("=")
        if Path(marker).exists():
            out.add(int(r.lstrip("r")))
    return out


def run(a) -> int:
    a.state.mkdir(parents=True, exist_ok=True)
    with locked(a.state / "run.lock"):          # one lane at a time: a lane left over from a failed run finishes first
        reset(a.state)
        log(f"started on GPU {a.gpu}")
        while True:
            if not parent_alive(a.parent_pid):
                log("the pod script is gone: stop"); return 0
            name = claim_next(a.state, a.ckpts, final_rounds(a.final_ok), shipped=not a.no_shipped)
            if name is None:
                if (a.state / "STOP").exists():
                    log("STOP: exit"); return 0
                time.sleep(a.poll); continue
            out = a.eval_root / name; out.mkdir(parents=True, exist_ok=True)
            log(f"eval {name} on GPU {a.gpu}")
            t = time.monotonic()
            with open(a.eval_root / f"{name}.all.log", "a") as f:
                rc = subprocess.run(eval_command(a, name), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
            ok = rc == 0 and eval_ok(out)
            (a.state / f"{name}.{'ok' if ok else 'failed'}").write_text(json.dumps({"rc": rc, "minutes": round((time.monotonic() - t) / 60, 1)}) + "\n")
            log(f"{'done' if ok else 'EVAL_FAILED'} {name} ({(time.monotonic() - t) / 60:.0f} min, rc {rc})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--state", type=Path, required=True); r.add_argument("--ckpts", type=Path, required=True)
    r.add_argument("--eval-root", type=Path, required=True); r.add_argument("--gpu", required=True)
    r.add_argument("--shipped", required=True, help="the shipped adapter (evaluated first, with its own calibration.json)")
    r.add_argument("--no-shipped", action="store_true"); r.add_argument("--final-ok", action="append", default=[])
    r.add_argument("--parent-pid", type=int, default=0); r.add_argument("--poll", type=float, default=60.0)
    r.add_argument("--eval-args", default="", help="the arguments every eval_checkpoint.py call shares (model, held-out specs, roots)")
    r.add_argument("--eval-cmd", default=f"{shlex.quote(sys.executable)} scripts/p3/eval_checkpoint.py")
    for c in ("pause", "resume", "stop"):
        sub.add_parser(c).add_argument("--state", type=Path, required=True)
    lf = sub.add_parser("left"); lf.add_argument("--state", type=Path, required=True); lf.add_argument("--names", default="")
    lf.add_argument("--lane-exited", action="store_true", help="the lane process has ended: unfinished claims count as not done")
    st = sub.add_parser("status"); st.add_argument("--state", type=Path, required=True); st.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    a.state.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("pause", "stop"):
        print(" ".join(set_flag(a.state, a.cmd.upper())))
    elif a.cmd == "resume":
        set_flag(a.state, "PAUSE", on=False)
    elif a.cmd == "left":
        print(" ".join(left(a.state, [n for n in a.names.split(",") if n], a.lane_exited)))
    else:
        print(status(a.state, a.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
