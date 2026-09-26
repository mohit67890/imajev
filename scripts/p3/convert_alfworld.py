"""ALFWorld (MIT) train games -> phase-3 agent-action choice items (docs/phase-3-plan.md, Stage 0 source B).

Two steps, because the TextWorld engine needs its own Python (3.11, `pip install alfworld`), not the project venv:

  1. dump   (alfworld venv): play every *train* game with ALFWorld's hand-coded expert and write one JSON line per game
            with every step's observation, admissible commands and expert action ->  data/p3/raw/alfworld/expert_train.jsonl
            ALFWORLD venv:  <venv>/bin/python scripts/p3/convert_alfworld.py dump
  2. convert (project venv): keep only *decisive* steps, where the expert action is the unique right answer given what the
            agent has seen, and write data/p3/candidates/B-alfworld.jsonl.

Why only decisive steps: the hand-coded expert explores (``go to shelf 3``, ``open drawer 6``, ``look``) without knowing where
things are, so an exploration step has many equally good answers. We keep: take / move(put) / heat / cool / clean / use, and a
``go to X`` only when X is where the needed thing was already seen or X is the task's own receptacle / appliance. Options are
the admissible commands minus every command that would be equally right (another instance of the same object or receptacle
type, or a ``go to`` any other place where the needed thing was seen).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/p3/raw/alfworld"
DUMP = RAW / "expert_train.jsonl"
OUT = ROOT / "data/p3/candidates/B-alfworld.jsonl"
URL = "https://github.com/alfworld/alfworld"
LICENCE = "MIT"
MAX_STEPS = 80
HISTORY_STEPS = MAX_STEPS   # full history: a 'go to' answer can depend on something seen many steps ago

TASK_TYPES = ("pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place")


# ----------------------------------------------------------------------------------------------------------- dump (alfworld venv)
def _play(game: str) -> dict:
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType, AlfredInfos
    ri = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile", "expert_plan"])
    env_id = textworld.gym.register_game(game, ri, max_episode_steps=MAX_STEPS,
                                         wrappers=[AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert(AlfredExpertType.HANDCODED)])
    env = textworld.gym.make(env_id)
    rel = os.path.relpath(os.path.dirname(game), RAW / "json_2.1.1")
    rec = {"game": rel, "task_type": rel.split("/")[1].split("-")[0], "steps": [], "won": False}
    try:
        obs, infos = env.reset()
        rec["intro"] = obs
        for _ in range(MAX_STEPS):
            plan = infos.get("extra.expert_plan") or []
            if not plan:
                break
            a = plan[0]
            step = {"admissible": list(infos["admissible_commands"]), "action": a}
            obs, _, done, infos = env.step(a)
            step["obs_after"] = obs
            rec["steps"].append(step)
            if done:
                rec["won"] = bool(infos.get("won"))
                break
    except Exception as e:  # noqa: BLE001 - an engine error drops the game, it is logged
        rec["error"] = repr(e)[:300]
    finally:
        env.close()
    return rec


def _solvable(game: str) -> bool:
    try:
        with open(game) as fh:
            return bool(json.load(fh).get("solvable"))
    except Exception:  # noqa: BLE001
        return False


def dump(workers: int = 4, limit: int = 0) -> None:
    from multiprocessing import Pool
    games = sorted(glob.glob(str(RAW / "json_2.1.1/train/*/*/game.tw-pddl")))
    games = [g for g in games if "movable" not in g and "Sliced" not in g and _solvable(g)]
    if limit:
        games = games[:limit]
    print(f"{len(games)} solvable train games", flush=True)
    done = set()
    if DUMP.exists():   # resume: keep complete lines, replay only the missing games
        good = []
        for line in open(DUMP):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(rec["game"]); good.append(line if line.endswith("\n") else line + "\n")
        DUMP.write_text("".join(good))
    todo = [g for g in games if os.path.relpath(os.path.dirname(g), RAW / "json_2.1.1") not in done]
    print(f"{len(done)} already dumped, {len(todo)} to go", flush=True)
    n = won = 0
    # workers are recycled: TextWorld's gym registry and engine state grow with every game played in one process
    with Pool(workers, maxtasksperchild=20) as pool, open(DUMP, "a") as fh:
        for rec in pool.imap_unordered(_play, todo, chunksize=2):
            fh.write(json.dumps(rec) + "\n")
            n += 1; won += rec["won"]
            if n % 200 == 0:
                print(f"{n}/{len(todo)} won={won}", flush=True)
    print(f"done {n} games, won {won}", flush=True)


# ------------------------------------------------------------------------------------------------------ convert (project venv)
_NUM = re.compile(r"\s+\d+\b")
DECISIVE = ("take", "move", "put", "heat", "cool", "clean", "use", "slice")


def type_of(entity: str) -> str:
    """'alarmclock 2' -> 'alarmclock'."""
    return _NUM.sub("", entity).strip()


def parse_cmd(cmd: str) -> tuple[str, str | None, str | None]:
    """(verb, object, receptacle/tool) with instance numbers kept."""
    m = re.match(r"^go to (.+)$", cmd)
    if m:
        return "go", None, m.group(1)
    m = re.match(r"^take (.+?) from (.+)$", cmd)
    if m:
        return "take", m.group(1), m.group(2)
    m = re.match(r"^(?:move|put) (.+?) (?:to|in/on|in|on) (.+)$", cmd)
    if m:
        return "move", m.group(1), m.group(2)
    m = re.match(r"^(heat|cool|clean|slice) (.+?) with (.+)$", cmd)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.match(r"^use (.+)$", cmd)
    if m:
        return "use", None, m.group(1)
    m = re.match(r"^(open|close|examine) (.+)$", cmd)
    if m:
        return m.group(1), None, m.group(2)
    return cmd.split(" ")[0], None, None


def equivalent(a: str, b: str) -> bool:
    """Two commands that are equally right: same verb, same object type and same receptacle type (instance numbers ignored)."""
    va, oa, ra = parse_cmd(a)
    vb, ob, rb = parse_cmd(b)
    if va != vb:
        return False
    t = lambda x: type_of(x) if x else None  # noqa: E731
    return t(oa) == t(ob) and t(ra) == t(rb)


def seen_at(obs: str) -> tuple[str | None, set[str]]:
    """Location and the object types an observation lists there (``you see a mug 1, and a pen 2``)."""
    loc, items = None, ""
    m = re.search(r"You arrive at (.+?)\.", obs)
    if m:
        loc = m.group(1)
    m2 = re.search(r"(?:On|In) the (.+?), you see (.+?)\.(?:\s|$)", obs)
    if m2:
        loc = loc or m2.group(1)
        items = m2.group(2)
    m3 = re.search(r"You open the (.+?)\. The .+? is open\. In it, you see (.+?)\.", obs)
    if m3:
        loc = m3.group(1); items = m3.group(2)
    types = set()
    if items and items != "nothing":
        for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", items):
            part = re.sub(r"^(a|an)\s+", "", part.strip())
            if part:
                types.add(type_of(part))
    return loc, types


def task_line(intro: str) -> str:
    m = re.search(r"Your task is to: (.+)", intro)
    return m.group(1).strip() if m else ""


def room_line(intro: str) -> str:
    m = re.search(r"(You are in the middle of a room\..+?)\n", intro + "\n", re.S)
    return m.group(1).strip() if m else intro.strip()


def decisive_steps(rec: dict):
    """Yield (step index, gold command, options) for the steps whose gold is the unique right answer given the history."""
    steps = rec["steps"]
    seen: dict[str, set[str]] = {}          # location -> object types last seen there
    for i, st in enumerate(steps):
        gold, adm = st["action"], st["admissible"]
        verb, obj, where = parse_cmd(gold)
        keep, need = False, None
        if verb in ("take", "move", "heat", "cool", "clean", "use", "slice"):
            keep = gold in adm
        elif verb == "go" and i + 1 < len(steps):
            nverb, nobj, nwhere = parse_cmd(steps[i + 1]["action"])
            if nverb == "take" and nwhere == where:
                need = type_of(nobj)
                keep = need in seen.get(where, set())         # the object was already seen there
            elif nverb == "use" and nwhere:
                need = type_of(nwhere)
                keep = need in seen.get(where, set())         # the lamp was already seen there
            elif nverb in ("move", "heat", "cool", "clean") and nwhere == where:
                keep = True                                   # the task names the receptacle / appliance type
        if keep:
            opts = [c for c in adm if c == gold or not equivalent(c, gold)]
            if verb == "take" and rec["task_type"] == "look_at_obj_in_light":
                # turning the lamp on first can also work in this task, so it is not a wrong option
                opts = [c for c in opts if c == gold or not c.startswith("use ")]
            if verb == "go" and need:
                # going to any other place where the needed thing was seen is equally right
                opts = [c for c in opts if c == gold or not (c.startswith("go to ") and need in seen.get(c[6:], set()))]
            if gold in opts and 2 <= len(opts) <= 254:
                yield i, gold, opts
        loc, types = seen_at(st["obs_after"])
        if loc and (types or "you see nothing" in st["obs_after"]):
            seen[loc] = types
        if verb == "take" and where in seen and obj and not _count_left(st, obj):
            seen[where] = seen[where] - {type_of(obj)}


def _count_left(st: dict, obj: str) -> bool:
    """After taking obj from a place, is another object of the same type still there (visible in the admissible takes)?"""
    return any(parse_cmd(c)[0] == "take" and type_of(parse_cmd(c)[1] or "") == type_of(obj) and parse_cmd(c)[1] != obj
               for c in st["admissible"])


def render_state(rec: dict, upto: int) -> str:
    task = task_line(rec["intro"])
    lines = ["Household task (text environment).", f"Task: {task}", "", "Start: " + room_line(rec["intro"]), ""]
    hist = rec["steps"][:upto]
    start = max(0, len(hist) - HISTORY_STEPS)
    if start:
        lines.append(f"(first {start} steps omitted)")
    lines.append("History:" if hist else "History: none yet.")
    for j in range(start, len(hist)):
        lines.append(f"{j + 1}. > {hist[j]['action']}")
        lines.append(f"   {hist[j]['obs_after'].strip()}")
    return "\n".join(lines)


def difficulty(rec: dict, i: int, n_opts: int, verb: str) -> int:
    d = 3
    if rec["task_type"] in ("pick_two_obj_and_place", "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
                            "pick_clean_then_place_in_recep"):
        d += 1
    if verb == "go":
        d += 1          # the answer depends on remembering where something was seen
    if i >= 15 or n_opts >= 45:
        d += 1
    return min(5, d)


def convert(dump_path: Path = DUMP, out: Path = OUT, cap: int = 40000) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from candidate import write
    from convert_common import option_key, stable_shuffle
    rows = []
    for line in open(dump_path):
        rec = json.loads(line)
        if not rec.get("won") or rec.get("error"):
            continue
        for i, gold, opts in decisive_steps(rec):
            opts = stable_shuffle(opts, f"{rec['game']}#{i}")
            taken: set[str] = set()
            keyed = [(option_key(c, taken), c) for c in opts]
            gkey = next(k for k, c in keyed if c == gold)
            verb = parse_cmd(gold)[0]
            rid = f"p3-alfworld-{len(rows):06d}"
            rows.append({
                "id": rid, "source": "B", "dataset": "alfworld", "family": "agent_action_choice",
                "difficulty": difficulty(rec, i, len(opts), verb), "state": render_state(rec, i), "images": [],
                "field": {"type": "choice", "question": "Which command should the agent issue next to make progress on the task?",
                          "options": [{"key": k, "text": c, "description": ""} for k, c in keyed]},
                "gold": gkey, "unknown_reason": None, "gold_kind": "dataset", "parent_id": None,
                "provenance": {"licence": LICENCE, "upstream_dataset": "alfworld", "upstream_split": "train",
                               "upstream_id": rec["game"], "step": i, "group_id": rec["game"], "url": URL,
                               "task_type": rec["task_type"], "expert": "alfworld handcoded expert (TextWorld)",
                               "decisive_verb": verb}})
    # stable ids independent of dump order
    rows.sort(key=lambda r: (r["provenance"]["upstream_id"], r["provenance"]["step"]))
    for n, r in enumerate(rows[:cap]):
        r["id"] = f"p3-alfworld-{n:06d}"
    n = write(out, rows[:cap])
    return {"written": n}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dump", "convert"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "dump":
        dump(a.workers, a.limit)
    else:
        print(convert())
