"""Screen model shared by the desktop / mobile / pro templates of scripts/p3/gen_screens.py.

A template returns a Screen: the HTML, the viewport, and a registry of the labelled elements (data-el) with what the
generator needs to ask about them:
  desc      noun phrase naming the element uniquely on this screen ("the Bluetooth switch", "the bell icon")
  label     visible text, when the element has one (used as a choice option)
  goals     goal phrases that tapping / clicking this element achieves ("turn Bluetooth on")
  action    what tapping it does, as one of ACTIONS (mobile_next)
  disabled  rendered disabled (greyed, not clickable)
  small     a small target (icon-sized); pro screens mark their dense controls
Groups: named lists of element ids whose labels are unique and form a sensible option list (bottom nav, a menu ...).
absent: (desc, goal) pairs of plausible controls that this screen does NOT have (unknown / not_listed questions).
dialog: the element id of a modal card that covers part of the screen (occlusion unknowns), or None.
"""
from __future__ import annotations

import html

E = html.escape

ACTIONS = ["opens another screen", "switches to another tab", "turns a setting on", "turns a setting off",
           "sends the message", "nothing, it is disabled", "closes the dialog", "types a letter", "starts playback",
           "pauses playback", "goes back to the previous screen", "places the order", "adds one more item",
           "removes one item", "applies the promo code", "starts a call"]


class Screen:
    def __init__(self, platform: str, template: str, width: int, height: int, dpr: float = 1.0, title: str = ""):
        self.platform, self.template = platform, template
        self.width, self.height, self.dpr = width, height, dpr
        self.title = title
        self.els: dict[str, dict] = {}
        self.groups: dict[str, list[str]] = {}
        self.absent: list[tuple[str, str | None]] = []
        self.dialog: str | None = None
        self.meta: dict = {}
        self.html = ""
        self.screen_desc = ""
        self.possible: list[str] = []      # element ids whose goal is a sensible "can you ... right now" question

    def el(self, eid: str, desc: str | None, label: str | None = None, goals=None, action: str | None = None,
           disabled: bool = False, group: str | None = None, small: bool = False, role: str = "control") -> str:
        assert eid not in self.els, eid
        if desc is not None:
            assert all(e["desc"] != desc for e in self.els.values()), (self.template, desc)
        self.els[eid] = {"desc": desc, "label": label, "goals": list(goals or []), "action": action,
                         "disabled": disabled, "small": small, "role": role}
        if group:
            self.groups.setdefault(group, []).append(eid)
        return f'data-el="{eid}"'

    def add_absent(self, desc: str, goal: str | None = None):
        if all(e["desc"] != desc for e in self.els.values()):
            self.absent.append((desc, goal))

    def to_meta(self) -> dict:
        return {"platform": self.platform, "template": self.template, "viewport": [self.width, self.height], "dpr": self.dpr,
                "groups": self.groups, "dialog": self.dialog, "meta": self.meta}


def doc(css: str, body: str, bg: str, font: str, width: int, height: int) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>*{{box-sizing:border-box}}'
            f'html,body{{margin:0;width:{width}px;height:{height}px;overflow:hidden;background:{bg};font-family:{font}}}'
            f'{css}</style></head><body>{body}</body></html>')
