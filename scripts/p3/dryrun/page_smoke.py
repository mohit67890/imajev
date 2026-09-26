"""Headless smoke of scripts/p3/review_page.py on port 8790: page loads, items never carry Kimi's verdict before the owner's,
images are served, every save returns Kimi's verdict, progress reaches the end. Owner verdicts are random (dry run only)."""
import json, random, sys, urllib.request
B = "http://127.0.0.1:8790"
get = lambda p: json.load(urllib.request.urlopen(B + p))
html = urllib.request.urlopen(B + "/").read().decode()
p = get("/api/progress")
rng = random.Random(1)
leak = imgs = revealed = 0
for i in range(p["total"]):
    it = get(f"/api/item?i={i}")
    leak += it.get("kimi") is not None
    for im in it.get("images") or []:
        u = im if isinstance(im, str) else (im.get("url") or im.get("src") or "")
        if u:
            imgs += urllib.request.urlopen(B + (u if u.startswith("/") else "/img?p=" + u)).status == 200
    v = rng.choices(["correct", "wrong", "ambiguous", "bad_question"], [0.8, 0.12, 0.05, 0.03])[0]
    req = urllib.request.Request(B + "/api/verdict", data=json.dumps({"id": it["id"], "verdict": v, "note": "dry run"}).encode(),
                                 headers={"content-type": "application/json"})
    revealed += json.load(urllib.request.urlopen(req)).get("kimi") is not None
end = get("/api/progress")
ok = "<title>" in html and leak == 0 and revealed == p["total"] and end["done"] == p["total"] > 0
print(f"{'PASS' if ok else 'FAIL'}  review page on :8790  ({p['total']} queued, Kimi shown before a verdict: {leak}, "
      f"images served {imgs}, Kimi revealed after save {revealed}, progress {end['done']}/{end['total']})")
sys.exit(0 if ok else 1)
