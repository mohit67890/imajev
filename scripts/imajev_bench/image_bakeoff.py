"""Image-generator bake-off for imajev-bench v2: which generator meets exact content specs cheapest?

For each spec, every generator makes a base image, and (for specs with an edit) a one-change variant
made by editing that base image. Two verifier families (Azure GPT and Vertex Gemini) independently
check each stated fact; a fact passes only when both say yes. Images, verdicts and usage are written
to a new output directory; nothing existing is modified.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from imajev_bench.api_models import GcloudToken, gcloud_project, load_setting  # noqa: E402

STYLE = ("Realistic photograph taken at eye level with natural lighting. Show only the scene itself: no phone or "
         "device frame, no camera or app interface, no watermark, no captions, no people.")
SPECS = [
    {"id": "menu", "prompt": "A chalkboard menu outside a small cafe listing exactly these five lines in white chalk "
     "handwriting: 'Tomato Soup 5.20', 'Cheese Toastie 6.80', 'Flat White 3.40', 'Carrot Cake 4.10', 'Lemonade 2.90'. "
     "No other text or prices on the board.",
     "facts": ["The board shows 'Tomato Soup 5.20'", "The board shows 'Cheese Toastie 6.80'",
               "The board shows 'Flat White 3.40'", "The board shows 'Carrot Cake 4.10'",
               "The board shows 'Lemonade 2.90'", "No other prices appear on the board"],
     "edit": "Change only the Flat White price from 3.40 to 3.90. Keep everything else identical.",
     "edit_facts": ["The board shows 'Flat White 3.90'", "The board shows 'Tomato Soup 5.20'",
                    "The board shows 'Lemonade 2.90'", "No price reads 3.40"]},
    {"id": "jars", "prompt": "A wooden kitchen shelf holding exactly seven glass jars in one row, left to right: "
     "blue, red, blue, blue, red, blue, red. The blue jars are clearly cobalt blue, the red jars clearly red. Plain wall behind.",
     "facts": ["Exactly 7 jars are visible", "Exactly 4 jars are blue", "Exactly 3 jars are red",
               "Left to right the colours are blue, red, blue, blue, red, blue, red"],
     "edit": "Remove only the rightmost red jar, leaving an empty space. Keep everything else identical.",
     "edit_facts": ["Exactly 6 jars are visible", "Exactly 4 jars are blue", "Exactly 2 jars are red"]},
    {"id": "timetable", "prompt": "A bus stop timetable panel for route 42 listing exactly four departure times: "
     "07:15, 07:45, 08:20, 08:50, under the heading 'Route 42 - Weekdays'.",
     "facts": ["The route number shown is 42", "07:15 is listed", "07:45 is listed", "08:20 is listed",
               "08:50 is listed", "Exactly four departure times are listed"],
     "edit": "Change only the time 08:20 to 08:35. Keep everything else identical.",
     "edit_facts": ["08:35 is listed", "08:20 is not listed", "07:15 is listed", "08:50 is listed"]},
    {"id": "parking", "prompt": "A street parking sign that reads exactly '2 HOUR PARKING', '8AM - 6PM', 'MON - FRI'.",
     "facts": ["The sign says 2 HOUR PARKING", "The sign says 8AM - 6PM", "The sign says MON - FRI"],
     "edit": "Change only '2 HOUR' to '1 HOUR'. Keep everything else identical.",
     "edit_facts": ["The sign says 1 HOUR PARKING", "The sign says 8AM - 6PM", "The sign says MON - FRI"]},
    {"id": "bins", "prompt": "Five wheelie bins in a row against a brick wall, left to right coloured blue, green, "
     "yellow, blue, grey. Only the green bin has its lid open; every other lid is closed.",
     "facts": ["Exactly 5 bins are visible", "Left to right the colours are blue, green, yellow, blue, grey",
               "The green bin's lid is open", "All other lids are closed"],
     "edit": "Close only the green bin's lid. Keep everything else identical.",
     "edit_facts": ["Exactly 5 bins are visible", "All five lids are closed"]},
    {"id": "shelf_prices", "prompt": "A supermarket shelf edge with three printed price labels reading exactly "
     "'Oat Milk 1.85', 'Rice 2.40' and 'Pasta 1.10', each under the matching product.",
     "facts": ["A label reads 'Oat Milk 1.85'", "A label reads 'Rice 2.40'", "A label reads 'Pasta 1.10'"]},
    {"id": "parcel", "prompt": "A cardboard parcel with a printed shipping label showing 'TO: Unit 14, Harbour Road', "
     "'WEIGHT: 3.2 kg' and 'TRACKING: BX-4471'.",
     "facts": ["The label shows 'Unit 14, Harbour Road'", "The label shows 'WEIGHT: 3.2 kg'",
               "The label shows 'TRACKING: BX-4471'"]},
    {"id": "desk", "prompt": "A tidy desk seen from above with exactly three pens and exactly two closed notebooks, "
     "all clearly separated. No other objects.",
     "facts": ["Exactly 3 pens are visible", "Exactly 2 notebooks are visible"]},
    {"id": "thermostat", "prompt": "A wall-mounted digital thermostat whose display reads exactly '21.5°C' and 'HEAT'.",
     "facts": ["The display reads 21.5°C", "The display shows HEAT"]},
    {"id": "nutrition", "prompt": "Close-up of a food package nutrition panel listing exactly: 'Energy 250 kcal', "
     "'Fat 9 g', 'Sugars 12 g', 'Salt 0.8 g'.",
     "facts": ["Energy is 250 kcal", "Fat is 9 g", "Sugars is 12 g", "Salt is 0.8 g"]},
]
ARTIFACTS = "a phone or device frame, a camera or app interface, a watermark, a caption, or a visible person"


def _post_json(url, body, headers, timeout=300):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    return _send(request, timeout)


def _send(request, timeout, retries=4):
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode(errors="replace")
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from None
        time.sleep(2 ** attempt * 3)


class Azure:
    def __init__(self):
        endpoint = urlparse(load_setting("AZURE_OPENAI_ENDPOINT"))
        self.base, self.key = f"{endpoint.scheme}://{endpoint.netloc}/openai/v1", load_setting("AZURE_OPENAI_API_KEY")

    def generate(self, deployment, prompt, image=None):
        if image is None:
            data = _post_json(f"{self.base}/images/generations",
                              {"model": deployment, "prompt": prompt, "size": "1536x1024", "quality": "medium", "n": 1},
                              {"api-key": self.key})
        else:
            boundary = uuid.uuid4().hex
            fields = {"model": deployment, "prompt": prompt, "size": "1536x1024", "quality": "medium"}
            body = b"".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
                            for k, v in fields.items())
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"base.png\"\r\n"
                     "Content-Type: image/png\r\n\r\n").encode() + image + f"\r\n--{boundary}--\r\n".encode()
            request = urllib.request.Request(f"{self.base}/images/edits", data=body, method="POST", headers={
                "api-key": self.key, "Content-Type": f"multipart/form-data; boundary={boundary}"})
            data = _send(request, 300)
        return base64.b64decode(data["data"][0]["b64_json"]), data.get("usage")

    def verify(self, deployment, image, question):
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}},
                   {"type": "text", "text": question}]
        data = _post_json(f"{self.base}/chat/completions", {"model": deployment, "messages": [{"role": "user", "content": content}],
                          "response_format": {"type": "json_object"}}, {"api-key": self.key})
        return json.loads(data["choices"][0]["message"]["content"])


class Vertex:
    def __init__(self):
        self.token, self.project = GcloudToken(), gcloud_project()

    def _url(self, model):
        return (f"https://aiplatform.googleapis.com/v1/projects/{self.project}/locations/global/publishers/google/"
                f"models/{model}:generateContent")

    def generate(self, model, prompt, image=None):
        parts = ([{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(image).decode()}}] if image else []) \
            + [{"text": prompt}]
        data = _post_json(self._url(model), {"contents": [{"role": "user", "parts": parts}], "generationConfig": {
            "responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "3:2", "imageSize": "1K"}}},
            {"Authorization": f"Bearer {self.token.get()}"})
        for part in data["candidates"][0]["content"]["parts"]:
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"]), data.get("usageMetadata")
        raise RuntimeError("No image returned")

    def verify(self, model, image, question):
        data = _post_json(self._url(model), {"contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(image).decode()}}, {"text": question}]}],
            "generationConfig": {"responseMimeType": "application/json"}}, {"Authorization": f"Bearer {self.token.get()}"})
        return json.loads("".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]))


def checklist(facts):
    lines = "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, 1))
    return ("Check this image strictly against each statement. Answer only from what is visible; if text is "
            "illegible or a count is ambiguous, answer 'unclear'.\n" + lines +
            f"\nAlso report whether the image contains {ARTIFACTS}.\n"
            'Reply as JSON: {"facts": ["yes"|"no"|"unclear", ...one per statement in order], '
            '"artifacts": true|false, "note": "<one short sentence>"}')


def evaluate(azure, vertex, image, facts):
    question = checklist(facts)
    verdicts = {}
    for name, call in (("gpt", lambda: azure.verify("gpt-5.4", image, question)),
                       ("gemini", lambda: vertex.verify("gemini-3.1-pro-preview", image, question))):
        try:
            verdicts[name] = call()
        except Exception as exc:  # recorded; a missing verdict means the fact is not confirmed
            verdicts[name] = {"error": str(exc)[:300]}
    per_fact = []
    for index in range(len(facts)):
        votes = [str((v.get("facts") or [None] * len(facts))[index]).lower() if "facts" in v else "error"
                 for v in verdicts.values()]
        per_fact.append("pass" if votes == ["yes", "yes"] else "fail" if votes == ["no", "no"] else "disputed")
    artifacts = [bool(v.get("artifacts")) for v in verdicts.values() if "facts" in v]
    return {"verdicts": verdicts, "per_fact": per_fact, "all_pass": all(x == "pass" for x in per_fact),
            "artifacts_flagged": any(artifacts)}


GENERATORS = {"flare": ("azure", "gpt-image-2.5-flare"), "gpt-image-2": ("azure", "gpt-image-2"),
              "nano-banana-2": ("vertex", "gemini-3.1-flash-image"), "nano-banana-2-lite": ("vertex", "gemini-3.1-flash-lite-image")}


def run_generator(name, out, azure, vertex, specs):
    backend, model = GENERATORS[name]
    client = azure if backend == "azure" else vertex
    folder = out / name
    folder.mkdir()
    rows = []
    for spec in specs:
        for stage in ("base", "edit") if spec.get("edit") else ("base",):
            row = {"generator": name, "model": model, "spec": spec["id"], "stage": stage}
            start = time.time()
            try:
                if stage == "base":
                    image, usage = client.generate(model, f"{spec['prompt']} {STYLE}")
                else:
                    base = (folder / f"{spec['id']}-base.png").read_bytes()
                    image, usage = client.generate(model, f"{spec['edit']} {STYLE}", image=base)
                path = folder / f"{spec['id']}-{stage}.png"
                path.write_bytes(image)
                row.update(seconds=round(time.time() - start, 1), usage=usage, image=str(path.relative_to(out)))
                row.update(evaluate(azure, vertex, image, spec["facts"] if stage == "base" else spec["edit_facts"]))
            except Exception as exc:
                row.update(error=str(exc)[:400], seconds=round(time.time() - start, 1))
            rows.append(row)
            print(json.dumps({k: row.get(k) for k in ("generator", "spec", "stage", "all_pass", "per_fact", "error")}), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generators", nargs="+", default=list(GENERATORS), choices=list(GENERATORS))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    azure, vertex = Azure(), Vertex()
    (args.output / "specs.json").write_text(json.dumps({"style": STYLE, "specs": SPECS}, indent=1))
    with ThreadPoolExecutor(len(args.generators)) as pool:
        results = list(pool.map(lambda g: run_generator(g, args.output, azure, vertex, SPECS), args.generators))
    rows = [row for group in results for row in group]
    with (args.output / "results.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print("done", len(rows))


if __name__ == "__main__":
    main()
