"""Image generation and fact checking for synthetic imajev-bench scenes.

Generators: Azure OpenAI image deployments (generations and edits) and Vertex AI Gemini image
models. Checkers: one Azure chat deployment and one Vertex Gemini model, each asked to confirm every
stated fact. A fact passes only when both confirm it. Checkers also report brand logos, people,
device frames and borderline facts, which are routed to human audit. Keys come from `.env.local`
(Azure) and the gcloud login (Vertex) and are never written to disk.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlparse

from .api_models import GcloudToken, gcloud_project, load_setting

STYLE = ("Realistic photograph of an everyday scene, natural lighting. Show only the scene: no phone or device "
         "frame, no camera or app interface, no watermark, no caption, no clearly visible faces, and no real brand "
         "names, logos or trademarks.")
GENERATORS = {"flare": ("azure", "gpt-image-2.5-flare"), "nano-banana-2": ("vertex", "gemini-3.1-flash-image")}
# Checkers from two model families. gpt-5.6-luna matched gpt-5.4 on 212/212 bake-off facts (including both known
# edit failures) at about a tenth of the cost; see reports/imajev-bench-v2-bakeoff/run-1/SUMMARY.md.
CHECKERS = (("azure", "gpt-5.6-luna"), ("vertex", "gemini-3.1-pro-preview"))
CHECKER_PRICES = {"gpt-5.6-luna": (0.20e-6, 1.20e-6), "gpt-5.4": (2.50e-6, 15.0e-6), "gemini-3.1-pro-preview": (2.0e-6, 12.0e-6)}


def checker_cost(model, usage):
    """Estimated USD for one checker call from its token usage (list prices; bills are authoritative)."""
    usage = usage or {}
    tokens_in = usage.get("prompt_tokens", 0) + usage.get("promptTokenCount", 0)
    tokens_out = (usage.get("completion_tokens", 0) + usage.get("candidatesTokenCount", 0)
                  + usage.get("thoughtsTokenCount", 0))
    price_in, price_out = CHECKER_PRICES.get(model, (0.0, 0.0))
    return tokens_in * price_in + tokens_out * price_out
# Estimated list prices (USD). Bills are authoritative; these only feed the run's cost ledger.
PRICES = {"gpt-image-2.5-flare": {"text_in": 5e-6, "image_in": 8e-6, "image_out": 30e-6},
          "gemini-3.1-flash-image": {"per_image": 0.067}}


def _send(request, timeout=300, retries=5, sleep=time.sleep):
    for attempt in range(retries + 1):
        wait = 2 ** attempt * 3
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode(errors="replace")
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f"HTTP {exc.code}: {detail}") from None
            if exc.code == 429:
                wait = min(120, 15 * 2 ** attempt)  # per-minute quotas need minute-scale backoff
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        sleep(wait)


def _json_request(url, body, headers):
    return urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                  headers={"Content-Type": "application/json", **headers})


class AzureClient:
    def __init__(self):
        endpoint = urlparse(load_setting("AZURE_OPENAI_ENDPOINT"))
        self.base, self._key = f"{endpoint.scheme}://{endpoint.netloc}/openai/v1", load_setting("AZURE_OPENAI_API_KEY")

    def image(self, deployment, prompt, source=None, quality="medium", size="1536x1024"):
        if source is None:
            data = _send(_json_request(f"{self.base}/images/generations", {
                "model": deployment, "prompt": prompt, "size": size, "quality": quality, "n": 1}, {"api-key": self._key}))
        else:
            boundary = uuid.uuid4().hex
            fields = {"model": deployment, "prompt": prompt, "size": size, "quality": quality}
            body = b"".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
                            for k, v in fields.items())
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"source.png\"\r\n"
                     "Content-Type: image/png\r\n\r\n").encode() + source + f"\r\n--{boundary}--\r\n".encode()
            data = _send(urllib.request.Request(f"{self.base}/images/edits", data=body, method="POST", headers={
                "api-key": self._key, "Content-Type": f"multipart/form-data; boundary={boundary}"}))
        usage = data.get("usage") or {}
        price = PRICES.get(deployment, {})
        cost = (usage.get("input_tokens_details", {}).get("text_tokens", 0) * price.get("text_in", 0)
                + usage.get("input_tokens_details", {}).get("image_tokens", 0) * price.get("image_in", 0)
                + usage.get("output_tokens", 0) * price.get("image_out", 0))
        return base64.b64decode(data["data"][0]["b64_json"]), {"usage": usage, "estimated_usd": round(cost, 5)}

    def check(self, deployment, images, question):
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(i).decode()}}
                   for i in images] + [{"type": "text", "text": question}]
        data = _send(_json_request(f"{self.base}/chat/completions", {
            "model": deployment, "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"}}, {"api-key": self._key}))
        return json.loads(data["choices"][0]["message"]["content"]), data.get("usage")


class VertexClient:
    def __init__(self, project=None, location="global", max_concurrent_images=2):
        self.token, self.project, self.location = GcloudToken(), project or gcloud_project(), location
        self._image_slots = threading.Semaphore(max_concurrent_images)  # image models have small per-minute quotas

    def _url(self, model):
        host = "aiplatform.googleapis.com" if self.location == "global" else f"{self.location}-aiplatform.googleapis.com"
        return f"https://{host}/v1/projects/{self.project}/locations/{self.location}/publishers/google/models/{model}:generateContent"

    def _call(self, model, body):
        try:
            return _send(_json_request(self._url(model), body, {"Authorization": f"Bearer {self.token.get()}"}))
        except RuntimeError as exc:
            if not str(exc).startswith("HTTP 401"):
                raise
            return _send(_json_request(self._url(model), body, {"Authorization": f"Bearer {self.token.get(refresh=True)}"}))

    def image(self, model, prompt, source=None, **_):
        parts = ([{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(source).decode()}}] if source else []) \
            + [{"text": prompt}]
        with self._image_slots:
            data = self._call(model, {"contents": [{"role": "user", "parts": parts}], "generationConfig": {
                "responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "3:2", "imageSize": "1K"}}})
        for part in data["candidates"][0]["content"]["parts"]:
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"]), {
                    "usage": data.get("usageMetadata"), "estimated_usd": PRICES.get(model, {}).get("per_image", 0)}
        raise RuntimeError("Vertex returned no image")

    def check(self, model, images, question):
        parts = [{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(i).decode()}} for i in images]
        data = self._call(model, {"contents": [{"role": "user", "parts": parts + [{"text": question}]}],
                                  "generationConfig": {"responseMimeType": "application/json"}})
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        return json.loads(text), data.get("usageMetadata")


def check_question(facts: list[str]) -> str:
    lines = "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts, 1))
    return ("Check this image strictly against each statement, using only what is visible. Answer 'unclear' only "
            "when you cannot tell whether a statement is true: for example text too small or blurry to read, a colour "
            "that could reasonably be named differently, or a count you cannot establish with confidence. A statement "
            "that something is covered or cannot be read is true when that thing genuinely cannot be read.\n"
            + lines + "\n"
            "Also report: (a) whether the image shows a phone or device frame, an app interface, a watermark or a "
            "caption; (b) whether any clearly identifiable human face is visible (blurred or distant background figures "
            "do not count); (c) whether any real brand name, logo or trademark is visible.\n"
            'Reply as JSON: {"facts": ["yes"|"no"|"unclear", ... one per statement, in order], '
            '"frame_or_ui": true|false, "identifiable_face": true|false, "brands": true|false, "note": "<one short sentence>"}')


def verify(clients: dict, image: bytes, facts: list[str]) -> dict:
    """Both checker families must say 'yes' to every fact; anything else fails or is flagged."""
    question = check_question(facts)
    verdicts = {}
    for backend, model in CHECKERS:
        try:
            verdict, usage = clients[backend].check(model, [image], question)
            verdicts[model] = {**verdict, "usage": usage}
        except Exception as exc:  # a missing verdict is a failed check, recorded
            verdicts[model] = {"error": str(exc)[:300]}
    votes = [[str(x).lower() for x in (v.get("facts") or [])] for v in verdicts.values()]
    complete = all(len(v) == len(facts) for v in votes)
    per_fact = ["pass" if complete and all(v[i] == "yes" for v in votes) else
                "fail" if complete and all(v[i] == "no" for v in votes) else "unclear" for i in range(len(facts))]
    # Generated people are not real persons; only clearly identifiable faces fail (privacy-by-default, and realism).
    hygiene = {key: any(bool(v.get(key)) for v in verdicts.values()) for key in ("frame_or_ui", "identifiable_face", "brands")}
    cost = sum(checker_cost(model, v.get("usage")) for model, v in verdicts.items())
    return {"per_fact": per_fact, "passed": all(x == "pass" for x in per_fact) and not any(hygiene.values()),
            "hygiene": hygiene, "verdicts": verdicts, "checker_usd": round(cost, 5)}
