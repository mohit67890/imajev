#!/usr/bin/env python3
"""Keyless OpenAI-compatible client for local vLLM servers (phase-3 label-then-train pod, reports/phase3/LABELTRAIN-RUNBOOK.md).

The pod serves the teacher (Qwen/Qwen3.6-35B-A3B-FP8, one vLLM server per GPU) and the variant writer (Qwen/Qwen3.6-27B-FP8) on
127.0.0.1. There is no key on the pod, ever: requests carry NO api-key and NO Authorization header.

    routes = parse_servers(["gpu0=http://127.0.0.1:8100/v1", "gpu1=http://127.0.0.1:8101/v1"])
    client = LocalClient(routes, served_model="teacher")
    client.post({"model": "gpu0", ...})        # -> POST http://127.0.0.1:8100/v1/chat/completions with model "teacher"

`body["model"]` names the ROUTE (the worker's deployment name in azure_teacher.Runner); the client rewrites it to the served model
name before sending. Retries and backoff are azure_teacher.AzureClient's (429 / 5xx / connection errors). `healthy(route)` is a
cached GET <base>/models check: a worker whose server is down (still loading, or switched from writer to teacher later) takes no
task, so nothing is dropped as solve_failed while a server is missing.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

from azure_teacher import AzureClient, TransientError

__all__ = ["LocalClient", "parse_servers", "server_models", "TransientError"]


def parse_servers(specs: list[str]) -> dict[str, str]:
    """["name=url", "url", ...] (repeatable, comma-separable) -> {route name: base url}; bare urls are named gpu0, gpu1, ..."""
    out: dict[str, str] = {}
    for spec in specs or []:
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            name, sep, url = part.partition("=")
            if not sep or name.startswith("http"):
                name, url = f"gpu{len(out)}", part
            if name in out:
                raise ValueError(f"route {name} given twice")
            out[name] = url.rstrip("/")
    if not out:
        raise ValueError("no --server given")
    return out


def server_models(base_url: str, timeout: float = 3.0) -> list[str] | None:
    """Model ids a server lists at <base>/models, or None when it does not answer."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as resp:
            return [m.get("id") for m in json.loads(resp.read().decode()).get("data", [])]
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError):
        return None


class _Keyless(AzureClient):
    """AzureClient's retry loop, without any credential header."""

    def post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        last = None
        for attempt in range(self.max_attempts):
            req = urllib.request.Request(self.url, data=data, headers={"content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                msg = f"HTTP {e.code} " + e.read()[:300].decode(errors="ignore")
                if e.code == 429 or e.code >= 500:
                    self.stats[f"http_{e.code}"] += 1
                    last = msg
                    self.sleep(min(60.0, (2 ** attempt) * 0.5))
                    continue
                raise RuntimeError(msg) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                self.stats["conn_error"] += 1
                last = f"{type(e).__name__}: {str(e)[:200]}"
                self.sleep(min(60.0, (2 ** attempt) * 0.5))
                continue
        raise TransientError(f"gave up after {self.max_attempts} attempts: {last}")


class LocalClient:
    """Routes a request to the server named by body["model"], rewrites the model to the served name, sends no key."""

    def __init__(self, routes: dict[str, str], served_model: str, timeout: float = 1800.0, max_attempts: int = 8,
                 sleep=time.sleep, health_ttl: float = 5.0):
        self.routes, self.served_model = dict(routes), served_model
        self.clients = {name: _Keyless(url, "", timeout=timeout, max_attempts=max_attempts, sleep=sleep)
                        for name, url in self.routes.items()}
        self.key = ""                      # Redactor / callers that read client.key see an empty secret
        self.health_ttl = health_ttl
        self._health: dict[str, tuple[float, bool]] = {}
        self._lock = threading.Lock()

    @property
    def stats(self) -> Counter:
        total = Counter()
        for c in self.clients.values():
            total.update(c.stats)
        return total

    def healthy(self, route: str) -> bool:
        now = time.time()
        with self._lock:
            t, ok = self._health.get(route, (0.0, False))
            if now - t < self.health_ttl:
                return ok
            self._health[route] = (now, ok)        # one probe per TTL, the others read the cached value meanwhile
        models = server_models(self.routes[route])
        ok = bool(models) and self.served_model in models
        with self._lock:
            self._health[route] = (time.time(), ok)
        return ok

    def post(self, body: dict) -> dict:
        route = body.get("model")
        if route not in self.clients:
            if len(self.clients) == 1:
                route = next(iter(self.clients))
            else:
                raise RuntimeError(f"HTTP 404 unknown route {route!r}")
        return self.clients[route].post(dict(body, model=self.served_model))
