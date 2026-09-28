"""Client OpenAI-compatibile minimo (solo stdlib), con misura di tempi e token."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class Client:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 1800,
                 extra_body: dict | None = None):
        self.base = base_url.rstrip("/")
        self.model = model
        self.key = api_key
        self.timeout = timeout
        self.extra_body = extra_body or {}

    def _headers(self):
        h = {"Content-Type": "application/json", "User-Agent": "gilpa-lab/1.0"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def get(self, path: str, timeout: float = 10):
        req = urllib.request.Request(self.base + path, headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def ready(self) -> bool:
        try:
            self.get("/models", timeout=5)
            return True
        except urllib.error.HTTPError as e:
            return e.code not in (503,)  # llama-server risponde 503 mentre carica
        except Exception:
            return False

    def chat(self, messages, *, max_tokens=256, temperature=0.0, seed=42, response_format=None,
             tools=None, stream=True, extra: dict | None = None) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "seed": seed, **self.extra_body, **(extra or {})}
        if response_format:
            body["response_format"] = response_format
        if tools:
            body["tools"] = tools
            stream = False
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        req = urllib.request.Request(self.base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers=self._headers(), method="POST")
        t0 = time.monotonic()
        out = {"content": "", "reasoning": "", "tool_calls": None, "usage": {}, "timings": {},
               "ttft_s": None, "wall_s": None, "finish_reason": None}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                if not stream:
                    data = json.loads(r.read())
                    ch = (data.get("choices") or [{}])[0]
                    msg = ch.get("message") or {}
                    out["content"] = msg.get("content") or ""
                    out["reasoning"] = msg.get("reasoning_content") or ""
                    out["tool_calls"] = msg.get("tool_calls")
                    out["finish_reason"] = ch.get("finish_reason")
                    out["usage"] = data.get("usage") or {}
                    out["timings"] = data.get("timings") or {}
                else:
                    for raw in r:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            data = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        for ch in data.get("choices") or []:
                            d = ch.get("delta") or {}
                            piece = d.get("content") or ""
                            rpiece = d.get("reasoning_content") or ""
                            if (piece or rpiece) and out["ttft_s"] is None:
                                out["ttft_s"] = time.monotonic() - t0
                            out["content"] += piece
                            out["reasoning"] += rpiece
                            if ch.get("finish_reason"):
                                out["finish_reason"] = ch["finish_reason"]
                        if data.get("usage"):
                            out["usage"] = data["usage"]
                        if data.get("timings"):
                            out["timings"] = data["timings"]
        finally:
            out["wall_s"] = time.monotonic() - t0
        out["content"] = THINK_RE.sub("", out["content"]).strip()
        return out


def speed_fields(res: dict) -> dict:
    """Estrae token e velocità, preferendo i 'timings' di llama-server se presenti."""
    t, u = res.get("timings") or {}, res.get("usage") or {}
    return {
        "wall_s": round(res["wall_s"], 3) if res.get("wall_s") else None,
        "ttft_s": round(res["ttft_s"], 3) if res.get("ttft_s") else None,
        "prompt_tokens": t.get("prompt_n", u.get("prompt_tokens")),
        "gen_tokens": t.get("predicted_n", u.get("completion_tokens")),
        "prompt_tps": round(t["prompt_per_second"], 2) if t.get("prompt_per_second") else None,
        "gen_tps": round(t["predicted_per_second"], 2) if t.get("predicted_per_second") else None,
    }
