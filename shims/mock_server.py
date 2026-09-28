"""Server OpenAI finto per collaudare l'orchestratore senza modelli.

--fail-start   esce subito con 'Illegal instruction' (simula la build Docker su CPU senza AVX2)
--wrong        risponde sempre sbagliato
--die-after N  termina dopo N richieste
"""
import argparse
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="mock")
ap.add_argument("--threads", default="?")
ap.add_argument("--fail-start", action="store_true")
ap.add_argument("--wrong", action="store_true")
ap.add_argument("--die-after", type=int, default=0)
a, _ = ap.parse_known_args()

if a.fail_start:
    print("Illegal instruction (core dumped)", flush=True)
    sys.exit(132)

count = 0


class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"data": [{"id": "mock"}]})

    def do_POST(self):
        global count
        count += 1
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        text = body["messages"][-1]["content"]
        m = re.search(r"(\d+)\+1", text)
        ans = str(int(m.group(1)) + 1) if m else '{"intent":"unknown"}'
        if a.wrong:
            ans = "boh"
        time.sleep(0.02)
        timings = {"prompt_n": 20, "predicted_n": 3, "prompt_per_second": 100.0, "predicted_per_second": 30.0}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in (ans[:1], ans[1:]):
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n".encode())
            self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'timings': timings})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self._send(200, {"choices": [{"message": {"content": ans}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 20, "completion_tokens": 3}, "timings": timings})
        if a.die_after and count >= a.die_after:
            import os
            os._exit(1)


print(f"mock in ascolto su {a.port}", flush=True)
ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
