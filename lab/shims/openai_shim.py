"""Server OpenAI minimo per motori Python senza server proprio: Transformers (PyTorch) e
ONNX Runtime GenAI. Solo /v1/models e /v1/chat/completions, senza streaming né grammatiche:
i test JSON vincolato e chiamata di strumenti falliranno o daranno risultati peggiori, ed è un
risultato anche questo (è ciò che ottiene chi usa questi runtime "nudi").

Sperimentale: non collaudato su NASGUL/Z87.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["transformers", "onnx"], required=True)
ap.add_argument("--model", required=True, help="repo HF o cartella locale")
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--dtype", default="float32")
a = ap.parse_args()

if a.backend == "transformers":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if a.threads:
        torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype))
    model.eval()

    def generate(messages, max_tokens, temperature):
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        t0 = time.monotonic()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                 temperature=temperature or None)
        dt = time.monotonic() - t0
        new = out[0][ids.shape[1]:]
        return tok.decode(new, skip_special_tokens=True), ids.shape[1], len(new), dt
else:
    import onnxruntime_genai as og
    from huggingface_hub import snapshot_download
    import os
    path = a.model if os.path.isdir(a.model) else snapshot_download(a.model)
    # i repo ONNX hanno spesso sottocartelle per variante: si usa la prima con genai_config.json
    for root, _, files in os.walk(path):
        if "genai_config.json" in files:
            path = root
            break
    model = og.Model(path)
    tok = og.Tokenizer(model)
    from transformers import AutoTokenizer
    try:
        chat_tok = AutoTokenizer.from_pretrained(path)
    except Exception:
        chat_tok = None

    def generate(messages, max_tokens, temperature):
        if chat_tok is not None:
            prompt = chat_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            prompt = "\n".join(m["content"] for m in messages) + "\n"
        ids = tok.encode(prompt)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(ids) + max_tokens, do_sample=temperature > 0)
        gen = og.Generator(model, params)
        gen.append_tokens(ids)
        t0 = time.monotonic()
        out = []
        while not gen.is_done():
            gen.generate_next_token()
            out.append(gen.get_next_tokens()[0])
        dt = time.monotonic() - t0
        return tok.decode(out), len(ids), len(out), dt


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
        self._send(200, {"data": [{"id": "lab"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if body.get("tools") or body.get("response_format"):
            pass  # ignorati: nessuna grammatica in questi runtime
        text, p, g, dt = generate(body["messages"], int(body.get("max_tokens", 256)), float(body.get("temperature", 0)))
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": p, "completion_tokens": g},
                         "timings": {"prompt_n": p, "predicted_n": g,
                                     "predicted_per_second": g / dt if dt else None}})


print(f"shim {a.backend} su {a.port}", flush=True)
HTTPServer(("127.0.0.1", a.port), H).serve_forever()
