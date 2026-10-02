"""Server OpenAI minimo per motori Python senza server proprio: Transformers (PyTorch) e
ONNX Runtime GenAI. Solo /v1/models e /v1/chat/completions, senza streaming né grammatiche:
i test JSON vincolato e chiamata di strumenti falliranno o daranno risultati peggiori, ed è un
risultato anche questo (è ciò che ottiene chi usa questi runtime "nudi").

Sperimentale: non collaudato su NASGUL/Z87.
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["transformers", "onnx"], required=True)
ap.add_argument("--model", required=True, help="repo HF o cartella locale")
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--dtype", default="float32")
ap.add_argument("--cache-dir", default="", help="cartella dove scaricare i modelli ONNX come file veri")
ap.add_argument("--thinking", choices=["on", "off"], default="off",
                help="passato al template di chat come enable_thinking (i template che non lo usano lo ignorano)")
a = ap.parse_args()

if a.backend == "transformers":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if a.threads:
        torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype))
    except TypeError:   # transformers < 4.56 conosce solo torch_dtype
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype))
    model.eval()

    def generate(messages, max_tokens, temperature):
        # transformers 5 restituisce un BatchEncoding (input_ids + attention_mask), non più il tensore:
        # chiedendolo esplicitamente con return_dict=True il codice vale per entrambe le versioni
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=a.thinking == "on")
        n_in = enc["input_ids"].shape[1]
        kw = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kw["temperature"] = temperature
        t0 = time.monotonic()
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        dt = time.monotonic() - t0
        new = out[0][n_in:]
        return tok.decode(new, skip_special_tokens=True), n_in, len(new), dt
else:
    import onnxruntime_genai as og
    from huggingface_hub import snapshot_download
    import os
    # i repo ONNX hanno sottocartelle per variante (cpu_and_mobile/..., gpu/...): si scarica e si usa
    # solo quella per CPU, altrimenti os.walk può prendere la variante GPU e og.Model fallisce
    # local_dir: file veri in una cartella sola. Nella cache di Hugging Face i file sono collegamenti
    # verso blobs/xx/..., e onnxruntime rifiuta model.onnx.data se "esce" dalla cartella del modello.
    local = os.path.join(a.cache_dir or os.path.expanduser("~/.cache/gilpa-lab-onnx"), a.model.replace("/", "--"))
    path = a.model if os.path.isdir(a.model) else snapshot_download(
        a.model, allow_patterns=["cpu*/**", "*.json", "*.py", "*.txt"], local_dir=local)
    found = sorted(root for root, _, files in os.walk(path) if "genai_config.json" in files)
    if not found:
        sys.exit(f"nessun genai_config.json in {path}")
    path = next((r for r in found if "cpu" in r.lower()), found[0])
    print(f"variante ONNX: {path}", flush=True)
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
        try:
            text, p, g, dt = generate(body["messages"], int(body.get("max_tokens", 256)),
                                      float(body.get("temperature", 0)))
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(tb, flush=True)       # finisce nel log del server
            self._send(500, {"error": {"message": tb[-2000:]}})
            return
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": p, "completion_tokens": g},
                         "timings": {"prompt_n": p, "predicted_n": g,
                                     "predicted_per_second": g / dt if dt else None}})


print(f"shim {a.backend} su {a.port}", flush=True)
HTTPServer(("127.0.0.1", a.port), H).serve_forever()
