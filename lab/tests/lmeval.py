"""Test pubblici eseguiti con lm-evaluation-harness contro l'endpoint OpenAI del motore.

Solo task generativi: llama-server non espone i logprob del prompt che servono ai task a scelta
multipla (per quelli ci sono le varianti generative in generative.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..util import LabError, run_cmd
from .base import QUALITY, Result, Test, lib_version


class LmEval(Test):
    kind = "lmeval"
    VERSION = "1"
    default_depends = QUALITY

    def data_fingerprint(self):
        return f"lm_eval={lib_version('lm_eval')}"

    def prepare(self, log):
        if lib_version("lm_eval") == "assente":
            raise LabError("missing_dep", "lm-eval non installato: esegui ./lab/setup.sh")
        # scarica i dataset dei task senza modello: --limit 0 non esiste, quindi si passa da TaskManager
        code = ("import sys; from lm_eval.tasks import TaskManager, get_task_dict;"
                "get_task_dict(sys.argv[1:], TaskManager())")
        run_cmd([sys.executable, "-c", code, *self.cfg["tasks"]], log, timeout=3600,
                env={"HF_HOME": str(self.lab.data_dir / "hf_home"), "HF_DATASETS_TRUST_REMOTE_CODE": "1"})

    def run(self, rt):
        c = self.cfg
        out_dir = rt.log.parent / (rt.log.stem + "-lmeval")
        margs = [f"model={rt.server.model_name}", f"base_url={rt.server.base_url}/chat/completions",
                 "num_concurrent=1", "max_retries=2", "tokenized_requests=False",
                 f"timeout={rt.lab.cfg['timeouts'].get('request_s', 1800)}"]
        if rt.server.think_end:
            margs.append(f"think_end_token={rt.server.think_end}")
        cmd = [sys.executable, "-m", "lm_eval", "--model", "local-chat-completions",
               "--model_args", ",".join(margs), "--tasks", ",".join(c["tasks"]),
               "--apply_chat_template", "--output_path", out_dir, "--log_samples",
               "--seed", str(c.get("seed", 1234))]
        if c.get("limit"):
            cmd += ["--limit", str(c["limit"])]
        if c.get("num_fewshot") is not None:
            cmd += ["--num_fewshot", str(c["num_fewshot"])]
            if c.get("num_fewshot"):
                cmd += ["--fewshot_as_multiturn"]
        gk = dict(c.get("gen_kwargs") or {})
        if rt.params.get("thinking") == "on" and c.get("max_gen_toks_thinking"):
            gk["max_gen_toks"] = c["max_gen_toks_thinking"]
        if gk:
            cmd += ["--gen_kwargs", ",".join(f"{k}={v}" for k, v in gk.items())]
        run_cmd(cmd, rt.log, timeout=rt.lab.cfg["timeouts"].get("test_s", 86400),
                env={"OPENAI_API_KEY": "sk-lab", "HF_HOME": str(rt.lab.data_dir / "hf_home"),
                     "HF_DATASETS_TRUST_REMOTE_CODE": "1"})
        res_files = sorted(out_dir.rglob("results_*.json"))
        if not res_files:
            raise LabError("parse", f"lm-eval non ha prodotto risultati in {out_dir}")
        data = json.loads(res_files[-1].read_text())
        metrics = {}
        for task, vals in (data.get("results") or {}).items():
            for k, v in vals.items():
                if isinstance(v, (int, float)) and not k.startswith("alias") and "_stderr" not in k:
                    metrics[f"{task}.{k.replace(',none', '')}"] = (v, None)
        samples = []
        for sf in sorted(out_dir.rglob("samples_*.jsonl")):
            task = sf.stem.replace("samples_", "").rsplit("_", 1)[0]
            primary = c.get("primary_metric")
            for line in sf.open(encoding="utf-8"):
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue
                val = s.get(primary) if primary else None
                if val is None:
                    val = next((v for k, v in s.items() if isinstance(v, (int, float, bool))
                                and k not in ("doc_id",) and "acc" in k), None)
                resp = s.get("filtered_resps") or s.get("resps")
                samples.append({"case_id": f"{task}:{s.get('doc_id')}", "rep": 0,
                                "correct": float(val) if isinstance(val, (int, float, bool)) else None,
                                "response": str(resp)[:2000]})
        return Result(metrics, samples, {"lm_eval_version": lib_version("lm_eval"),
                                         "results_file": str(res_files[-1])})
