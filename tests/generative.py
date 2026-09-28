"""Test pubblici eseguiti con un nostro valutatore generativo, direttamente sull'endpoint OpenAI.

Servono dove lm-eval userebbe i logprob (Belebele, Evalita SA) o dove la valutazione passa da
funzioni del motore (JSON con grammatica, chiamata di strumenti). I prompt sono nostri: i numeri
sono confrontabili tra le nostre configurazioni, non con le classifiche pubbliche in valore assoluto.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..client import speed_fields
from ..util import LabError, sha256_file
from .base import QUALITY, Result, Test, check_errors, lib_version, load_hf, sample, speed_summary


class _Gen(Test):
    default_depends = QUALITY
    dataset = None  # (path, config, split)

    def items(self):
        path, name, split = self.dataset_spec()
        ds = load_hf(self.lab, path, name, split)
        return sample(list(ds), self.cfg.get("limit"), self.cfg.get("seed", 1234))

    def dataset_spec(self):
        return self.dataset

    def data_fingerprint(self):
        return f"datasets={lib_version('datasets')}:{self.dataset_spec()}"

    def prepare(self, log):
        self.items()

    def max_tokens(self, rt, base):
        return self.cfg.get("max_tokens_thinking", 2048) if rt.params.get("thinking") == "on" else \
            self.cfg.get("max_tokens", base)

    def loop(self, rt, items, fn):
        cl = rt.client()
        samples = []
        for i, it in enumerate(items):
            try:
                s = fn(cl, it)
            except Exception as e:
                s = {"case_id": str(i), "correct": None, "error": f"{type(e).__name__}: {e}"[:500]}
            s.setdefault("case_id", str(i))
            s.setdefault("rep", 0)
            samples.append(s)
        check_errors(samples)
        return samples

    @staticmethod
    def acc(samples):
        vals = [s["correct"] for s in samples if s.get("correct") is not None]
        return (100 * sum(vals) / len(samples) if samples else None, "%")


class Belebele(_Gen):
    kind = "belebele"
    VERSION = "1"
    PROMPTS = {
        "ita_Latn": ("Leggi il testo e rispondi alla domanda. Rispondi solo con la lettera dell'opzione "
                     "corretta (A, B, C o D).\n\nTesto: {p}\n\nDomanda: {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}"),
        "eng_Latn": ("Read the passage and answer the question. Reply with only the letter of the correct "
                     "option (A, B, C or D).\n\nPassage: {p}\n\nQuestion: {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}"),
    }

    def dataset_spec(self):
        return ("facebook/belebele", self.cfg.get("lang", "ita_Latn"), "test")

    def run(self, rt):
        tpl = self.PROMPTS.get(self.cfg.get("lang", "ita_Latn"), self.PROMPTS["eng_Latn"])

        def one(cl, it):
            msg = tpl.format(p=it["flores_passage"], q=it["question"], a=it["mc_answer1"], b=it["mc_answer2"],
                             c=it["mc_answer3"], d=it["mc_answer4"])
            r = cl.chat([{"role": "user", "content": msg}], max_tokens=self.max_tokens(rt, 8))
            m = re.search(r"\b([ABCD])\b", r["content"].upper())
            gold = "ABCD"[int(it["correct_answer_num"]) - 1]
            return {"case_id": f"{str(it.get('link', ''))[-16:]}:{it.get('question_number')}", "correct": float(bool(m and m.group(1) == gold)),
                    "answer": r["content"][:50], **speed_fields(r)}

        samples = self.loop(rt, self.items(), one)
        return Result({"acc_pct": self.acc(samples), **speed_summary(samples)}, samples)


class EvalitaSA(_Gen):
    """Evalita-LLM sentiment (SENTIPOLC), in forma generativa. Metrica come l'originale:
    F1 su 'opos' e 'oneg' separatamente, poi la media."""
    kind = "evalita_sa"
    VERSION = "1"
    dataset = ("evalitahf/sentiment_analysis", None, "test")
    LABELS = {"positivo": (1, 0), "negativo": (0, 1), "neutrale": (0, 0), "misto": (1, 1)}

    def run(self, rt):
        schema = {"type": "json_schema", "json_schema": {"name": "sentiment", "strict": True, "schema": {
            "type": "object", "properties": {"sentiment": {"type": "string", "enum": list(self.LABELS)}},
            "required": ["sentiment"], "additionalProperties": False}}}
        use_schema = self.cfg.get("constrained", True)

        def one(cl, it):
            msg = ("Qual è il sentiment espresso nel seguente tweet? Rispondi con una sola parola tra: "
                   f"positivo, negativo, neutrale, misto.\n\nTweet: {it['text']}")
            if use_schema:
                msg += '\n\nRispondi in JSON: {"sentiment": "..."}'
            r = cl.chat([{"role": "user", "content": msg}], max_tokens=self.max_tokens(rt, 24),
                        response_format=schema if use_schema else None)
            txt = r["content"].lower()
            pred = next((k for k in self.LABELS if k in txt), None)
            gold = (int(it["opos"]), int(it["oneg"]))
            p = self.LABELS.get(pred, (0, 0))
            return {"correct": float(p == gold), "pred": pred, "gold": list(gold), "p_opos": p[0], "p_oneg": p[1],
                    **speed_fields(r)}

        samples = self.loop(rt, self.items(), one)
        ok = [s for s in samples if not s.get("error")]

        def f1_macro(y, yp):
            f = []
            for c in (0, 1):
                tp = sum(1 for a, b in zip(y, yp) if a == c and b == c)
                fp = sum(1 for a, b in zip(y, yp) if a != c and b == c)
                fn = sum(1 for a, b in zip(y, yp) if a == c and b != c)
                f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
            return sum(f) / 2

        f_pos = f1_macro([s["gold"][0] for s in ok], [s["p_opos"] for s in ok])
        f_neg = f1_macro([s["gold"][1] for s in ok], [s["p_oneg"] for s in ok])
        return Result({"f1": (100 * (f_pos + f_neg) / 2, "%"), "f1_opos": (100 * f_pos, "%"),
                       "f1_oneg": (100 * f_neg, "%"), "acc_pct": self.acc(samples),
                       "unparsed": (sum(1 for s in ok if s["pred"] is None), "n"), **speed_summary(samples)}, samples)


class JsonSchemaConstrained(_Gen):
    """JSONSchemaBench, zero-shot, con o senza decodifica vincolata (response_format del motore).

    Versione ridotta: `max_schema_chars` tiene solo gli schemi corti, perché su NASGUL ogni token di
    prompt costa ~0,17 s. Stessi casi con e senza grammatica: la differenza misura quanto la
    grammatica del motore salva un modello piccolo."""
    kind = "jsonschema_constrained"
    VERSION = "2"

    def items(self):
        out = []
        cap = self.cfg.get("max_schema_chars")
        for sub in self.cfg.get("subsets", ["Github_easy"]):
            ds = [r for r in load_hf(self.lab, "epfl-dlab/JSONSchemaBench", sub, "test")
                  if not cap or len(r["json_schema"]) <= cap]
            for i, it in enumerate(sample(ds, self.cfg.get("limit"), self.cfg.get("seed", 1234))):
                out.append({"sub": sub, "i": i, **it})
        return out

    def data_fingerprint(self):
        return f"datasets={lib_version('datasets')}:jsonschema={lib_version('jsonschema')}"

    def run(self, rt):
        try:
            import jsonschema
        except ImportError:
            raise LabError("missing_dep", "manca jsonschema: esegui ./lab/setup.sh")
        import urllib.error

        constrained = self.cfg.get("constrained", True)

        def one(cl, it):
            schema = json.loads(it["json_schema"])
            rf = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}} \
                if constrained else None
            msg = ("Generate a JSON object that matches the following JSON schema. Reply with the JSON object only."
                   f"\n\nJSON schema: {it['json_schema']}\n\nJSON object:")
            base = {"case_id": f"{it['sub']}:{it.get('unique_id', it['i'])}"}
            try:
                r = cl.chat([{"role": "user", "content": msg}], max_tokens=self.max_tokens(rt, 1024),
                            response_format=rf)
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:
                    return {**base, "correct": 0.0, "supported": False, "valid": False,
                            "detail": e.read()[:200].decode("utf-8", "replace")}
                raise
            txt = r["content"].strip()
            if not constrained:
                txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
                a, b = txt.find("{"), txt.rfind("}")
                txt = txt[a:b + 1] if a >= 0 and b > a else txt
            try:
                obj = json.loads(txt)
                valid = True
            except json.JSONDecodeError:
                return {**base, "correct": 0.0, "supported": True, "valid": False, **speed_fields(r)}
            try:
                jsonschema.validate(obj, schema, format_checker=jsonschema.FormatChecker())
                comp = True
            except jsonschema.ValidationError:
                comp = False
            except Exception:
                comp = False
            return {**base, "correct": float(comp), "supported": True, "valid": valid, **speed_fields(r)}

        samples = self.loop(rt, self.items(), one)
        n = len(samples) or 1
        sup = [s for s in samples if s.get("supported")]
        m = {"compliance_pct": self.acc(samples),
             "supported_pct": (100 * len(sup) / n, "%"),
             "valid_json_pct": (100 * sum(1 for s in samples if s.get("valid")) / n, "%"),
             "compliance_if_supported_pct": (100 * sum(s["correct"] for s in sup) / len(sup) if sup else None, "%"),
             **speed_summary(samples)}
        for sub in self.cfg.get("subsets", []):
            S = [s for s in samples if s["case_id"].startswith(sub + ":")]
            if S:
                m[f"{sub}.compliance_pct"] = self.acc(S)
        return Result(m, samples)


class MultiIF(_Gen):
    """Multi-IF (Meta): IFEval tradotto, a più turni. Usa i verificatori IFEval di lm-eval.
    I nomi dei campi del dataset vanno verificati al primo prepare: se non corrispondono il test
    fallisce con errore 'schema' invece di dare numeri sbagliati."""
    kind = "multi_if"
    VERSION = "1"

    def dataset_spec(self):
        return ("facebook/Multi-IF", None, "train")

    def items(self):
        ds = load_hf(self.lab, "facebook/Multi-IF", None, self.cfg.get("split", "train"))
        lang = self.cfg.get("language", "Italian").lower()
        rows = [r for r in ds if str(r.get("language", "")).lower() == lang]
        if not rows:
            langs = sorted({str(r.get("language")) for r in ds})[:20]
            raise LabError("schema", f"nessuna riga con language={lang}; lingue trovate: {langs}")
        return sample(rows, self.cfg.get("limit"), self.cfg.get("seed", 1234))

    def data_fingerprint(self):
        return f"datasets={lib_version('datasets')}:lm_eval={lib_version('lm_eval')}:{self.cfg.get('language')}"

    def run(self, rt):
        try:
            from lm_eval.tasks.ifeval import instructions_registry as reg
        except ImportError:
            raise LabError("missing_dep", "servono i verificatori IFEval di lm-eval: esegui ./lab/setup.sh")

        def parse(v):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return v
            return v

        def check(ids, kwargs, resp):
            ok = []
            for iid, kw in zip(ids, kwargs):
                cls = reg.INSTRUCTION_DICT.get(iid)
                if cls is None:
                    ok.append(None)
                    continue
                inst = cls(iid)
                kw = {k: v for k, v in (kw or {}).items() if v is not None}
                try:
                    inst.build_description(**kw)
                except Exception:
                    pass
                ok.append(bool(resp.strip()) and bool(inst.check_following(resp)))
            return ok

        n_turns = self.cfg.get("turns", 3)

        def one(cl, it):
            msgs, out = [], {"case_id": str(it.get("key", "")), "turns": []}
            for t in range(1, n_turns + 1):
                p = parse(it.get(f"turn_{t}_prompt"))
                if not p:
                    break
                content = p.get("content") if isinstance(p, dict) else p
                ids = parse(it.get(f"turn_{t}_instruction_id_list")) or []
                kws = parse(it.get(f"turn_{t}_kwargs")) or [{}] * len(ids)
                kws = [parse(k) if isinstance(k, str) else k for k in kws]
                msgs.append({"role": "user", "content": content})
                r = cl.chat(msgs, max_tokens=self.max_tokens(rt, 1024), stream=True)
                msgs.append({"role": "assistant", "content": r["content"]})
                res = check(ids, kws, r["content"])
                evaluable = [x for x in res if x is not None]
                out["turns"].append({"prompt_ok": all(evaluable) if evaluable else None,
                                     "inst_ok": sum(evaluable), "inst_n": len(evaluable),
                                     "unknown": sum(1 for x in res if x is None), **speed_fields(r)})
            if not out["turns"]:
                raise LabError("schema", f"campi turn_1_* assenti: {list(it)[:12]}")
            last = out["turns"][-1]
            out.update({k: last.get(k) for k in ("wall_s", "ttft_s", "prompt_tokens", "gen_tokens",
                                                 "prompt_tps", "gen_tps")})
            out["correct"] = float(bool(last["prompt_ok"]))
            return out

        samples = self.loop(rt, self.items(), one)
        m = speed_summary(samples)
        for t in range(n_turns):
            T = [s["turns"][t] for s in samples if not s.get("error") and len(s.get("turns", [])) > t]
            if not T:
                continue
            m[f"turn{t + 1}.prompt_acc_pct"] = (100 * sum(1 for x in T if x["prompt_ok"]) / len(T), "%")
            inst_n = sum(x["inst_n"] for x in T)
            m[f"turn{t + 1}.inst_acc_pct"] = (100 * sum(x["inst_ok"] for x in T) / inst_n if inst_n else None, "%")
            m[f"turn{t + 1}.unknown_inst"] = (sum(x["unknown"] for x in T), "n")
        return Result(m, samples)


class BFCL(_Gen):
    """BFCL (Berkeley Function Calling Leaderboard), categorie a chiamata singola e 'irrelevance',
    con i dati del pacchetto bfcl-eval. Chiamate native via 'tools' dell'API OpenAI.
    Il controllo degli argomenti è una versione semplificata del checker AST ufficiale."""
    kind = "bfcl"
    VERSION = "1"

    TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string", "list": "array",
                "int": "integer", "bool": "boolean", "str": "string"}

    def _data_dir(self):
        import importlib.util
        spec = importlib.util.find_spec("bfcl_eval")  # non importa il pacchetto: servono solo i dati
        if spec is None or not spec.submodule_search_locations:
            raise LabError("missing_dep", "manca bfcl-eval: esegui ./lab/setup.sh")
        return Path(list(spec.submodule_search_locations)[0]) / "data"

    def data_fingerprint(self):
        return f"bfcl_eval={lib_version('bfcl-eval')}:{self.cfg.get('categories')}"

    def items(self):
        d = self._data_dir()
        out = []
        for cat in self.cfg.get("categories", ["simple_python", "irrelevance"]):
            qf = next(iter(sorted(d.glob(f"BFCL_v*_{cat}.json"))), None)
            if not qf:
                raise LabError("missing_data", f"BFCL: categoria {cat} non trovata in {d}")
            qs = [json.loads(line) for line in qf.open()]
            af = d / "possible_answer" / qf.name
            ans = {}
            if af.exists():
                for line in af.open():
                    a = json.loads(line)
                    ans[a["id"]] = a["ground_truth"]
            for q in sample(qs, self.cfg.get("limit_per_category", self.cfg.get("limit")), self.cfg.get("seed", 1234)):
                out.append({"cat": cat, "q": q, "gt": ans.get(q["id"])})
        return out

    def prepare(self, log):
        self.items()

    def _fix(self, schema):
        if isinstance(schema, dict):
            s = {k: self._fix(v) for k, v in schema.items()}
            if isinstance(s.get("type"), str):
                s["type"] = self.TYPE_MAP.get(s["type"], s["type"])
            return s
        if isinstance(schema, list):
            return [self._fix(x) for x in schema]
        return schema

    @staticmethod
    def _eq(v, allowed):
        for a in allowed:
            if a == "" and v is None:
                return True
            if isinstance(a, (int, float)) and isinstance(v, (int, float)) and not isinstance(v, bool):
                if abs(float(a) - float(v)) < 1e-6:
                    return True
            elif isinstance(a, str) and isinstance(v, str):
                if a.strip().lower() == v.strip().lower():
                    return True
            elif a == v:
                return True
        return False

    def run(self, rt):
        def one(cl, it):
            q = it["q"]
            name_map = {}
            tools = []
            for f in q["function"]:
                safe = re.sub(r"[^a-zA-Z0-9_-]", "_", f["name"])
                name_map[safe] = f["name"]
                tools.append({"type": "function", "function": {"name": safe, "description": f.get("description", ""),
                                                               "parameters": self._fix(f.get("parameters", {}))}})
            msgs = q["question"][0]
            r = cl.chat(msgs, max_tokens=self.max_tokens(rt, 512), tools=tools)
            calls = r.get("tool_calls") or []
            base = {"case_id": q["id"], "cat": it["cat"], "n_calls": len(calls), **speed_fields(r)}
            if it["cat"].endswith("irrelevance"):
                return {**base, "correct": float(len(calls) == 0)}
            if len(calls) != 1 or not it["gt"]:
                return {**base, "correct": 0.0, "why": "numero di chiamate"}
            fn = calls[0].get("function", {})
            name = name_map.get(fn.get("name"), fn.get("name"))
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                return {**base, "correct": 0.0, "why": "argomenti non JSON"}
            gt = it["gt"][0]
            gname, gargs = next(iter(gt.items()))
            if name != gname:
                return {**base, "correct": 0.0, "why": f"funzione {name}"}
            for k, allowed in gargs.items():
                if not self._eq(args.get(k), allowed):
                    return {**base, "correct": 0.0, "why": f"argomento {k}={args.get(k)!r}"}
            extra = set(args) - set(gargs)
            if extra:
                return {**base, "correct": 0.0, "why": f"argomenti inventati {sorted(extra)}"}
            return {**base, "correct": 1.0}

        samples = self.loop(rt, self.items(), one)
        m = {"acc_pct": self.acc(samples), **speed_summary(samples)}
        for cat in self.cfg.get("categories", []):
            S = [s for s in samples if s.get("cat") == cat]
            if S:
                m[f"{cat}.acc_pct"] = self.acc(S)
        return Result(m, samples)
