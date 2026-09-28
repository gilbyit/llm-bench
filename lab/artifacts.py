"""Risoluzione e download dei pesi dei modelli (Hugging Face, solo stdlib), con SHA256.

Un artefatto è identificato da (modello, quantizzazione). Lo SHA256 entra nell'impronta dei
risultati: se il file cambia (nuovo upload dell'autore, quantizzazione rifatta), i test che ne
dipendono tornano da eseguire, gli altri no.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .util import LabError, h, sha256_file

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


@dataclass
class Ref:
    model: str
    quant: str
    fmt: str
    source: str                     # repo HF o percorso locale
    files: list = field(default_factory=list)  # [{path, size, sha256}] relativi al repo
    size_bytes: int | None = None
    sha256: str | None = None       # sha del file principale (o combinato se split)
    local: Path | None = None       # file principale sul disco
    make: dict | None = None        # ricetta di quantizzazione locale
    revision: str | None = None

    @property
    def size_gb(self):
        return round(self.size_bytes / 1e9, 3) if self.size_bytes else None


def _req(url, timeout=60, headers=None):
    hdr = {"User-Agent": "gilpa-lab/1.0"}
    if os.environ.get("HF_TOKEN"):
        hdr["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    hdr.update(headers or {})
    return urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout)


def _quant_regex(q: str):
    return re.compile(r"(?i)(^|[-_.])" + re.escape(q) + r"([-_.]|$)")


class Artifacts:
    def __init__(self, cfg, store, data_dir: Path):
        self.cfg, self.store = cfg, store
        self.models_dir = data_dir / "models"
        self.cache_dir = data_dir / "hf_cache"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --- listing HF ------------------------------------------------------------------------
    def hf_tree(self, repo: str, revision: str = "main", refresh: bool = False) -> dict:
        cache = self.cache_dir / (repo.replace("/", "__") + f"@{revision}.json")
        if cache.exists() and not refresh:
            return json.loads(cache.read_text())
        try:
            with _req(f"{HF}/api/models/{repo}/revision/{revision}") as r:
                sha = json.loads(r.read()).get("sha")
            with _req(f"{HF}/api/models/{repo}/tree/{revision}?recursive=1") as r:
                items = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise LabError("download", f"HF {repo}@{revision}: HTTP {e.code}")
        except Exception as e:
            raise LabError("download", f"HF {repo}@{revision}: {e}")
        files = [{"path": i["path"], "size": i.get("size"),
                  "sha256": (i.get("lfs") or {}).get("oid")}
                 for i in items if i.get("type") == "file"]
        data = {"repo": repo, "revision": sha or revision, "files": files, "fetched": time.time()}
        cache.write_text(json.dumps(data))
        return data

    # --- risoluzione -----------------------------------------------------------------------
    def resolve(self, model_id: str, quant_id: str) -> Ref:
        m = self.cfg["models"][model_id]
        q = self.cfg["quants"][quant_id]
        over = (m.get("quants") or {}).get(quant_id) or {}
        fmt = q["format"]

        if over.get("local_path"):
            p = Path(over["local_path"])
            return self._local_ref(model_id, quant_id, fmt, p)

        make = over.get("make") or q.get("make")
        if make:
            base = self.resolve(model_id, make["from"])
            params_b = m.get("params_b") or 1
            est = int(params_b * 1e9 * q.get("bits", 4.8) / 8)
            recipe = {"base_sha": base.sha256, "from": make["from"], "type": make["type"],
                      "imatrix": make.get("imatrix", False)}
            recipe["key"] = f"make:{model_id}:{quant_id}:{h(recipe)}"
            ref = Ref(model_id, quant_id, fmt, f"make:{base.source}", size_bytes=est,
                      sha256=None, local=self.models_dir / model_id / f"{quant_id}.gguf", make=recipe)
            row = self.store.get_artifact(recipe["key"])
            if row:
                ref.sha256, ref.size_bytes = row["sha256"], row["size_bytes"]
            return ref

        src = (m.get("sources") or {}).get(q.get("source", fmt))
        if not src:
            raise LabError("unsupported", f"{model_id}: nessuna sorgente per il formato '{fmt}'")
        repo = over.get("repo") or src["repo"]
        rev = src.get("revision", "main")

        if fmt in ("gguf", "gguf-bitnet"):
            tree = self.hf_tree(repo, rev)
            files = self._pick_gguf(tree["files"], over.get("file") or q.get("file") or quant_id)
            if not files:
                raise LabError("download", f"{repo}: nessun file per la quantizzazione {quant_id}")
            main = files[0]
            size = sum(f["size"] or 0 for f in files)
            sha = main["sha256"] if len(files) == 1 else h([f["sha256"] for f in files], 64)
            local = self.models_dir / model_id / Path(main["path"]).name
            return Ref(model_id, quant_id, fmt, repo, files, size, sha, local, revision=tree["revision"])

        # Formati gestiti dal motore (OpenVINO, ONNX, safetensors): identifichiamo la revisione del repo
        tree = self.hf_tree(repo, rev)
        size = sum(f["size"] or 0 for f in tree["files"]
                   if f["path"].endswith((".bin", ".safetensors", ".onnx", ".onnx.data", ".xml")))
        return Ref(model_id, quant_id, fmt, repo, [], size or None, tree["revision"],
                   self.models_dir / model_id / quant_id, revision=tree["revision"])

    def _pick_gguf(self, files, pattern: str):
        ggufs = [f for f in files if f["path"].lower().endswith(".gguf") and "mmproj" not in f["path"].lower()]
        if any(c in pattern for c in "*?["):
            cand = [f for f in ggufs if fnmatch.fnmatch(f["path"].lower(), pattern.lower())]
        else:
            rx = _quant_regex(pattern)
            cand = [f for f in ggufs if rx.search(Path(f["path"]).stem)]
            if not pattern.upper().startswith("UD-"):
                cand = [f for f in cand if "UD-" not in f["path"]]
        if not cand:
            return []
        split = [f for f in cand if re.search(r"-\d{5}-of-\d{5}\.gguf$", f["path"])]
        if split:
            first = sorted(split, key=lambda f: f["path"])[0]
            prefix = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", first["path"])
            return sorted([f for f in split if f["path"].startswith(prefix)], key=lambda f: f["path"])
        return [sorted(cand, key=lambda f: len(f["path"]))[0]]

    def _local_ref(self, model_id, quant_id, fmt, p: Path) -> Ref:
        if not p.exists():
            raise LabError("download", f"file locale mancante: {p}")
        st = p.stat()
        key = f"local:{p}:{st.st_size}:{int(st.st_mtime)}"
        row = self.store.get_artifact(key)
        sha = row["sha256"] if row else None
        if not sha:
            sha = sha256_file(p) if p.is_file() else h(sorted(str(x) for x in p.rglob("*")), 64)
            self.store.save_artifact(key, "local", str(p), p, st.st_size, sha)
        return Ref(model_id, quant_id, fmt, str(p), [], st.st_size, sha, p)

    # --- download --------------------------------------------------------------------------
    def ensure(self, ref: Ref, log, quantize_fn=None) -> Ref:
        """Porta l'artefatto sul disco. Per i formati gestiti dal motore non fa nulla."""
        if ref.make:
            return self._make(ref, log, quantize_fn)
        if ref.fmt not in ("gguf", "gguf-bitnet") or not ref.files:
            return ref
        for f in ref.files:
            dest = ref.local.parent / Path(f["path"]).name
            self._download(ref.source, ref.revision or "main", f, dest, log)
        return ref

    def _download(self, repo, rev, f, dest: Path, log):
        key = f"hf:{repo}:{f['path']}"
        row = self.store.get_artifact(key)
        if dest.exists() and dest.stat().st_size == f["size"] and row and row["sha256"] == f["sha256"]:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        url = f"{HF}/{repo}/resolve/{rev}/{urllib.parse.quote(f['path'])}"
        for attempt in range(5):
            have = part.stat().st_size if part.exists() else 0
            try:
                hdr = {"Range": f"bytes={have}-"} if have else {}
                with _req(url, timeout=120, headers=hdr) as r, open(part, "ab" if have else "wb") as out:
                    shutil.copyfileobj(r, out, 1 << 22)
                break
            except Exception as e:
                log(f"download {f['path']}: {e} (tentativo {attempt + 1}/5)")
                time.sleep(10 * (attempt + 1))
        else:
            raise LabError("download", f"download fallito: {repo}/{f['path']}")
        log(f"verifico sha256 di {dest.name}...")
        sha = sha256_file(part)
        if f["sha256"] and sha != f["sha256"]:
            part.unlink(missing_ok=True)
            raise LabError("download", f"sha256 non corrisponde per {f['path']}")
        part.rename(dest)
        self.store.save_artifact(key, "gguf", repo, dest, dest.stat().st_size, sha, {"revision": rev})

    def _make(self, ref: Ref, log, quantize_fn) -> Ref:
        """Quantizzazione locale da un file base (es. con/senza imatrix), con llama-quantize."""
        key = ref.make["key"]
        row = self.store.get_artifact(key)
        if row and Path(row["path"]).exists():
            ref.sha256, ref.size_bytes, ref.local = row["sha256"], row["size_bytes"], Path(row["path"])
            return ref
        if quantize_fn is None:
            raise LabError("unsupported", "quantizzazione locale senza un motore llama.cpp nativo")
        base = self.ensure(self.resolve(ref.model, ref.make["from"]), log)
        imatrix = None
        if ref.make.get("imatrix"):
            imatrix = self._imatrix(ref.model, log)
        ref.local.parent.mkdir(parents=True, exist_ok=True)
        quantize_fn(base.local, ref.local, ref.make["type"], imatrix)
        ref.sha256 = sha256_file(ref.local)
        ref.size_bytes = ref.local.stat().st_size
        self.store.save_artifact(key, "gguf-made", ref.source, ref.local, ref.size_bytes, ref.sha256, ref.make)
        return ref

    def _imatrix(self, model_id, log) -> Path:
        m = self.cfg["models"][model_id]
        src = (m.get("sources") or {}).get("imatrix") or (m.get("sources") or {}).get("gguf")
        tree = self.hf_tree(src["repo"])
        cand = [f for f in tree["files"] if "imatrix" in f["path"].lower()]
        if not cand:
            raise LabError("download", f"{src['repo']}: nessun file imatrix")
        f = cand[0]
        dest = self.models_dir / model_id / Path(f["path"]).name
        self._download(src["repo"], tree["revision"], f, dest, log)
        return dest

    def delete_local(self, ref: Ref):
        if ref.local and ref.local.exists() and ref.fmt.startswith("gguf") and not ref.source.startswith("/"):
            for f in ref.files or [{"path": ref.local.name}]:
                (ref.local.parent / Path(f["path"]).name).unlink(missing_ok=True)
