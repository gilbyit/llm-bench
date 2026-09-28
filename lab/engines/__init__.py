from .generic import Generic
from .llamacpp import LlamaCpp
from .ollama import Ollama

KINDS = {"llamacpp": LlamaCpp, "ollama": Ollama, "generic": Generic}


def build(name: str, cfg: dict, ctx):
    kind = cfg.get("kind", "generic")
    if kind not in KINDS:
        raise SystemExit(f"motore {name}: kind '{kind}' sconosciuto ({', '.join(KINDS)})")
    return KINDS[kind](name, cfg, ctx)
