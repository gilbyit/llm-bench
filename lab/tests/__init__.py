from .gilpa import GilpaIntent
from .generative import BFCL, Belebele, EvalitaSA, JsonSchemaConstrained, MultiIF
from .kld import KLD
from .lmeval import LmEval
from .mock import MockQuality
from .speed import LlamaBench, LocalScore

KINDS = {c.kind: c for c in (GilpaIntent, BFCL, Belebele, EvalitaSA, JsonSchemaConstrained, MultiIF, KLD,
                             LmEval, MockQuality, LlamaBench, LocalScore)}


def build(name: str, cfg: dict, lab):
    kind = cfg.get("kind", name)
    if kind not in KINDS:
        raise SystemExit(f"test {name}: kind '{kind}' sconosciuto ({', '.join(sorted(KINDS))})")
    return KINDS[kind](name, cfg, lab)
