"""Test finto per i collaudi dell'orchestratore (nessun dataset, nessun modello vero)."""
from __future__ import annotations

from ..client import speed_fields
from .base import QUALITY, Result, Test, check_errors, speed_summary


class MockQuality(Test):
    kind = "mock_quality"
    VERSION = "1"
    default_depends = QUALITY

    def run(self, rt):
        cl = rt.client()
        samples = []
        for i in range(int(self.cfg.get("n", 5))):
            try:
                r = cl.chat([{"role": "user", "content": f"quanto fa {i}+1?"}], max_tokens=8)
                samples.append({"case_id": str(i), "correct": float(r["content"].strip() == str(i + 1)),
                                **speed_fields(r)})
            except Exception as e:
                samples.append({"case_id": str(i), "error": str(e)})
        check_errors(samples)
        acc = 100 * sum(s.get("correct") or 0 for s in samples) / len(samples)
        return Result({"acc_pct": (acc, "%"), **speed_summary(samples)}, samples)
