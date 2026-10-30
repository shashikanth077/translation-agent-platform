"""Model-based evaluators (extra ``ml``): COMET-Kiwi quality estimation and LaBSE semantic similarity.

Both lazy-load heavy models on first use. Pin model versions, accept the checkpoint licence where required
(COMET-Kiwi is gated), run on GPU for production throughput, and calibrate scores per language pair.
"""

from __future__ import annotations

import threading
from typing import Any

from ..parse.tokens import numbers_and_units
from .base import EvalResult, ScoreRequest, Slot, clamp


class CometKiwiEvaluator:  # pragma: no cover - needs optional heavy dependencies and a model download
    slot: Slot = "comet"

    def __init__(self, model_name: str = "Unbabel/wmt22-cometkiwi-da", gpus: int = 0, batch_size: int = 16) -> None:
        self.name = f"cometkiwi:{model_name}"
        self.version = model_name
        self._model_name, self._gpus, self._batch = model_name, gpus, batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from comet import download_model, load_from_checkpoint

                self._model = load_from_checkpoint(download_model(self._model_name))
            return self._model

    def evaluate(self, req: ScoreRequest) -> EvalResult:
        model = self._load()
        out = model.predict([{"src": req.source, "mt": req.target}], batch_size=self._batch, gpus=self._gpus, progress_bar=False)
        return EvalResult(float(out.scores[0]))


class LabseSemanticEvaluator:  # pragma: no cover - needs optional heavy dependencies and a model download
    slot: Slot = "semantic"
    name = "labse+numbers"
    version = "LaBSE"

    def __init__(self, model_name: str = "sentence-transformers/LaBSE") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._lock = threading.Lock()

    def evaluate(self, req: ScoreRequest) -> EvalResult:
        from sentence_transformers import util

        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
        sim = float(util.cos_sim(self._model.encode(req.source), self._model.encode(req.target)))
        findings: list[str] = []
        if numbers_and_units(req.source)[0] != numbers_and_units(req.target)[0]:
            findings.append("numbers differ between source and translation")
            sim = min(sim, 0.4)  # hard cap: embeddings are blind to a changed dose
        return EvalResult(clamp(sim), findings)
