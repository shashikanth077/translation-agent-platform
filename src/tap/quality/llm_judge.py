"""LLM-as-judge evaluators (semantic accuracy, fluency) with structured verdicts.

LLM judges are biased (verbosity, self-preference) and can be fooled by the text they judge. Mitigations here:
the judged text is wrapped as data, the verdict is a forced tool call with a numeric score and concrete
findings, and the consensus combines the judge with COMET-style and deterministic signals instead of trusting it alone.
"""

from __future__ import annotations

from ..engines import prompts
from ..engines.llm_client import LLMClient
from .base import EvalResult, ScoreRequest, Slot, clamp

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "findings": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
    },
    "required": ["score", "findings"],
    "additionalProperties": False,
}


class LLMJudge:
    def __init__(self, slot: Slot, client: LLMClient, *, model: str | None = None) -> None:
        if slot not in ("semantic", "fluency"):
            raise ValueError("LLM judges fill the semantic or fluency slot")
        self.slot: Slot = slot
        self.client = client
        self.model = model
        self.name = f"llm-judge-{slot}"
        self.version = prompts.JUDGE_SEMANTIC_V1 if slot == "semantic" else prompts.JUDGE_FLUENCY_V1

    def evaluate(self, req: ScoreRequest) -> EvalResult:
        rubric = prompts.JUDGE_SEMANTIC if self.slot == "semantic" else prompts.JUDGE_FLUENCY
        user = (
            rubric.format(src=req.src_lang, tgt=req.tgt_lang) + f"\nDomain: {req.domain}\n"
            f"<source>{req.source}</source>\n<translation>{req.target}</translation>"
        )
        resp = self.client.call_tool(
            system=prompts.JUDGE_SYSTEM,
            user=user,
            tool_name="submit_verdict",
            tool_description="Submit the quality verdict.",
            schema=VERDICT_SCHEMA,
            max_tokens=400,
            model=self.model,
        )
        findings = [str(f) for f in resp.data.get("findings", [])][:10]
        return EvalResult(clamp(float(resp.data.get("score", 0.0))), findings)
