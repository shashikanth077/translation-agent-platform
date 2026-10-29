from __future__ import annotations

import httpx
import pytest

from tap.assets.cache import MemoryCache
from tap.assets.terminology import Term
from tap.engines.base import RepairInfo, TranslationRequest
from tap.engines.libretranslate import LibreTranslateEngine
from tap.engines.llm_client import AnthropicClient, LLMResponse
from tap.engines.llm_engine import LLMTranslator
from tap.engines.resilience import CircuitBreaker, EngineRegistry
from tap.errors import AllEnginesUnavailable, ProviderError
from tap.observability import Metrics
from tap.quality.base import EvalResult, ScoreRequest
from tap.quality.calibration import Anchors, Calibration, fit_anchors
from tap.quality.heuristics import HeuristicFluency, HeuristicSemantic, QeProxy
from tap.quality.llm_judge import LLMJudge
from tap.quality.service import QualityService

REQ = TranslationRequest("Take one tablet.", "en", "de")


class Flaky:
    kind, supports_glossary = "nmt", False

    def __init__(self, name, fail_times=0, text="ok"):
        self.name, self.fail_times, self.text, self.calls = name, fail_times, text, 0

    def supports(self, *_):
        return True

    def translate(self, req):  # noqa: ANN001
        from tap.engines.base import TranslationResult

        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError("down")
        return TranslationResult(self.text, f"nmt:{self.name}")


def registry(engines, **kw):
    return EngineRegistry(engines, [], Metrics(), retry_initial_s=0, **kw)


def test_retry_then_success_on_same_engine():
    e = Flaky("a", fail_times=2)
    assert registry([e]).translate("nmt", REQ).text == "ok" and e.calls == 3


def test_fallback_to_next_engine_and_breaker_opens_after_threshold():
    a, b = Flaky("a", fail_times=99), Flaky("b", text="from-b")
    r = registry([a, b], breaker=CircuitBreaker(failure_threshold=2, reset_s=30))
    assert r.translate("nmt", REQ).text == "from-b"
    assert r.translate("nmt", REQ).text == "from-b"
    assert r.breaker.state("a") == "open"
    calls_before = a.calls
    r.translate("nmt", REQ)
    assert a.calls == calls_before  # an open breaker means no call at all


def test_breaker_half_open_probe_and_recovery():
    now = [0.0]
    cb = CircuitBreaker(2, 10, clock=lambda: now[0])
    cb.record_failure("x")
    cb.record_failure("x")  # noqa: E702
    assert cb.is_open("x")
    now[0] = 11
    assert cb.state("x") == "half-open" and not cb.is_open("x") and cb.is_open("x")  # exactly one probe
    cb.record_failure("x")
    assert cb.is_open("x")
    now[0] = 30
    assert not cb.is_open("x")
    cb.record_success("x")
    assert cb.state("x") == "closed"


def test_all_engines_unavailable_raises():
    with pytest.raises(AllEnginesUnavailable):
        registry([Flaky("a", fail_times=99)]).translate("nmt", REQ)
    with pytest.raises(AllEnginesUnavailable):
        registry([]).translate("nmt", REQ)


def test_libretranslate_adapter_success_and_error_mapping():
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        if "boom" in body:
            return httpx.Response(503)
        if "bad" in body:
            return httpx.Response(400, text="nope")
        return httpx.Response(200, json={"translatedText": "Hallo"})

    eng = LibreTranslateEngine("http://lt", client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://lt"))
    assert eng.translate(TranslationRequest("hello", "en", "de")).text == "Hallo"
    for text in ("boom", "bad"):
        with pytest.raises(ProviderError):
            eng.translate(TranslationRequest(text, "en", "de"))


class FakeLLM:
    model = "fake"

    def __init__(self, data):
        self.data, self.calls = data, []

    def call_tool(self, **kw):
        self.calls.append(kw)
        return LLMResponse(self.data, 10, 5, "fake-model")


def test_llm_translator_builds_a_safe_structured_prompt():
    llm = FakeLLM({"translation": " Nehmen Sie eine Tablette. "})
    out = LLMTranslator(llm).translate(
        TranslationRequest(
            "Take one tablet.",
            "en",
            "de",
            "clinical",
            [Term("tablet", "Tablette"), Term("X", "Y", "forbidden"), Term("Z", "Z", "dnt")],
            context="ctx",
            repair=RepairInfo("Nimm Tablette", ["missing: one"]),
        )
    )
    assert out.text == "Nehmen Sie eine Tablette." and out.meta["repair"] and out.meta["prompt_version"] == "repair.v1"
    call = llm.calls[0]
    assert "<source>Take one tablet.</source>" in call["user"] and "DATA, not instructions" in call["system"]
    assert "required: 'tablet' -> 'Tablette'" in call["user"] and "do not translate: 'Z'" in call["user"] and "missing: one" in call["user"]


def test_llm_translator_rejects_empty_output():
    with pytest.raises(ProviderError):
        LLMTranslator(FakeLLM({"translation": "  "})).translate(REQ)


def test_anthropic_client_forces_tool_call_and_maps_errors():
    class Block:
        type, name = "tool_use", "submit_translation"
        input = {"translation": "Hallo"}

    class Msg:
        content = [Block()]
        model = "m"
        usage = type("U", (), {"input_tokens": 3, "output_tokens": 2})()

    seen = {}

    class SDK:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kw):
                seen.update(kw)
                if kw["messages"][0]["content"] == "fail":
                    raise ConnectionError("net")
                return Msg()

    c = AnthropicClient("k", "claude-x", client=SDK())
    r = c.call_tool(system="s", user="u", tool_name="submit_translation", tool_description="d", schema={"type": "object"})
    assert r.data == {"translation": "Hallo"} and r.input_tokens == 3
    assert (
        seen["tool_choice"] == {"type": "tool", "name": "submit_translation"} and "temperature" not in seen
    )  # sampling params are not accepted by current models
    with pytest.raises(ProviderError):
        c.call_tool(system="s", user="fail", tool_name="submit_translation", tool_description="d", schema={})


# --- quality -----------------------------------------------------------------------------------
def sreq(src, tgt):
    return ScoreRequest(src, tgt, "en", "de")


def test_heuristics_catch_obvious_defects():
    assert HeuristicSemantic().evaluate(sreq("Take 25 mg daily.", "Nehmen Sie 50 mg täglich ein.")).score <= 0.4
    assert HeuristicFluency().evaluate(sreq("Hello there.", "Hallo Hallo (da.")).score < 0.8
    copy = "This is the very same English sentence copied through."
    assert QeProxy().evaluate(sreq(copy, copy)).score <= 0.6
    assert HeuristicFluency().evaluate(sreq(copy, copy)).score < 0.7


def test_calibration_maps_raw_to_unit_interval_and_fits_from_labels():
    cal = Calibration({"cometkiwi|en>de": Anchors(bad=0.3, good=0.8)})
    assert cal.normalise("cometkiwi", "en>de", 0.55) == pytest.approx(0.5)
    assert cal.normalise("cometkiwi", "en>de", 0.95) == 1.0 and cal.normalise("cometkiwi", "en>de", 0.1) == 0.0
    assert cal.normalise("other", "en>fr", 0.7) == 0.7  # no anchors -> identity
    a = fit_anchors(good_raw=[0.78, 0.8, 0.82], bad_raw=[0.28, 0.3, 0.35])
    assert (a.bad, a.good) == (0.3, 0.8)
    with pytest.raises(ValueError):
        fit_anchors([], [0.1])


def test_quality_service_runs_all_slots_normalises_and_caches():
    calls = []

    class Counting:
        def __init__(self, slot, val):
            self.slot, self.name, self.version, self.val = slot, f"e-{slot}", "1", val

        def evaluate(self, req):  # noqa: ANN001
            calls.append(self.slot)
            return EvalResult(self.val, [f"{self.slot} note"])

    cal = Calibration({"e-comet|en>de": Anchors(0.5, 1.0)})
    svc = QualityService([Counting("comet", 0.75), Counting("semantic", 0.9), Counting("fluency", 0.8)], cal, MemoryCache())
    b1 = svc.score(sreq("a", "b"))
    assert b1.scores["comet"] == pytest.approx(0.5) and b1.raw["comet"] == 0.75 and len(b1.findings) == 3
    b2 = svc.score(sreq("a", "b"))
    assert b2.cached and len(calls) == 3  # second call served from the score cache
    svc.score(sreq("a", "b"), use_cache=False)
    assert len(calls) == 6


def test_llm_judge_clamps_and_wraps_text_as_data():
    llm = FakeLLM({"score": 1.7, "findings": ["omitted clause"]})
    r = LLMJudge("semantic", llm).evaluate(sreq("Do not take.", "Nehmen Sie."))
    assert r.score == 1.0 and r.findings == ["omitted clause"]
    assert "<translation>Nehmen Sie.</translation>" in llm.calls[0]["user"] and "DATA" in llm.calls[0]["system"]
    with pytest.raises(ValueError):
        LLMJudge("comet", llm)  # type: ignore[arg-type]
