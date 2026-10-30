from __future__ import annotations

import pytest

from tap.assets.cache import MemoryCache, cache_key
from tap.assets.reuse import ReuseContext, ReuseService
from tap.assets.semantic import SemanticIndex
from tap.assets.terminology import TermChange, Terminology, parse_version
from tap.assets.tm import TranslationMemory
from tap.db import Database
from tap.errors import ValidationFailed
from tap.observability import Metrics


@pytest.fixture
def stack():
    db = Database()
    tm, terms = TranslationMemory(db), Terminology(db)
    reuse = ReuseService(MemoryCache(), tm, SemanticIndex(db), Metrics())
    return tm, terms, reuse


def test_cache_key_is_tenant_domain_and_version_scoped():
    base = ("Hello  world", "en>de", "ui", "acme", "v7", "s1")
    assert cache_key(*base) == cache_key("Hello world", *base[1:])  # whitespace-normalised
    assert cache_key(*base) != cache_key(*base[:3], "beta", *base[4:])
    assert cache_key(*base) != cache_key(*base[:4], "v8", base[5])
    assert cache_key(*base) != cache_key(base[0], base[1], "clinical", *base[3:])
    assert ":acme:" in cache_key(*base)


def test_memory_cache_lru_and_ttl():
    now = [0.0]
    c = MemoryCache(max_items=2, clock=lambda: now[0])
    c.set("a", "1")
    c.set("b", "2")
    c.get("a")
    c.set("c", "3")  # noqa: E702
    assert c.get("b") is None and c.get("a") == "1" and c.get("c") == "3"
    c.set("t", "x", ttl_s=10)
    now[0] = 11
    assert c.get("t") is None


def test_tm_fuzzy_search_flags_number_differences(stack):
    tm, _, _ = stack
    tm.add("acme", "en", "de", "clinical", "Participants must report any serious adverse event within 48 hours.", "…48 Stunden melden.")
    tm.add("beta", "en", "de", "clinical", "Participants must report any serious adverse event within 24 hours.", "other tenant")
    hits = tm.search("acme", "en", "de", "Participants must report any serious adverse event within 24 hours.", min_pct=75)
    assert len(hits) == 1 and hits[0].pct >= 95 and hits[0].numbers_differ  # and never leaks beta's entry
    assert tm.search("acme", "en", "de", "Completely unrelated sentence about parking.", min_pct=75) == []


def test_tm_upsert_and_validation(stack):
    tm, _, _ = stack
    a = tm.add("acme", "en", "de", "ui", "Save", "Speichern")
    b = tm.add("acme", "en", "de", "ui", "Save", "Sichern")
    assert a == b and tm.count("acme") == 1
    assert tm.exact("acme", "en", "de", "ui", "v0", "Save").target == "Sichern"
    with pytest.raises(ValidationFailed):
        tm.add("acme", "en", "de", "ui", "  ", "x")


def test_tmx_roundtrip(stack):
    tm, _, _ = stack
    tm.add("acme", "en", "de", "ui", "Save <b>now</b> & go", "Jetzt <b>speichern</b> & los")
    xml = tm.export_tmx("acme", "en", "de")
    other = TranslationMemory(Database())
    assert other.import_tmx("acme", xml) == {"added": 1, "skipped": 0}
    assert other.exact("acme", "en", "de", "ui", "v0", "Save <b>now</b> & go").target == "Jetzt <b>speichern</b> & los"
    with pytest.raises(ValidationFailed):
        other.import_tmx("acme", "<tmx><unclosed>")


def test_glossary_versions_are_point_in_time(stack):
    _, terms, _ = stack
    v1 = terms.commit("acme", [TermChange("add", "en", "de", "IMP", "Prüfpräparat")])
    v2 = terms.commit("acme", [TermChange("add", "en", "de", "IMP", "Studienmedikation")])
    assert (v1, v2) == (1, 2)
    assert [t.translation for t in terms.terms_at("acme", "en", "de", 1)] == ["Prüfpräparat"]
    assert [t.translation for t in terms.terms_at("acme", "en", "de")] == ["Studienmedikation"]
    v3 = terms.commit("acme", [TermChange("remove", "en", "de", "IMP")])
    assert terms.terms_at("acme", "en", "de", v3) == [] and terms.terms_at("acme", "en", "de", 2)
    assert terms.terms_at("beta", "en", "de") == []
    with pytest.raises(ValidationFailed):
        terms.commit("acme", [])
    assert parse_version("v7", 3) == 7 and parse_version(None, 3) == 3
    with pytest.raises(ValidationFailed):
        parse_version("7", 3)


def test_term_change_impact_lists_affected_tm_entries(stack):
    tm, terms, _ = stack
    tm.add("acme", "en", "de", "clinical", "The IMP is stored cold.", "x")
    tm.add("acme", "en", "de", "clinical", "Unrelated sentence.", "y")
    assert [e["source"] for e in terms.impact("acme", tm, "en", "de", ["IMP"])] == ["The IMP is stored cold."]


def test_reuse_tiers_and_glossary_version_invalidates_exact(stack):
    tm, _, reuse = stack
    ctx7 = ReuseContext("acme", "en", "de", "clinical", "v7")
    reuse.store("Store below 25 degrees C.", "Unter 25 °C lagern.", ctx7, origin="human", approved_by="lin")
    assert reuse.lookup("Store below 25 degrees C.", ctx7).type == "exact"
    reuse.cache._data.clear()  # type: ignore[attr-defined]
    assert reuse.lookup("Store below 25 degrees C.", ctx7).type == "exact"  # backfilled from durable TM
    hit8 = reuse.lookup("Store below 25 degrees C.", ReuseContext("acme", "en", "de", "clinical", "v8"))
    assert hit8.type == "fuzzy" and hit8.pct == 100 and hit8.context_mismatch  # never a silent exact hit across versions
    assert reuse.lookup("Store below 25 degrees C.", ReuseContext("beta", "en", "de", "clinical", "v7")).type == "none"


def test_semantic_suggestions_are_context_only(stack):
    tm, _, reuse = stack
    tm.add("acme", "en", "de", "clinical", "Participants must report any serious adverse event within 48 hours.", "x")
    h = reuse.lookup("Participants should tell staff about adverse events quickly.", ReuseContext("acme", "en", "de", "clinical", "v1"))
    assert h.type in ("semantic", "none") and h.target is None


def test_redis_cache_adapter_roundtrip_ttl_and_ping(monkeypatch):
    import fakeredis

    from tap.assets.cache import RedisCache

    server = fakeredis.FakeServer()
    monkeypatch.setattr("redis.Redis.from_url", lambda url, **kw: fakeredis.FakeRedis(server=server, **kw))
    cache = RedisCache("redis://fake:6379/0", prefix="t:")
    cache.set("k", "v", ttl_s=60)
    assert cache.get("k") == "v" and cache.get("missing") is None and cache.ping()
    assert 0 < fakeredis.FakeRedis(server=server).ttl("t:k") <= 60  # keys are prefixed and expire
    cache.delete("k")
    assert cache.get("k") is None
