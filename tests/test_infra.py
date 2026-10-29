from __future__ import annotations

import pytest

from tap.db import Database
from tap.events import EventBus
from tap.services.audit import AuditLog


def test_events_delivered_once_per_consumer_and_in_order():
    db, seen = Database(), []
    bus = EventBus(db)
    bus.subscribe("a", "job.*", lambda e: seen.append(("a", e.payload["n"])))
    bus.subscribe("b", "job.created", lambda e: seen.append(("b", e.payload["n"])))
    for n in range(3):
        bus.publish("job.created", {"n": n})
    bus.publish("other.topic", {"n": 99})
    assert bus.flush() == 4 and bus.flush() == 0
    assert [x for x in seen if x[0] == "a"] == [("a", 0), ("a", 1), ("a", 2)]
    assert len([x for x in seen if x[0] == "b"]) == 3


def test_failed_consumer_is_retried_then_dead_lettered_without_blocking_others():
    db, ok, calls = Database(), [], []
    bus = EventBus(db, max_attempts=3)

    def bad(e):  # noqa: ANN001
        calls.append(1)
        raise RuntimeError("down")

    bus.subscribe("bad", "x", bad)
    bus.subscribe("good", "x", lambda e: ok.append(e.id))
    bus.publish("x", {})
    for _ in range(5):
        bus.flush()
    assert len(calls) == 3 and len(ok) == 1  # retried up to max_attempts, good consumer ran exactly once
    assert bus.pending_count() == 0 and bus.dead_letters()[0]["consumer"] == "bad"


def test_publish_is_atomic_with_the_callers_transaction():
    db = Database()
    bus = EventBus(db)
    db.execute("CREATE TABLE t (x)")
    with pytest.raises(RuntimeError), db.tx():
        db.execute("INSERT INTO t VALUES (1)")
        bus.publish("x", {})
        raise RuntimeError("rollback")
    assert db.scalar("SELECT COUNT(*) FROM t") == 0 and bus.pending_count() == 0


def test_handlers_may_publish_follow_up_events():
    db, seen = Database(), []
    bus = EventBus(db)
    bus.subscribe("one", "a", lambda e: bus.publish("b", {}))
    bus.subscribe("two", "b", lambda e: seen.append(1))
    bus.publish("a", {})
    bus.flush()
    assert seen == [1]


def test_relay_thread_delivers_in_background():
    import time

    db, seen = Database(), []
    bus = EventBus(db)
    bus.subscribe("c", "*", lambda e: seen.append(e.topic))
    bus.start_relay(0.01)
    bus.publish("hello", {})
    deadline = time.time() + 3
    while not seen and time.time() < deadline:
        time.sleep(0.01)
    bus.stop_relay()
    assert seen == ["hello"]


def test_nested_transactions_join_the_outer_one():
    db = Database()
    db.execute("CREATE TABLE t (x)")
    with pytest.raises(RuntimeError), db.tx():
        db.execute("INSERT INTO t VALUES (1)")
        with db.tx():
            db.execute("INSERT INTO t VALUES (2)")
        raise RuntimeError
    assert db.scalar("SELECT COUNT(*) FROM t") == 0


def test_migrations_apply_once_and_incrementally():
    db = Database()
    db.migrate("svc", ["CREATE TABLE a (x)"])
    db.migrate("svc", ["CREATE TABLE a (x)", "CREATE TABLE b (y)"])  # only the new step runs
    db.migrate("svc", ["CREATE TABLE a (x)", "CREATE TABLE b (y)"])
    assert db.scalar("SELECT version FROM schema_migrations WHERE service='svc'") == 2


def test_audit_chain_detects_tampering_deletion_and_reordering():
    db = Database()
    audit = AuditLog(db)
    for i in range(5):
        audit.append("u", f"act{i}", tenant_id="acme", subject="s", details={"i": i})
    assert audit.verify() == {"ok": True, "checked": 5, "head": audit.verify()["head"]}
    db.execute("UPDATE audit_log SET actor='mallory' WHERE seq=3")
    assert audit.verify()["first_bad_seq"] == 3
    db.execute("UPDATE audit_log SET actor='u' WHERE seq=3")
    assert audit.verify()["ok"]
    db.execute("DELETE FROM audit_log WHERE seq=2")
    assert not audit.verify()["ok"]


def test_audit_search_is_filtered():
    audit = AuditLog(Database())
    audit.append("u", "a", tenant_id="acme", subject="x")
    audit.append("u", "b", tenant_id="beta", subject="x")
    assert [r["action"] for r in audit.search(tenant_id="acme")] == ["a"]


def test_dead_letters_can_be_requeued_after_the_consumer_is_fixed():
    db, ok = Database(), []
    bus = EventBus(db, max_attempts=2)
    broken = {"on": True}

    def flaky(e):  # noqa: ANN001
        if broken["on"]:
            raise RuntimeError("down")
        ok.append(e.id)

    bus.subscribe("c", "x", flaky)
    bus.publish("x", {})
    for _ in range(4):
        bus.flush()
    assert bus.dead_letters() and not ok
    broken["on"] = False
    assert bus.requeue_dead() == 1
    bus.flush()
    assert len(ok) == 1 and not bus.dead_letters()
